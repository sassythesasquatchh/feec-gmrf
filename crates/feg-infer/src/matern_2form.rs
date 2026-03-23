use crate::matern_1form::{
    build_matern_mass_inverse_1form, MaternMassInverse as Matern1FormMassInverse,
};
use crate::sparse::{add_sparse, diag_matrix, invert_diag, matrix_diag, scale_matrix};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr};
use formoniq::{
    assemble::assemble_whitney_2form_projected_sparse_inverse_galmat,
    problems::hodge_laplace::MixedGalmats,
};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

pub struct HodgeLaplacian2Form {
    pub mass_u: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternMassInverse {
    ExactTopDegreeDiagonalOrProjectedNc2,
}

impl Default for MaternMassInverse {
    fn default() -> Self {
        Self::ExactTopDegreeDiagonalOrProjectedNc2
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
}

pub fn build_hodge_laplacian_2form(
    topology: &Complex,
    metric: &MeshLengths,
) -> Result<HodgeLaplacian2Form, String> {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    build_hodge_laplacian_2form_from_galmats(topology, metric, &galmats)
}

pub fn build_hodge_laplacian_2form_from_galmats(
    topology: &Complex,
    metric: &MeshLengths,
    galmats: &MixedGalmats,
) -> Result<HodgeLaplacian2Form, String> {
    let mass_u = galmats.mass_u_csr();
    let codifdif_u = if galmats.codifdif_u().nrows() == 0 {
        FeecCsr::from(&FeecCoo::new(mass_u.nrows(), mass_u.ncols()))
    } else {
        FeecCsr::from(galmats.codifdif_u())
    };

    let laplacian = if galmats.mass_sigma().nrows() == 0 {
        codifdif_u
    } else {
        let mass_sigma = FeecCsr::from(galmats.mass_sigma());
        let sigma_inverse = build_matern_mass_inverse_1form(
            topology,
            metric,
            &mass_sigma,
            Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        );
        let dif_sigma = FeecCsr::from(galmats.dif_sigma());
        let codif_u = FeecCsr::from(galmats.codif_u());
        let schur_mid = &dif_sigma * &sigma_inverse;
        let schur = schur_mid * &codif_u;
        add_sparse(&codifdif_u, &schur)
    };

    Ok(HodgeLaplacian2Form { mass_u, laplacian })
}

pub fn build_matern_system_matrix_2form(hodge: &HodgeLaplacian2Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_matern_mass_inverse_2form(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecCsr, String> {
    match strategy {
        MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2 => match topology.dim() {
            2 => Ok(diag_matrix(&invert_diag(&matrix_diag(mass_u)))),
            3 => {
                let projected =
                    assemble_whitney_2form_projected_sparse_inverse_galmat(topology, metric);
                let projected = FeecCsr::from(&projected);
                if projected.nrows() != mass_u.nrows() || projected.ncols() != mass_u.ncols() {
                    return Err(format!(
                        "projected 2-form sparse inverse dimensions {}x{} do not match 2-form mass {}x{}",
                        projected.nrows(),
                        projected.ncols(),
                        mass_u.nrows(),
                        mass_u.ncols()
                    ));
                }
                Ok(projected)
            }
            dim => Err(format!(
                "2-form Matérn mass inverse is only implemented for intrinsic mesh dimensions 2 and 3, got {dim}"
            )),
        },
    }
}

pub fn build_matern_precision_2form_with_mass_inverse(
    hodge: &HodgeLaplacian2Form,
    mass_inverse: &FeecCsr,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    let a = build_matern_system_matrix_2form(hodge, kappa);
    let middle = mass_inverse * &a;

    let mut precision = &a * &middle;
    if (tau - 1.0).abs() > f64::EPSILON {
        precision = scale_matrix(&precision, tau * tau);
    }
    precision
}

pub fn build_matern_precision_2form(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian2Form,
    config: MaternConfig,
) -> Result<FeecCsr, String> {
    let mass_inverse =
        build_matern_mass_inverse_2form(topology, metric, &hodge.mass_u, config.mass_inverse)?;
    Ok(build_matern_precision_2form_with_mass_inverse(
        hodge,
        &mass_inverse,
        config.kappa,
        config.tau,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn diagonal_entries(mat: &FeecCsr) -> Vec<f64> {
        let mut diag = vec![0.0; mat.nrows()];
        for (row, col, value) in mat.triplet_iter() {
            if row == col {
                diag[row] += *value;
            }
        }
        diag
    }

    #[test]
    fn matern_precision_2form_top_degree_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("2-form Hodge Laplacian should assemble");
        let precision = build_matern_precision_2form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("2-form precision should build on a surface mesh");

        assert!(diagonal_entries(&precision)
            .iter()
            .all(|value| *value > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("2-form top-degree precision should factorize");
    }

    #[test]
    fn matern_precision_2form_3d_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_2form(&topology, &metric)
            .expect("3d 2-form Hodge Laplacian should assemble");
        let precision = build_matern_precision_2form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.25,
                tau: 1.0,
                mass_inverse: MaternMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
            },
        )
        .expect("3d 2-form precision should build");

        assert!(diagonal_entries(&precision)
            .iter()
            .all(|value| *value > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("3d 2-form precision should factorize");
    }
}
