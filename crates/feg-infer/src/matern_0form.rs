use crate::sparse::{add_sparse, diag_matrix, invert_diag, lumped_diag, scale_matrix};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use formoniq::problems::laplace_beltrami::LaplaceBeltramiGalmats;
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

pub struct LaplaceBeltrami0Form {
    pub mass: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternMassInverse {
    RowSumLumped,
}

impl Default for MaternMassInverse {
    fn default() -> Self {
        Self::RowSumLumped
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MaternConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mass_inverse: MaternMassInverse,
}

pub fn build_laplace_beltrami_0form(
    topology: &Complex,
    metric: &MeshLengths,
) -> LaplaceBeltrami0Form {
    let galmats = LaplaceBeltramiGalmats::compute(topology, metric);
    build_laplace_beltrami_0form_from_galmats(&galmats)
}

pub fn build_laplace_beltrami_0form_from_galmats(
    galmats: &LaplaceBeltramiGalmats,
) -> LaplaceBeltrami0Form {
    LaplaceBeltrami0Form {
        mass: galmats.mass_csr(),
        laplacian: galmats.stiffness_csr(),
    }
}

pub fn build_matern_system_matrix_0form(laplace: &LaplaceBeltrami0Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&laplace.laplacian, &scale_matrix(&laplace.mass, kappa2))
}

pub fn build_matern_mass_inverse_0form(mass: &FeecCsr, strategy: MaternMassInverse) -> FeecCsr {
    match strategy {
        MaternMassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(mass))),
    }
}

pub fn build_matern_precision_0form(
    laplace: &LaplaceBeltrami0Form,
    config: MaternConfig,
) -> FeecCsr {
    let a = build_matern_system_matrix_0form(laplace, config.kappa);
    let mass_inverse = build_matern_mass_inverse_0form(&laplace.mass, config.mass_inverse);
    let middle = &mass_inverse * &a;

    let mut precision = &a * &middle;
    if (config.tau - 1.0).abs() > f64::EPSILON {
        precision = scale_matrix(&precision, config.tau * config.tau);
    }
    precision
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
    fn laplace_beltrami_0form_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let laplace = build_laplace_beltrami_0form(&topology, &metric);

        assert!(laplace.mass.nrows() > 0);
        assert_eq!(laplace.mass.nrows(), laplace.mass.ncols());
        assert_eq!(laplace.laplacian.nrows(), laplace.mass.nrows());
        assert_eq!(laplace.laplacian.ncols(), laplace.mass.ncols());
    }

    #[test]
    fn matern_precision_0form_has_positive_diagonal_and_factorizes() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let laplace = build_laplace_beltrami_0form(&topology, &metric);
        let precision = build_matern_precision_0form(
            &laplace,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        assert!(diagonal_entries(&precision).iter().all(|v| *v > 0.0));
        feec_csr_to_gmrf(&precision)
            .cholesky_sqrt_lower()
            .expect("0-form precision should factorize");
    }
}
