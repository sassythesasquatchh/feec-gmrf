use crate::sparse::{add_sparse, diag_matrix, invert_diag, lumped_diag, scale_matrix};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::ExteriorField;
use formoniq::{
    assemble::assemble_whitney_projected_sparse_inverse_galmat,
    problems::hodge_laplace::MixedGalmats,
};
use manifold::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexHandleExt},
        },
        metric::mesh::MeshLengths,
    },
    topology::complex::Complex,
};

pub use crate::sparse::{feec_csr_to_gmrf, feec_vec_to_gmrf};

const RECONSTRUCTION_EPS: f64 = 1e-12;

pub struct HodgeLaplacian1Form {
    pub mass_u: FeecCsr,
    pub laplacian: FeecCsr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaternMassInverse {
    RowSumLumped,
    Nc1ProjectedSparseInverse,
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

#[derive(Debug, Clone)]
struct SparseRowLinearOperator {
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
}

#[derive(Debug, Clone)]
pub struct ReconstructedBarycenterFieldOperator {
    ambient_dim: usize,
    component_operators: Vec<SparseRowLinearOperator>,
}

#[derive(Debug, Clone)]
pub struct ReconstructedBarycenterField {
    ambient_dim: usize,
    component_values: Vec<FeecVector>,
}

impl SparseRowLinearOperator {
    fn new(ncols: usize, rows: Vec<Vec<(usize, f64)>>) -> Result<Self, String> {
        if rows
            .iter()
            .flatten()
            .any(|(col, value)| *col >= ncols || !value.is_finite())
        {
            return Err(
                "sparse row operator contains invalid column indices or non-finite values"
                    .to_string(),
            );
        }
        Ok(Self { ncols, rows })
    }

    fn nrows(&self) -> usize {
        self.rows.len()
    }

    fn apply(&self, input: &[f64]) -> Result<FeecVector, String> {
        if input.len() != self.ncols {
            return Err(format!(
                "operator input length {} does not match expected column count {}",
                input.len(),
                self.ncols
            ));
        }

        Ok(FeecVector::from_iterator(
            self.nrows(),
            self.rows.iter().map(|row| {
                row.iter()
                    .map(|(col, value)| *value * input[*col])
                    .sum::<f64>()
            }),
        ))
    }
}

impl ReconstructedBarycenterFieldOperator {
    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn cell_count(&self) -> usize {
        self.component_operators
            .first()
            .map_or(0, SparseRowLinearOperator::nrows)
    }

    pub fn apply_to_slice(&self, input: &[f64]) -> Result<ReconstructedBarycenterField, String> {
        let mut component_values = Vec::with_capacity(self.component_operators.len());
        for operator in &self.component_operators {
            component_values.push(operator.apply(input)?);
        }
        ReconstructedBarycenterField::from_components(component_values)
    }
}

impl ReconstructedBarycenterField {
    pub fn from_components(component_values: Vec<FeecVector>) -> Result<Self, String> {
        let Some(first) = component_values.first() else {
            return Err("at least one ambient component is required".to_string());
        };
        let cell_count = first.len();
        if component_values
            .iter()
            .any(|component| component.len() != cell_count)
        {
            return Err("all ambient components must have the same cell count".to_string());
        }

        Ok(Self {
            ambient_dim: component_values.len(),
            component_values,
        })
    }

    pub fn ambient_dim(&self) -> usize {
        self.ambient_dim
    }

    pub fn cell_count(&self) -> usize {
        self.component_values
            .first()
            .map_or(0, |component| component.len())
    }

    pub fn components(&self) -> &[FeecVector] {
        &self.component_values
    }

    pub fn component(&self, component_index: usize) -> Option<&FeecVector> {
        self.component_values.get(component_index)
    }

    pub fn trace(&self) -> FeecVector {
        let mut trace = FeecVector::zeros(self.cell_count());
        for component in &self.component_values {
            trace += component;
        }
        trace
    }

    pub fn vtk_vectors(&self) -> Vec<[f64; 3]> {
        (0..self.cell_count())
            .map(|cell_index| {
                [
                    self.component_values
                        .first()
                        .map_or(0.0, |component| component[cell_index]),
                    self.component_values
                        .get(1)
                        .map_or(0.0, |component| component[cell_index]),
                    self.component_values
                        .get(2)
                        .map_or(0.0, |component| component[cell_index]),
                ]
            })
            .collect()
    }
}

pub fn build_hodge_laplacian_1form(
    topology: &Complex,
    metric: &MeshLengths,
) -> HodgeLaplacian1Form {
    let galmats = MixedGalmats::compute(topology, metric, 1);
    build_hodge_laplacian_1form_from_galmats(&galmats)
}

pub fn build_hodge_laplacian_1form_from_galmats(galmats: &MixedGalmats) -> HodgeLaplacian1Form {
    let mass_u = galmats.mass_u_csr();
    let laplacian = galmats.hodge_laplacian_schur_complement_lumped();
    HodgeLaplacian1Form { mass_u, laplacian }
}

pub fn build_matern_system_matrix_1form(hodge: &HodgeLaplacian1Form, kappa: f64) -> FeecCsr {
    let kappa2 = kappa * kappa;
    add_sparse(&hodge.laplacian, &scale_matrix(&hodge.mass_u, kappa2))
}

pub fn build_matern_mass_inverse_1form(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u: &FeecCsr,
    strategy: MaternMassInverse,
) -> FeecCsr {
    match strategy {
        MaternMassInverse::RowSumLumped => diag_matrix(&invert_diag(&lumped_diag(mass_u))),
        MaternMassInverse::Nc1ProjectedSparseInverse => {
            let projected = assemble_whitney_projected_sparse_inverse_galmat(topology, metric);
            let projected = FeecCsr::from(&projected);
            assert_eq!(projected.nrows(), mass_u.nrows());
            assert_eq!(projected.ncols(), mass_u.ncols());
            projected
        }
    }
}

pub fn build_matern_precision_1form_with_mass_inverse(
    hodge: &HodgeLaplacian1Form,
    mass_inverse: &FeecCsr,
    kappa: f64,
    tau: f64,
) -> FeecCsr {
    let a = build_matern_system_matrix_1form(hodge, kappa);
    let middle = mass_inverse * &a;

    let mut precision = &a * &middle;
    if (tau - 1.0).abs() > f64::EPSILON {
        precision = scale_matrix(&precision, tau * tau);
    }
    precision
}

pub fn build_matern_precision_1form(
    topology: &Complex,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
) -> FeecCsr {
    let mass_inverse =
        build_matern_mass_inverse_1form(topology, metric, &hodge.mass_u, config.mass_inverse);
    build_matern_precision_1form_with_mass_inverse(hodge, &mass_inverse, config.kappa, config.tau)
}

pub fn build_reconstructed_barycenter_field_operator(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<ReconstructedBarycenterFieldOperator, String> {
    let topo_dim = topology.dim();
    if topo_dim == 0 {
        return Err("topology dimension must be at least 1 to reconstruct a 1-form".to_string());
    }
    if topo_dim > coords.dim() {
        return Err(format!(
            "invalid mesh dimensions: topology dim {} > coordinate dim {}",
            topo_dim,
            coords.dim()
        ));
    }

    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let edge_count = topology.skeleton(1).len();
    let ambient_dim = coords.dim();
    let mut component_rows = vec![Vec::with_capacity(cell_skeleton.len()); ambient_dim];

    for cell in cell_skeleton.handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let jacobian_pinv = (topo_dim < coords.dim()).then(|| cell_coords.inv_linear_transform());
        let mut cell_component_rows = vec![Vec::new(); ambient_dim];

        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let local_value = lsf.at_point(&bary_local).into_grade1();
            let ambient_value = if let Some(jacobian_pinv) = &jacobian_pinv {
                jacobian_pinv.transpose() * local_value
            } else {
                local_value
            };

            for component_index in 0..ambient_dim {
                let coefficient = ambient_value[component_index];
                if coefficient.abs() > RECONSTRUCTION_EPS {
                    cell_component_rows[component_index].push((dof_simp.kidx(), coefficient));
                }
            }
        }

        for component_index in 0..ambient_dim {
            component_rows[component_index]
                .push(std::mem::take(&mut cell_component_rows[component_index]));
        }
    }

    let component_operators = component_rows
        .into_iter()
        .map(|rows| SparseRowLinearOperator::new(edge_count, rows))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ReconstructedBarycenterFieldOperator {
        ambient_dim,
        component_operators,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ddf::cochain::Cochain;
    use formoniq::io::write_1form_vector_field_vtk;
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn diagonal_entries(mat: &FeecCsr) -> Vec<f64> {
        let mut diag = vec![0.0; mat.nrows()];
        for (row, col, value) in mat.triplet_iter() {
            if row == col {
                diag[row] += *value;
            }
        }
        diag
    }

    fn max_abs_entry_diff(lhs: &FeecCsr, rhs: &FeecCsr) -> f64 {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());

        let mut entries = HashMap::new();
        for (row, col, value) in lhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) += *value;
        }
        for (row, col, value) in rhs.triplet_iter() {
            *entries.entry((row, col)).or_insert(0.0) -= *value;
        }
        entries
            .values()
            .map(|value| value.abs())
            .fold(0.0, f64::max)
    }

    fn parse_vtk_vectors(content: &str, field_name: &str, count: usize) -> Vec<[f64; 3]> {
        let header = format!("VECTORS {field_name} double");
        let start = content
            .lines()
            .position(|line| line.trim() == header)
            .expect("vector header should be present")
            + 1;
        content
            .lines()
            .skip(start)
            .take(count)
            .map(|line| {
                let values = line
                    .split_whitespace()
                    .map(|token| token.parse::<f64>().expect("vector component should parse"))
                    .collect::<Vec<_>>();
                assert_eq!(values.len(), 3);
                [values[0], values[1], values[2]]
            })
            .collect()
    }

    #[test]
    fn hodge_laplacian_1form_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        assert!(hodge.mass_u.nrows() > 0);
        assert_eq!(hodge.mass_u.nrows(), hodge.mass_u.ncols());
        assert_eq!(hodge.laplacian.nrows(), hodge.mass_u.nrows());
        assert_eq!(hodge.laplacian.ncols(), hodge.mass_u.ncols());
    }

    #[test]
    fn matern_precision_has_positive_diagonal_with_row_sum_lumping() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let precision = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        assert!(diagonal_entries(&precision).iter().all(|v| *v > 0.0));
    }

    #[test]
    fn matern_precision_projected_sparse_inverse_differs_from_row_sum() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let row_sum = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let projected = build_matern_precision_1form(
            &topology,
            &metric,
            &hodge,
            MaternConfig {
                kappa: 1.5,
                tau: 1.0,
                mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
            },
        );

        assert_eq!(row_sum.nrows(), projected.nrows());
        assert_eq!(row_sum.ncols(), projected.ncols());
        assert!(diagonal_entries(&projected).iter().all(|v| *v > 0.0));

        let row_sum_gmrf = feec_csr_to_gmrf(&row_sum);
        let projected_gmrf = feec_csr_to_gmrf(&projected);
        row_sum_gmrf
            .cholesky_sqrt_lower()
            .expect("row-sum precision should factorize");
        projected_gmrf
            .cholesky_sqrt_lower()
            .expect("projected precision should factorize");

        assert!(max_abs_entry_diff(&row_sum, &projected) > 1e-9);
    }

    #[test]
    fn precision_from_supplied_mass_inverse_matches_enum_path() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let config = MaternConfig {
            kappa: 1.5,
            tau: 0.75,
            mass_inverse: MaternMassInverse::RowSumLumped,
        };

        let from_enum = build_matern_precision_1form(&topology, &metric, &hodge, config);
        let mass_inverse = build_matern_mass_inverse_1form(
            &topology,
            &metric,
            &hodge.mass_u,
            MaternMassInverse::RowSumLumped,
        );
        let from_supplied = build_matern_precision_1form_with_mass_inverse(
            &hodge,
            &mass_inverse,
            config.kappa,
            config.tau,
        );

        assert!(max_abs_entry_diff(&from_enum, &from_supplied) <= 1e-12);
    }

    #[test]
    fn hodge_laplacian_1form_torus_mesh_dimensions() {
        let mesh_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("Failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);

        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        assert_eq!(topology.dim(), 2);
        assert_eq!(coords.dim(), 3);
        assert!(hodge.mass_u.nrows() > 0);
        assert_eq!(hodge.mass_u.nrows(), hodge.mass_u.ncols());
        assert_eq!(hodge.laplacian.nrows(), hodge.mass_u.nrows());
        assert_eq!(hodge.laplacian.ncols(), hodge.mass_u.ncols());
    }

    #[test]
    fn reconstructed_barycenter_field_trace_sums_component_fields() {
        let field = ReconstructedBarycenterField::from_components(vec![
            FeecVector::from_vec(vec![1.0, 2.0]),
            FeecVector::from_vec(vec![0.5, 1.5]),
            FeecVector::from_vec(vec![2.0, 3.0]),
        ])
        .expect("field components should be compatible");

        let trace = field.trace();
        assert_eq!(field.ambient_dim(), 3);
        assert_eq!(field.cell_count(), 2);
        assert!((trace[0] - 3.5).abs() < 1e-12);
        assert!((trace[1] - 6.5).abs() < 1e-12);
    }

    #[test]
    fn reconstructed_barycenter_operator_matches_formoniq_vector_writer_on_torus() {
        let mesh_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);

        let operator = build_reconstructed_barycenter_field_operator(&topology, &coords)
            .expect("reconstruction operator should build");
        assert_eq!(operator.ambient_dim(), coords.dim());
        assert_eq!(operator.cell_count(), topology.cells().len());

        let edge_count = topology.skeleton(1).len();
        let coefficients =
            FeecVector::from_iterator(edge_count, (0..edge_count).map(|i| ((i + 1) as f64).sin()));
        let field = operator
            .apply_to_slice(coefficients.as_slice())
            .expect("operator application should succeed");
        let expected_vectors = field.vtk_vectors();

        let cochain = Cochain::new(1, coefficients);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "reconstructed_barycenter_operator_matches_formoniq_{stamp}.vtk"
        ));
        write_1form_vector_field_vtk(&path, &coords, &topology, &cochain, "embedded")
            .expect("vector field VTK should write");

        let content = std::fs::read_to_string(&path).expect("VTK should be readable");
        let _ = std::fs::remove_file(&path);
        let actual_vectors = parse_vtk_vectors(&content, "embedded", topology.cells().len());

        assert_eq!(expected_vectors.len(), actual_vectors.len());
        for (expected, actual) in expected_vectors.iter().zip(actual_vectors.iter()) {
            for component in 0..3 {
                assert!(
                    (expected[component] - actual[component]).abs() < 1e-10,
                    "component mismatch: expected {} got {}",
                    expected[component],
                    actual[component]
                );
            }
        }
    }
}
