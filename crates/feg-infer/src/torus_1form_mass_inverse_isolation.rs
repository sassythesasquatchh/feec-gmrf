use crate::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints, edge_lengths,
};
use crate::matern_1form::{
    build_matern_mass_inverse_1form, build_matern_precision_1form_with_mass_inverse,
    HodgeLaplacian1Form, MaternMassInverse,
};
use crate::sparse::{dense_to_feec_csr, feec_csr_to_dense};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::path::PathBuf;

const DENSE_DROP_TOLERANCE: f64 = 1e-14;
const VARIANCE_FLOOR_SCALE: f64 = 1e-14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationMassInverseKind {
    RowSumLumped,
    Nc1ProjectedSparseInverse,
    ExactConsistentInverse,
}

impl IsolationMassInverseKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::RowSumLumped => "row_sum_lumped",
            Self::Nc1ProjectedSparseInverse => "nc1_projected_sparse_inverse",
            Self::ExactConsistentInverse => "exact_consistent_inverse",
        }
    }
}

#[derive(Debug, Clone)]
pub struct IsolationStrategyReport {
    pub kind: IsolationMassInverseKind,
    pub prior_precision: FeecCsr,
    pub unconstrained_variances: FeecVector,
    pub harmonic_free_variances: FeecVector,
    pub harmonic_removed_variances: FeecVector,
    pub log_harmonic_free_variance_per_length2: FeecVector,
    pub centered_log_harmonic_free_variance_per_length2: FeecVector,
    pub h_raw: f64,
    pub h_hf: f64,
    pub h_post_len: f64,
    pub harmonic_removed_fraction_mean: f64,
    pub distance_to_exact: Option<f64>,
    pub delta_h_hf_vs_exact: Option<f64>,
    pub delta_h_post_len_vs_exact: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Torus1FormMassInverseIsolationReport {
    pub edge_lengths: FeecVector,
    pub harmonic_constraint_rank: usize,
    pub row_sum: IsolationStrategyReport,
    pub projected: IsolationStrategyReport,
    pub exact_consistent: IsolationStrategyReport,
}

pub fn default_torus_shell_coarse_mesh_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_coarse.msh")
}

pub fn compute_torus_1form_mass_inverse_isolation_report(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    kappa: f64,
    tau: f64,
) -> Result<Torus1FormMassInverseIsolationReport, String> {
    let edge_lengths = edge_lengths(topology, coords);
    let harmonic_basis = build_analytic_torus_harmonic_basis(topology, coords, metric)?;
    let harmonic_constraints =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)?;
    let harmonic_constraints = gmrf_dense_to_feec_dense(&harmonic_constraints);

    let row_sum_mass_inverse = build_matern_mass_inverse_1form(
        topology,
        metric,
        &hodge.mass_u,
        MaternMassInverse::RowSumLumped,
    );
    let projected_mass_inverse = build_matern_mass_inverse_1form(
        topology,
        metric,
        &hodge.mass_u,
        MaternMassInverse::Nc1ProjectedSparseInverse,
    );
    let exact_consistent_mass_inverse_dense = feec_csr_to_dense(&hodge.mass_u)
        .try_inverse()
        .ok_or_else(|| "failed to invert consistent Whitney mass matrix".to_string())?;
    let exact_consistent_mass_inverse =
        dense_to_feec_csr(&exact_consistent_mass_inverse_dense, DENSE_DROP_TOLERANCE);

    let mut row_sum = build_strategy_report(
        IsolationMassInverseKind::RowSumLumped,
        hodge,
        &row_sum_mass_inverse,
        &harmonic_constraints,
        &edge_lengths,
        kappa,
        tau,
    )?;
    let mut projected = build_strategy_report(
        IsolationMassInverseKind::Nc1ProjectedSparseInverse,
        hodge,
        &projected_mass_inverse,
        &harmonic_constraints,
        &edge_lengths,
        kappa,
        tau,
    )?;
    let exact_consistent = build_strategy_report(
        IsolationMassInverseKind::ExactConsistentInverse,
        hodge,
        &exact_consistent_mass_inverse,
        &harmonic_constraints,
        &edge_lengths,
        kappa,
        tau,
    )?;

    row_sum.distance_to_exact = Some(rms_difference(
        &row_sum.centered_log_harmonic_free_variance_per_length2,
        &exact_consistent.centered_log_harmonic_free_variance_per_length2,
    )?);
    row_sum.delta_h_hf_vs_exact = Some(row_sum.h_hf - exact_consistent.h_hf);
    row_sum.delta_h_post_len_vs_exact = Some(row_sum.h_post_len - exact_consistent.h_post_len);

    projected.distance_to_exact = Some(rms_difference(
        &projected.centered_log_harmonic_free_variance_per_length2,
        &exact_consistent.centered_log_harmonic_free_variance_per_length2,
    )?);
    projected.delta_h_hf_vs_exact = Some(projected.h_hf - exact_consistent.h_hf);
    projected.delta_h_post_len_vs_exact = Some(projected.h_post_len - exact_consistent.h_post_len);

    Ok(Torus1FormMassInverseIsolationReport {
        edge_lengths,
        harmonic_constraint_rank: harmonic_constraints.nrows(),
        row_sum,
        projected,
        exact_consistent,
    })
}

fn build_strategy_report(
    kind: IsolationMassInverseKind,
    hodge: &HodgeLaplacian1Form,
    mass_inverse: &FeecCsr,
    harmonic_constraints: &FeecMatrix,
    edge_lengths: &FeecVector,
    kappa: f64,
    tau: f64,
) -> Result<IsolationStrategyReport, String> {
    let prior_precision =
        build_matern_precision_1form_with_mass_inverse(hodge, mass_inverse, kappa, tau);
    let covariance = feec_csr_to_dense(&prior_precision)
        .try_inverse()
        .ok_or_else(|| format!("failed to invert prior precision for {}", kind.label()))?;
    let harmonic_free_covariance = constrained_covariance(&covariance, harmonic_constraints)?;

    let unconstrained_variances_raw = dense_diagonal(&covariance);
    let harmonic_free_variances_raw = dense_diagonal(&harmonic_free_covariance);
    let unconstrained_variances = stabilize_variance_vector(&unconstrained_variances_raw)?;
    let harmonic_free_variances = stabilize_variance_vector(&harmonic_free_variances_raw)?;
    let harmonic_removed_variances = FeecVector::from_iterator(
        unconstrained_variances.len(),
        (0..unconstrained_variances.len())
            .map(|i| (unconstrained_variances[i] - harmonic_free_variances[i]).max(0.0)),
    );
    let log_harmonic_free_variance_per_length2 =
        log_harmonic_free_per_length2(&harmonic_free_variances, edge_lengths)?;
    let centered_log_harmonic_free_variance_per_length2 =
        center_vector(&log_harmonic_free_variance_per_length2);

    Ok(IsolationStrategyReport {
        kind,
        prior_precision,
        h_raw: log_field_variance(&unconstrained_variances)?,
        h_hf: log_field_variance(&harmonic_free_variances)?,
        h_post_len: vector_variance(&log_harmonic_free_variance_per_length2)?,
        harmonic_removed_fraction_mean: mean_removed_fraction(
            &unconstrained_variances,
            &harmonic_free_variances,
        )?,
        unconstrained_variances,
        harmonic_free_variances,
        harmonic_removed_variances,
        log_harmonic_free_variance_per_length2,
        centered_log_harmonic_free_variance_per_length2,
        distance_to_exact: None,
        delta_h_hf_vs_exact: None,
        delta_h_post_len_vs_exact: None,
    })
}

fn gmrf_dense_to_feec_dense(matrix: &gmrf_core::types::DenseMatrix) -> FeecMatrix {
    FeecMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| matrix[(i, j)])
}

fn dense_diagonal(matrix: &FeecMatrix) -> FeecVector {
    FeecVector::from_iterator(
        matrix.nrows().min(matrix.ncols()),
        (0..matrix.nrows().min(matrix.ncols())).map(|i| matrix[(i, i)]),
    )
}

fn constrained_covariance(
    covariance: &FeecMatrix,
    constraint_matrix: &FeecMatrix,
) -> Result<FeecMatrix, String> {
    if constraint_matrix.ncols() != covariance.ncols() {
        return Err(format!(
            "constraint columns {} do not match covariance dimension {}",
            constraint_matrix.ncols(),
            covariance.ncols()
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return Ok(covariance.clone());
    }

    let covariance_times_constraint_t = covariance * constraint_matrix.transpose();
    let schur = constraint_matrix * &covariance_times_constraint_t;
    let schur_inverse = schur
        .try_inverse()
        .ok_or_else(|| "failed to invert harmonic constraint Schur complement".to_string())?;
    let correction =
        &covariance_times_constraint_t * schur_inverse * covariance_times_constraint_t.transpose();
    Ok(covariance - correction)
}

fn stabilize_variance_vector(values: &FeecVector) -> Result<FeecVector, String> {
    if values.is_empty() {
        return Ok(values.clone());
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err("variance vector contains non-finite entries".to_string());
    }

    let positive_mean = values
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .sum::<f64>()
        / (values.iter().filter(|value| **value > 0.0).count().max(1) as f64);
    let scale = positive_mean.abs().max(1.0);
    let floor = scale * VARIANCE_FLOOR_SCALE;
    let negative_tolerance = scale * 1e-12;

    let mut stabilized = FeecVector::zeros(values.len());
    for i in 0..values.len() {
        let value = values[i];
        if value < -negative_tolerance {
            return Err(format!(
                "variance vector contains a materially negative entry {value} at index {i}"
            ));
        }
        stabilized[i] = if value > floor { value } else { floor };
    }
    Ok(stabilized)
}

fn log_field_variance(values: &FeecVector) -> Result<f64, String> {
    vector_variance(&log_vector(values)?)
}

fn log_harmonic_free_per_length2(
    harmonic_free_variances: &FeecVector,
    edge_lengths: &FeecVector,
) -> Result<FeecVector, String> {
    if harmonic_free_variances.len() != edge_lengths.len() {
        return Err(format!(
            "harmonic-free variance length {} does not match edge length count {}",
            harmonic_free_variances.len(),
            edge_lengths.len()
        ));
    }
    let scaled = FeecVector::from_iterator(
        harmonic_free_variances.len(),
        (0..harmonic_free_variances.len())
            .map(|i| harmonic_free_variances[i] / (edge_lengths[i] * edge_lengths[i])),
    );
    log_vector(&scaled)
}

fn log_vector(values: &FeecVector) -> Result<FeecVector, String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("log metrics require finite positive values".to_string());
    }
    Ok(values.map(|value| value.ln()))
}

fn center_vector(values: &FeecVector) -> FeecVector {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.map(|value| value - mean)
}

fn vector_variance(values: &FeecVector) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot compute variance of an empty vector".to_string());
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Ok(values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
}

fn rms_difference(lhs: &FeecVector, rhs: &FeecVector) -> Result<f64, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "RMS vectors differ in length: {} vs {}",
            lhs.len(),
            rhs.len()
        ));
    }
    Ok(((0..lhs.len())
        .map(|i| {
            let diff = lhs[i] - rhs[i];
            diff * diff
        })
        .sum::<f64>()
        / lhs.len() as f64)
        .sqrt())
}

fn mean_removed_fraction(
    unconstrained_variances: &FeecVector,
    harmonic_free_variances: &FeecVector,
) -> Result<f64, String> {
    if unconstrained_variances.len() != harmonic_free_variances.len() {
        return Err(format!(
            "variance lengths differ: {} vs {}",
            unconstrained_variances.len(),
            harmonic_free_variances.len()
        ));
    }
    Ok((0..unconstrained_variances.len())
        .map(|i| {
            let removed = (unconstrained_variances[i] - harmonic_free_variances[i]).max(0.0);
            removed / unconstrained_variances[i]
        })
        .sum::<f64>()
        / unconstrained_variances.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::CooMatrix as FeecCoo;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn constrained_covariance_matches_coordinate_constraint() {
        let covariance = FeecMatrix::from_diagonal_element(2, 2, 1.0);
        let constraints = FeecMatrix::from_row_slice(1, 2, &[1.0, 0.0]);

        let constrained =
            constrained_covariance(&covariance, &constraints).expect("constraint should work");

        assert!(approx_eq(constrained[(0, 0)], 0.0));
        assert!(approx_eq(constrained[(1, 1)], 1.0));
        assert!(approx_eq(constrained[(0, 1)], 0.0));
        assert!(approx_eq(constrained[(1, 0)], 0.0));
    }

    #[test]
    fn stabilize_variance_vector_clamps_tiny_negative_roundoff() {
        let values = FeecVector::from_vec(vec![1.0, -1e-16, 2.0]);
        let stabilized = stabilize_variance_vector(&values).expect("tiny negatives should clamp");

        assert!(approx_eq(stabilized[0], 1.0));
        assert!(stabilized[1] > 0.0);
        assert!(approx_eq(stabilized[2], 2.0));
    }

    #[test]
    fn stabilize_variance_vector_rejects_material_negative_entries() {
        let values = FeecVector::from_vec(vec![1.0, -1e-3, 2.0]);
        let err = stabilize_variance_vector(&values).expect_err("negative variance should fail");
        assert!(err.contains("materially negative"));
    }

    #[test]
    fn dense_diagonal_reads_expected_entries() {
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(0, 1, 0.5);
        coo.push(1, 1, 3.0);
        let csr = FeecCsr::from(&coo);
        let diag = dense_diagonal(&feec_csr_to_dense(&csr));

        assert!(approx_eq(diag[0], 2.0));
        assert!(approx_eq(diag[1], 3.0));
    }
}
