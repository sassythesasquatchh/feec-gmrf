use crate::matern_0form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, MaternConfig as Matern0FormConfig,
    MaternMassInverse as Matern0FormMassInverse,
};
use crate::matern_1form::{
    build_hodge_laplacian_1form, build_matern_mass_inverse_1form, build_matern_precision_1form,
    build_reconstructed_barycenter_field_operator, MaternConfig as Matern1FormConfig,
    MaternMassInverse as Matern1FormMassInverse, ReconstructedBarycenterFieldOperator,
};
use crate::matern_2form::{
    build_hodge_laplacian_2form, build_matern_precision_2form, MaternConfig as Matern2FormConfig,
    MaternMassInverse as Matern2FormMassInverse,
};
use crate::sparse::{
    dense_to_feec_csr, feec_csr_to_dense, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use formoniq::problems::hodge_laplace::{solve_hodge_laplace_harmonics_with_galmats, MixedGalmats};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::{GmrfError, Vector as GmrfVector};
use gmrf_core::Gmrf;
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use std::{collections::BTreeMap, iter};

#[cfg(test)]
use std::sync::{Mutex, MutexGuard, OnceLock};

const EXACT_VARIANCE_TOLERANCE: f64 = 1e-10;
const EPS: f64 = 1e-12;

#[cfg(test)]
static PETSC_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

#[cfg(test)]
pub(crate) fn lock_feec_harmonic_tests() -> MutexGuard<'static, ()> {
    match PETSC_TEST_LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hodge1FormBranchKind {
    Exact,
    Coexact,
    Harmonic,
}

impl Hodge1FormBranchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Coexact => "coexact",
            Self::Harmonic => "harmonic",
        }
    }
}

pub struct Hodge1FormConditioningConfig<'a> {
    pub topology: &'a Complex,
    pub coords: &'a MeshCoords,
    pub metric: &'a MeshLengths,
    pub truth: &'a FeecVector,
    pub observation_matrix: &'a FeecCsr,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub harmonic_dim: usize,
    pub harmonic_basis_override: Option<&'a FeecMatrix>,
}

#[derive(Debug, Clone)]
pub struct ReconstructedBarycenterVarianceFields {
    pub prior_components: Vec<FeecVector>,
    pub posterior_components: Vec<FeecVector>,
    pub prior_trace: FeecVector,
    pub posterior_trace: FeecVector,
    pub trace_ratio: FeecVector,
}

impl ReconstructedBarycenterVarianceFields {
    pub fn ambient_dim(&self) -> usize {
        self.prior_components.len()
    }

    pub fn cell_count(&self) -> usize {
        self.prior_trace.len()
    }

    pub fn prior_vtk_vectors(&self) -> Vec<[f64; 3]> {
        components_to_vtk_vectors(&self.prior_components)
    }

    pub fn posterior_vtk_vectors(&self) -> Vec<[f64; 3]> {
        components_to_vtk_vectors(&self.posterior_components)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Hodge1FormPosteriorBiasDiagnostics {
    pub curl_residual_norm: f64,
    pub curl_residual_relative: f64,
    pub coclosed_residual_norm: f64,
    pub coclosed_residual_relative: f64,
}

#[derive(Debug, Clone)]
pub struct Hodge1FormBranchResult {
    pub kind: Hodge1FormBranchKind,
    pub latent_dimension: usize,
    pub observation_count: usize,
    pub latent_posterior_mean: FeecVector,
    pub posterior_mean: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub observation_values: FeecVector,
    pub posterior_observation_mean: FeecVector,
    pub observation_residual: FeecVector,
    pub max_abs_observation_error: f64,
    pub mean_abs_observation_error: f64,
    pub harmonic_coefficients_truth: FeecVector,
    pub harmonic_coefficients_posterior_mean: FeecVector,
    pub harmonic_residual_norm_truth: f64,
    pub harmonic_residual_norm_posterior_mean: f64,
    pub posterior_bias_diagnostics: Hodge1FormPosteriorBiasDiagnostics,
    pub reconstructed_barycenter_variance: ReconstructedBarycenterVarianceFields,
}

#[derive(Debug, Clone)]
pub struct Hodge1FormConditioningResult {
    pub observations: FeecVector,
    pub harmonic_basis: FeecMatrix,
    pub exact: Hodge1FormBranchResult,
    pub coexact: Hodge1FormBranchResult,
    pub harmonic: Hodge1FormBranchResult,
}

#[derive(Debug, Clone)]
struct SparseRowLinearOperator {
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
}

struct ComposedBarycenterOperators {
    ambient_dim: usize,
    cell_count: usize,
    component_operators: Vec<SparseRowLinearOperator>,
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

    fn from_sparse_matrix(matrix: &FeecCsr) -> Result<Self, String> {
        let mut rows = vec![Vec::new(); matrix.nrows()];
        for (row, col, value) in matrix.triplet_iter() {
            if value.abs() > EPS {
                rows[row].push((col, *value));
            }
        }
        Self::new(matrix.ncols(), rows)
    }

    fn from_dense_matrix(matrix: &FeecMatrix, drop_tolerance: f64) -> Result<Self, String> {
        let tol = drop_tolerance.abs();
        let mut rows = Vec::with_capacity(matrix.nrows());
        for row in 0..matrix.nrows() {
            let mut entries = Vec::new();
            for col in 0..matrix.ncols() {
                let value = matrix[(row, col)];
                if value.abs() > tol {
                    entries.push((col, value));
                }
            }
            rows.push(entries);
        }
        Self::new(matrix.ncols(), rows)
    }

    fn nrows(&self) -> usize {
        self.rows.len()
    }

    fn apply_feec(&self, input: &FeecVector) -> Result<FeecVector, String> {
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

pub fn build_exact_1form_transform(topology: &Complex) -> FeecCsr {
    FeecCsr::from(&topology.exterior_derivative_operator(0))
}

pub fn build_coexact_1form_transform(
    topology: &Complex,
    metric: &MeshLengths,
    mass_u_1form: &FeecCsr,
) -> FeecCsr {
    let galmats = MixedGalmats::compute(topology, metric, 2);
    let codif_u = FeecCsr::from(galmats.codif_u());
    let mass_inverse_1form = build_matern_mass_inverse_1form(
        topology,
        metric,
        mass_u_1form,
        Matern1FormMassInverse::Nc1ProjectedSparseInverse,
    );
    &mass_inverse_1form * &codif_u
}

pub fn compute_harmonic_basis_1form(
    topology: &Complex,
    metric: &MeshLengths,
    harmonic_dim: usize,
    harmonic_basis_override: Option<&FeecMatrix>,
) -> Result<FeecMatrix, String> {
    if let Some(basis) = harmonic_basis_override {
        if basis.nrows() != topology.skeleton(1).len() {
            return Err(format!(
                "harmonic basis row count {} does not match 1-form dimension {}",
                basis.nrows(),
                topology.skeleton(1).len()
            ));
        }
        if basis.ncols() != harmonic_dim {
            return Err(format!(
                "harmonic basis column count {} does not match requested harmonic dimension {}",
                basis.ncols(),
                harmonic_dim
            ));
        }
        return Ok(basis.clone());
    }

    let galmats = MixedGalmats::compute(topology, metric, 1);
    Ok(solve_hodge_laplace_harmonics_with_galmats(
        topology,
        &galmats,
        1,
        harmonic_dim,
        None,
        None,
    ))
}

pub fn mass_orthonormalize_harmonic_basis_1form(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<FeecMatrix, String> {
    if harmonic_basis.ncols() == 0 {
        return Ok(FeecMatrix::zeros(harmonic_basis.nrows(), 0));
    }

    let mut columns = Vec::with_capacity(harmonic_basis.ncols());
    for j in 0..harmonic_basis.ncols() {
        let mut column = harmonic_basis.column(j).into_owned();
        for previous in &columns {
            let coeff = mass_inner_product(previous, &column, mass_u);
            column -= previous * coeff;
        }

        let norm_sq = mass_inner_product(&column, &column, mass_u);
        if !norm_sq.is_finite() || norm_sq <= EPS {
            return Err(format!(
                "harmonic basis column {j} became singular during mass orthonormalization"
            ));
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }

    Ok(FeecMatrix::from_columns(&columns))
}

pub fn harmonic_coefficients_1form(
    field: &FeecVector,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
) -> FeecVector {
    FeecVector::from_iterator(
        harmonic_basis_orthonormal.ncols(),
        (0..harmonic_basis_orthonormal.ncols()).map(|j| {
            mass_inner_product(
                &harmonic_basis_orthonormal.column(j).into_owned(),
                field,
                mass_u,
            )
        }),
    )
}

pub fn run_hodge_1form_conditioning(
    config: &Hodge1FormConditioningConfig<'_>,
) -> Result<Hodge1FormConditioningResult, String> {
    validate_config(config)?;

    let hodge_1form = build_hodge_laplacian_1form(config.topology, config.metric);
    let q1 = build_matern_precision_1form(
        config.topology,
        config.metric,
        &hodge_1form,
        Matern1FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );

    let harmonic_basis = compute_harmonic_basis_1form(
        config.topology,
        config.metric,
        config.harmonic_dim,
        config.harmonic_basis_override,
    )?;
    let harmonic_basis =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis, &hodge_1form.mass_u)?;

    let observation_values = config.observation_matrix * config.truth;
    let reconstructed_barycenter =
        build_reconstructed_barycenter_field_operator(config.topology, config.coords)?;
    let d0 = FeecCsr::from(&config.topology.exterior_derivative_operator(0));
    let d1 = FeecCsr::from(&config.topology.exterior_derivative_operator(1));

    let exact_transform = build_exact_1form_transform(config.topology);
    let exact_branch = condition_branch_sparse(
        Hodge1FormBranchKind::Exact,
        &exact_transform,
        &build_matern_precision_0form(
            &build_laplace_beltrami_0form(config.topology, config.metric),
            Matern0FormConfig {
                kappa: config.kappa,
                tau: config.tau,
                mass_inverse: Matern0FormMassInverse::RowSumLumped,
            },
        ),
        &observation_values,
        config,
        &harmonic_basis,
        &hodge_1form.mass_u,
        &reconstructed_barycenter,
        &d0,
        &d1,
    )?;

    let coexact_transform =
        build_coexact_1form_transform(config.topology, config.metric, &hodge_1form.mass_u);
    let q2 = build_matern_precision_2form(
        config.topology,
        config.metric,
        &build_hodge_laplacian_2form(config.topology, config.metric)?,
        Matern2FormConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
        },
    )?;
    let coexact_branch = condition_branch_sparse(
        Hodge1FormBranchKind::Coexact,
        &coexact_transform,
        &q2,
        &observation_values,
        config,
        &harmonic_basis,
        &hodge_1form.mass_u,
        &reconstructed_barycenter,
        &d0,
        &d1,
    )?;

    let harmonic_precision = build_harmonic_restricted_precision(&q1, &harmonic_basis);
    let harmonic_branch = condition_branch_dense(
        Hodge1FormBranchKind::Harmonic,
        &harmonic_basis,
        &harmonic_precision,
        &observation_values,
        config,
        &harmonic_basis,
        &hodge_1form.mass_u,
        &reconstructed_barycenter,
        &d0,
        &d1,
    )?;

    Ok(Hodge1FormConditioningResult {
        observations: observation_values,
        harmonic_basis,
        exact: exact_branch,
        coexact: coexact_branch,
        harmonic: harmonic_branch,
    })
}

fn validate_config(config: &Hodge1FormConditioningConfig<'_>) -> Result<(), String> {
    let ambient_dim = config.topology.skeleton(1).len();
    if config.truth.len() != ambient_dim {
        return Err(format!(
            "truth length {} does not match 1-form dimension {}",
            config.truth.len(),
            ambient_dim
        ));
    }
    if config.observation_matrix.ncols() != ambient_dim {
        return Err(format!(
            "observation matrix column count {} does not match 1-form dimension {}",
            config.observation_matrix.ncols(),
            ambient_dim
        ));
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err("kappa must be finite and positive".to_string());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err("tau must be finite and positive".to_string());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err("noise_variance must be finite and positive".to_string());
    }
    Ok(())
}

pub(crate) fn build_harmonic_restricted_precision(
    q1: &FeecCsr,
    harmonic_basis: &FeecMatrix,
) -> FeecCsr {
    if harmonic_basis.ncols() == 0 {
        return FeecCsr::from(&common::linalg::nalgebra::CooMatrix::new(0, 0));
    }

    let q1_dense = feec_csr_to_dense(q1);
    let reduced = harmonic_basis.transpose() * q1_dense * harmonic_basis;
    dense_to_feec_csr(&reduced, 0.0)
}

fn condition_branch_sparse(
    kind: Hodge1FormBranchKind,
    ambient_transform: &FeecCsr,
    latent_precision: &FeecCsr,
    observations: &FeecVector,
    config: &Hodge1FormConditioningConfig<'_>,
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
    reconstructed_barycenter: &ReconstructedBarycenterFieldOperator,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Result<Hodge1FormBranchResult, String> {
    let ambient_operator = SparseRowLinearOperator::from_sparse_matrix(ambient_transform)?;
    let barycenter_operator =
        compose_reconstructed_barycenter_operators(reconstructed_barycenter, &ambient_operator)?;
    let latent_observation_matrix = config.observation_matrix * ambient_transform;
    condition_branch(
        kind,
        &ambient_operator,
        &barycenter_operator,
        &latent_observation_matrix,
        latent_precision,
        observations,
        config.truth,
        config.observation_matrix,
        harmonic_basis,
        mass_u,
        &d0,
        &d1,
        config.noise_variance,
    )
}

fn condition_branch_dense(
    kind: Hodge1FormBranchKind,
    ambient_transform: &FeecMatrix,
    latent_precision: &FeecCsr,
    observations: &FeecVector,
    config: &Hodge1FormConditioningConfig<'_>,
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
    reconstructed_barycenter: &ReconstructedBarycenterFieldOperator,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Result<Hodge1FormBranchResult, String> {
    let ambient_operator = SparseRowLinearOperator::from_dense_matrix(ambient_transform, 0.0)?;
    let barycenter_operator =
        compose_reconstructed_barycenter_operators(reconstructed_barycenter, &ambient_operator)?;
    let latent_observation_dense = feec_csr_to_dense(config.observation_matrix) * ambient_transform;
    let latent_observation_matrix = dense_to_feec_csr(&latent_observation_dense, 0.0);
    condition_branch(
        kind,
        &ambient_operator,
        &barycenter_operator,
        &latent_observation_matrix,
        latent_precision,
        observations,
        config.truth,
        config.observation_matrix,
        harmonic_basis,
        mass_u,
        &d0,
        &d1,
        config.noise_variance,
    )
}

#[allow(clippy::too_many_arguments)]
fn condition_branch(
    kind: Hodge1FormBranchKind,
    ambient_operator: &SparseRowLinearOperator,
    barycenter_operator: &ComposedBarycenterOperators,
    latent_observation_matrix: &FeecCsr,
    latent_precision: &FeecCsr,
    observations: &FeecVector,
    truth: &FeecVector,
    ambient_observation_matrix: &FeecCsr,
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
    d0: &FeecCsr,
    d1: &FeecCsr,
    noise_variance: f64,
) -> Result<Hodge1FormBranchResult, String> {
    let latent_dim = latent_precision.nrows();
    let ambient_dim = ambient_operator.nrows();

    if latent_dim == 0 {
        let posterior_mean = FeecVector::zeros(ambient_dim);
        let posterior_observation_mean = ambient_observation_matrix * &posterior_mean;
        let observation_residual = &posterior_observation_mean - observations;
        return Ok(Hodge1FormBranchResult {
            kind,
            latent_dimension: 0,
            observation_count: observations.len(),
            latent_posterior_mean: FeecVector::zeros(0),
            posterior_mean: posterior_mean.clone(),
            absolute_mean_error: absolute_difference(&posterior_mean, truth),
            prior_variance: FeecVector::zeros(ambient_dim),
            posterior_variance: FeecVector::zeros(ambient_dim),
            variance_reduction: FeecVector::zeros(ambient_dim),
            observation_values: observations.clone(),
            posterior_observation_mean: posterior_observation_mean.clone(),
            observation_residual: observation_residual.clone(),
            max_abs_observation_error: max_abs(&observation_residual),
            mean_abs_observation_error: mean_abs(&observation_residual),
            harmonic_coefficients_truth: harmonic_coefficients_1form(truth, harmonic_basis, mass_u),
            harmonic_coefficients_posterior_mean: FeecVector::zeros(harmonic_basis.ncols()),
            harmonic_residual_norm_truth: harmonic_coefficients_1form(
                truth,
                harmonic_basis,
                mass_u,
            )
            .norm(),
            harmonic_residual_norm_posterior_mean: 0.0,
            posterior_bias_diagnostics: compute_posterior_bias_diagnostics(
                &posterior_mean,
                mass_u,
                d0,
                d1,
            ),
            reconstructed_barycenter_variance: zero_reconstructed_barycenter_variance_fields(
                barycenter_operator,
            ),
        });
    }

    let prior_precision_gmrf = feec_csr_to_gmrf(latent_precision);
    let empty_constraints = gmrf_core::types::DenseMatrix::zeros(0, latent_dim);

    let mut prior =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(latent_dim), prior_precision_gmrf.clone())
            .map_err(|err| err.to_string())?;
    let prior_variance =
        exact_transformed_variances(&mut prior, ambient_operator).map_err(|err| err.to_string())?;
    let prior_barycenter_variance =
        exact_reconstructed_barycenter_variances(&mut prior, barycenter_operator)
            .map_err(|err| err.to_string())?;

    let (posterior_precision, information) = apply_gaussian_observations(
        &prior_precision_gmrf,
        &feec_csr_to_gmrf(latent_observation_matrix),
        &feec_vec_to_gmrf(observations),
        None,
        noise_variance,
    );
    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .map_err(|err| {
            format!(
                "failed to build posterior for {} branch: {err}",
                kind.as_str()
            )
        })?;
    let latent_posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let posterior_mean = ambient_operator.apply_feec(&latent_posterior_mean)?;
    let posterior_variance = exact_transformed_variances(&mut posterior, ambient_operator)
        .map_err(|err| err.to_string())?;
    let posterior_barycenter_variance =
        exact_reconstructed_barycenter_variances(&mut posterior, barycenter_operator)
            .map_err(|err| err.to_string())?;
    let _posterior_latent_variance = posterior
        .exact_constrained_variance_decomposition(&empty_constraints)
        .map_err(|err| err.to_string())?
        .unconstrained_diag;

    let observation_values = observations.clone();
    let posterior_observation_mean = ambient_observation_matrix * &posterior_mean;
    let observation_residual = &posterior_observation_mean - observations;
    let harmonic_coefficients_truth = harmonic_coefficients_1form(truth, harmonic_basis, mass_u);
    let harmonic_coefficients_posterior_mean =
        harmonic_coefficients_1form(&posterior_mean, harmonic_basis, mass_u);
    let posterior_bias_diagnostics =
        compute_posterior_bias_diagnostics(&posterior_mean, mass_u, d0, d1);

    Ok(Hodge1FormBranchResult {
        kind,
        latent_dimension: latent_dim,
        observation_count: observations.len(),
        latent_posterior_mean,
        posterior_mean: posterior_mean.clone(),
        absolute_mean_error: absolute_difference(&posterior_mean, truth),
        prior_variance: gmrf_vec_to_feec(&prior_variance),
        posterior_variance: gmrf_vec_to_feec(&posterior_variance),
        variance_reduction: FeecVector::from_iterator(
            ambient_dim,
            (0..ambient_dim).map(|i| prior_variance[i] - posterior_variance[i]),
        ),
        observation_values,
        posterior_observation_mean: posterior_observation_mean.clone(),
        observation_residual: observation_residual.clone(),
        max_abs_observation_error: max_abs(&observation_residual),
        mean_abs_observation_error: mean_abs(&observation_residual),
        harmonic_residual_norm_truth: harmonic_coefficients_truth.norm(),
        harmonic_residual_norm_posterior_mean: harmonic_coefficients_posterior_mean.norm(),
        harmonic_coefficients_truth,
        harmonic_coefficients_posterior_mean,
        posterior_bias_diagnostics,
        reconstructed_barycenter_variance: build_reconstructed_barycenter_variance_fields(
            &prior_barycenter_variance,
            &posterior_barycenter_variance,
        ),
    })
}

fn exact_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowLinearOperator,
) -> Result<GmrfVector, GmrfError> {
    let mut variances = GmrfVector::zeros(operator.nrows());
    for (row_index, row) in operator.rows.iter().enumerate() {
        let rhs = sparse_row_rhs(row, operator.ncols);
        let solved = gmrf.solve_precision(&rhs)?;
        let variance = row
            .iter()
            .map(|(state_index, weight)| *weight * solved[*state_index])
            .sum::<f64>();
        variances[row_index] = clamp_small_negative_variance(
            variance,
            rhs.norm().max(1.0),
            "transformed marginal variance must be nonnegative",
        )?;
    }
    Ok(variances)
}

pub(crate) fn compute_posterior_bias_diagnostics(
    posterior_mean: &FeecVector,
    mass_u: &FeecCsr,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Hodge1FormPosteriorBiasDiagnostics {
    let curl_residual = d1 * posterior_mean;
    let weighted_mean = mass_u * posterior_mean;
    let coclosed_residual = d0.transpose() * &weighted_mean;
    let state_norm = posterior_mean.norm().max(EPS);
    let weighted_state_norm = weighted_mean.norm().max(EPS);

    Hodge1FormPosteriorBiasDiagnostics {
        curl_residual_norm: curl_residual.norm(),
        curl_residual_relative: curl_residual.norm() / state_norm,
        coclosed_residual_norm: coclosed_residual.norm(),
        coclosed_residual_relative: coclosed_residual.norm() / weighted_state_norm,
    }
}

fn sparse_row_rhs(row: &[(usize, f64)], dimension: usize) -> GmrfVector {
    let mut rhs = GmrfVector::zeros(dimension);
    for (col, value) in row {
        rhs[*col] = *value;
    }
    rhs
}

fn compose_reconstructed_barycenter_operators(
    reconstruction: &ReconstructedBarycenterFieldOperator,
    ambient_operator: &SparseRowLinearOperator,
) -> Result<ComposedBarycenterOperators, String> {
    if reconstruction.component_count() != reconstruction.ambient_dim() {
        return Err(
            "reconstructed barycenter operator has inconsistent component count".to_string(),
        );
    }

    let mut component_operators = Vec::with_capacity(reconstruction.ambient_dim());
    for component_index in 0..reconstruction.ambient_dim() {
        let Some(rows) = reconstruction.component_rows(component_index) else {
            return Err(format!(
                "missing reconstructed barycenter component rows for component {component_index}"
            ));
        };
        component_operators.push(compose_sparse_rows(rows, ambient_operator)?);
    }

    Ok(ComposedBarycenterOperators {
        ambient_dim: reconstruction.ambient_dim(),
        cell_count: reconstruction.cell_count(),
        component_operators,
    })
}

fn compose_sparse_rows(
    lhs_rows: &[Vec<(usize, f64)>],
    rhs: &SparseRowLinearOperator,
) -> Result<SparseRowLinearOperator, String> {
    let mut composed_rows = Vec::with_capacity(lhs_rows.len());
    for row in lhs_rows {
        let mut accum = BTreeMap::<usize, f64>::new();
        for (mid_index, lhs_value) in row {
            if *mid_index >= rhs.nrows() {
                return Err(format!(
                    "composition index {} exceeds rhs row count {}",
                    mid_index,
                    rhs.nrows()
                ));
            }
            for (col, rhs_value) in &rhs.rows[*mid_index] {
                *accum.entry(*col).or_insert(0.0) += *lhs_value * *rhs_value;
            }
        }
        composed_rows.push(
            accum
                .into_iter()
                .filter(|(_, value)| value.abs() > EPS)
                .collect(),
        );
    }
    SparseRowLinearOperator::new(rhs.ncols, composed_rows)
}

fn exact_reconstructed_barycenter_variances(
    gmrf: &mut Gmrf,
    operators: &ComposedBarycenterOperators,
) -> Result<Vec<GmrfVector>, GmrfError> {
    operators
        .component_operators
        .iter()
        .map(|operator| exact_transformed_variances(gmrf, operator))
        .collect()
}

fn zero_reconstructed_barycenter_variance_fields(
    operators: &ComposedBarycenterOperators,
) -> ReconstructedBarycenterVarianceFields {
    let zero_components = iter::repeat_with(|| FeecVector::zeros(operators.cell_count))
        .take(operators.ambient_dim)
        .collect::<Vec<_>>();
    ReconstructedBarycenterVarianceFields {
        prior_components: zero_components.clone(),
        posterior_components: zero_components,
        prior_trace: FeecVector::zeros(operators.cell_count),
        posterior_trace: FeecVector::zeros(operators.cell_count),
        trace_ratio: FeecVector::zeros(operators.cell_count),
    }
}

fn build_reconstructed_barycenter_variance_fields(
    prior_components: &[GmrfVector],
    posterior_components: &[GmrfVector],
) -> ReconstructedBarycenterVarianceFields {
    let prior_components = prior_components
        .iter()
        .map(gmrf_vec_to_feec)
        .collect::<Vec<_>>();
    let posterior_components = posterior_components
        .iter()
        .map(gmrf_vec_to_feec)
        .collect::<Vec<_>>();
    let cell_count = prior_components.first().map_or(0, FeecVector::len);
    let prior_trace = sum_component_vectors(&prior_components, cell_count);
    let posterior_trace = sum_component_vectors(&posterior_components, cell_count);
    let trace_ratio = ratio_vector(&posterior_trace, &prior_trace);

    ReconstructedBarycenterVarianceFields {
        prior_components,
        posterior_components,
        prior_trace,
        posterior_trace,
        trace_ratio,
    }
}

fn sum_component_vectors(components: &[FeecVector], length: usize) -> FeecVector {
    let mut sum = FeecVector::zeros(length);
    for component in components {
        sum += component;
    }
    sum
}

fn components_to_vtk_vectors(components: &[FeecVector]) -> Vec<[f64; 3]> {
    let cell_count = components.first().map_or(0, FeecVector::len);
    (0..cell_count)
        .map(|cell_index| {
            [
                components
                    .first()
                    .map_or(0.0, |component| component[cell_index]),
                components
                    .get(1)
                    .map_or(0.0, |component| component[cell_index]),
                components
                    .get(2)
                    .map_or(0.0, |component| component[cell_index]),
            ]
        })
        .collect()
}

fn ratio_vector(numerator: &FeecVector, denominator: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(
        numerator.len(),
        (0..numerator.len()).map(|i| safe_ratio(numerator[i], denominator[i])),
    )
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn clamp_small_negative_variance(
    value: f64,
    scale: f64,
    message: &'static str,
) -> Result<f64, GmrfError> {
    let tol = EXACT_VARIANCE_TOLERANCE * scale.max(1.0);
    if value >= -tol {
        Ok(value.max(0.0))
    } else {
        Err(GmrfError::NumericalInstability(message))
    }
}

fn mass_inner_product(lhs: &FeecVector, rhs: &FeecVector, mass_u: &FeecCsr) -> f64 {
    let weighted_rhs = mass_u * rhs;
    lhs.dot(&weighted_rhs)
}

fn absolute_difference(lhs: &FeecVector, rhs: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(lhs.len(), (0..lhs.len()).map(|i| (lhs[i] - rhs[i]).abs()))
}

fn max_abs(vector: &FeecVector) -> f64 {
    vector.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn mean_abs(vector: &FeecVector) -> f64 {
    if vector.is_empty() {
        0.0
    } else {
        vector.iter().map(|value| value.abs()).sum::<f64>() / vector.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torus_1form_pde_conditioning::default_torus_shell_resolution_1_mesh_path;
    use manifold::{gen::cartesian::CartesianMeshInfo, io::gmsh::gmsh2coord_complex};
    use std::fs;

    fn selector_from_indices(dimension: usize, indices: &[usize]) -> FeecCsr {
        let selector = gmrf_core::observation::observation_selector(dimension, indices);
        let mut coo = common::linalg::nalgebra::CooMatrix::new(selector.nrows(), selector.ncols());
        for (row, col, value) in selector.triplet_iter() {
            coo.push(row, col, *value);
        }
        FeecCsr::from(&coo)
    }

    #[test]
    fn coexact_transform_has_expected_shape_in_3d() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let transform = build_coexact_1form_transform(&topology, &metric, &hodge.mass_u);

        assert_eq!(transform.nrows(), topology.skeleton(1).len());
        assert_eq!(transform.ncols(), topology.skeleton(2).len());
    }

    #[test]
    fn feec_harmonic_basis_default_for_torus_has_requested_dimension() {
        let _lock = lock_feec_harmonic_tests();
        let mesh_bytes = fs::read(default_torus_shell_resolution_1_mesh_path())
            .expect("torus mesh should be readable");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);

        let harmonic_basis = compute_harmonic_basis_1form(&topology, &metric, 2, None)
            .expect("harmonic basis should compute through FEEC");
        let harmonic_basis =
            mass_orthonormalize_harmonic_basis_1form(&harmonic_basis, &hodge.mass_u)
                .expect("harmonic basis should orthonormalize");

        assert_eq!(harmonic_basis.ncols(), 2);
        let gram = harmonic_basis.transpose() * (&hodge.mass_u * &harmonic_basis);
        for i in 0..gram.nrows() {
            for j in 0..gram.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (gram[(i, j)] - expected).abs() <= 1e-8,
                    "harmonic gram entry ({i},{j}) expected {expected}, got {}",
                    gram[(i, j)]
                );
            }
        }
    }

    #[test]
    fn exact_branch_recovers_exact_truth_on_torus() {
        let _lock = lock_feec_harmonic_tests();
        let mesh_bytes = fs::read(default_torus_shell_resolution_1_mesh_path())
            .expect("torus mesh should be readable");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);
        let exact_transform = build_exact_1form_transform(&topology);
        let phi = FeecVector::from_iterator(
            topology.vertices().len(),
            (0..topology.vertices().len())
                .map(|i| ((i + 1) as f64) / topology.vertices().len() as f64),
        );
        let truth = &exact_transform * &phi;
        let observation_matrix = selector_from_indices(
            truth.len(),
            &[0, 1, 2, 3, 4, 5.min(truth.len().saturating_sub(1))],
        );

        let result = run_hodge_1form_conditioning(&Hodge1FormConditioningConfig {
            topology: &topology,
            coords: &coords,
            metric: &metric,
            truth: &truth,
            observation_matrix: &observation_matrix,
            kappa: 2.0,
            tau: 1.0,
            noise_variance: 1e-8,
            harmonic_dim: 2,
            harmonic_basis_override: None,
        })
        .expect("conditioning should succeed");

        assert!(result.exact.max_abs_observation_error <= 1e-4);
        assert!(result
            .exact
            .posterior_mean
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .exact
            .prior_variance
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .exact
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
        assert!(
            result
                .exact
                .posterior_bias_diagnostics
                .curl_residual_relative
                <= 1e-10,
            "exact posterior mean curl residual too large: {:?}",
            result.exact.posterior_bias_diagnostics
        );
        assert_eq!(
            result.exact.reconstructed_barycenter_variance.ambient_dim(),
            coords.dim()
        );
        assert_eq!(
            result.exact.reconstructed_barycenter_variance.cell_count(),
            topology.cells().len()
        );
        assert!(result
            .exact
            .reconstructed_barycenter_variance
            .posterior_trace
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn coexact_branch_posterior_mean_has_small_coclosed_defect_on_torus() {
        let _lock = lock_feec_harmonic_tests();
        let mesh_bytes = fs::read(default_torus_shell_resolution_1_mesh_path())
            .expect("torus mesh should be readable");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let coexact_transform = build_coexact_1form_transform(&topology, &metric, &hodge.mass_u);
        let psi = FeecVector::from_iterator(
            topology.skeleton(2).len(),
            (0..topology.skeleton(2).len())
                .map(|i| ((i + 1) as f64) / topology.skeleton(2).len() as f64),
        );
        let truth = &coexact_transform * &psi;
        let observation_matrix = selector_from_indices(
            truth.len(),
            &[0, 1, 2, 3, 4, 5.min(truth.len().saturating_sub(1))],
        );

        let result = run_hodge_1form_conditioning(&Hodge1FormConditioningConfig {
            topology: &topology,
            coords: &coords,
            metric: &metric,
            truth: &truth,
            observation_matrix: &observation_matrix,
            kappa: 2.0,
            tau: 1.0,
            noise_variance: 1e-8,
            harmonic_dim: 2,
            harmonic_basis_override: None,
        })
        .expect("conditioning should succeed");

        assert!(
            result
                .coexact
                .posterior_bias_diagnostics
                .coclosed_residual_relative
                <= 3e-2,
            "coexact posterior mean coclosed defect regressed: {:?}",
            result.coexact.posterior_bias_diagnostics
        );
        assert!(result
            .coexact
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn harmonic_branch_recovers_harmonic_truth_on_torus() {
        let _lock = lock_feec_harmonic_tests();
        let mesh_bytes = fs::read(default_torus_shell_resolution_1_mesh_path())
            .expect("torus mesh should be readable");
        let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
        let metric = coords.to_edge_lengths(&topology);
        let hodge = build_hodge_laplacian_1form(&topology, &metric);
        let harmonic_basis = compute_harmonic_basis_1form(&topology, &metric, 2, None)
            .expect("harmonic basis should compute through FEEC");
        let harmonic_basis =
            mass_orthonormalize_harmonic_basis_1form(&harmonic_basis, &hodge.mass_u)
                .expect("harmonic basis should orthonormalize");
        let truth = harmonic_basis.column(0).into_owned().scale(0.7)
            + harmonic_basis.column(1).into_owned().scale(-0.4);
        let observation_matrix = selector_from_indices(
            truth.len(),
            &[0, 1, 2, 3, 4, 5.min(truth.len().saturating_sub(1))],
        );

        let result = run_hodge_1form_conditioning(&Hodge1FormConditioningConfig {
            topology: &topology,
            coords: &coords,
            metric: &metric,
            truth: &truth,
            observation_matrix: &observation_matrix,
            kappa: 2.0,
            tau: 1.0,
            noise_variance: 1e-8,
            harmonic_dim: 2,
            harmonic_basis_override: None,
        })
        .expect("conditioning should succeed");

        assert!(result.harmonic.max_abs_observation_error <= 1e-4);
        assert_eq!(result.harmonic.latent_dimension, 2);
        assert_eq!(
            result
                .harmonic
                .reconstructed_barycenter_variance
                .cell_count(),
            topology.cells().len()
        );
        assert!(result
            .harmonic
            .reconstructed_barycenter_variance
            .trace_ratio
            .iter()
            .all(|value| value.is_finite()));
    }
}
