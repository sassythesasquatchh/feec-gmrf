use crate::sparse::{
    core_triplet_to_feec_csr, feec_csr_to_gmrf, feec_vec_to_gmrf, gmrf_vec_to_feec,
    lift_vector_with_layout,
};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_core::{
    GaussianPriorSpec, LinearGaussianMeasurementSpec, LinearUncertainInputSpec,
    RepresentationPreference, SparseTripletMatrix, StateLayout,
};
use formoniq::problems::linear_uq::ReducedLinearPdeSystem;
use gmrf_core::types::{
    CooMatrix as GmrfCoo, DenseMatrix as GmrfDenseMatrix, SparseCholeskyFactor,
    SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{
    apply_gaussian_observations, apply_gaussian_observations_with_precision, Gmrf,
    LinearObservationStackBuilder, SparseRowOperator,
};
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::mem::size_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPdeVarianceMode {
    Exact,
    Rbmc,
    RbmcClipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearPdeVarianceConfig {
    pub mode: LinearPdeVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
}

impl Default for LinearPdeVarianceConfig {
    fn default() -> Self {
        Self {
            mode: LinearPdeVarianceMode::Exact,
            num_rbmc_probes: 64,
            rbmc_batch_count: 4,
            rng_seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearPdeUqSolverConfig {
    pub variance: LinearPdeVarianceConfig,
    pub stabilize_precision: bool,
    pub log_diagnostics: bool,
}

impl Default for LinearPdeUqSolverConfig {
    fn default() -> Self {
        Self {
            variance: LinearPdeVarianceConfig::default(),
            stabilize_precision: true,
            log_diagnostics: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedInputRepresentation {
    Collapsed,
    Latent,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputRepresentationDebug {
    pub name: String,
    pub representation: ResolvedInputRepresentation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearPdeFactorizationDebug {
    pub dimension: usize,
    pub matrix_nnz: usize,
    pub matrix_lower_triangle_nnz: usize,
    pub factor_nnz: usize,
    pub fill_in_ratio_vs_lower_triangle: f64,
    pub factor_numeric_values_mib: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearPdeUqDebug {
    pub input_representations: Vec<InputRepresentationDebug>,
    pub joint_dimension: usize,
    pub prior_factorization: LinearPdeFactorizationDebug,
    pub posterior_factorization: LinearPdeFactorizationDebug,
}

#[derive(Debug, Clone)]
pub struct LinearPdeUqProblem {
    pub state_prior: GaussianPriorSpec,
    pub system: ReducedLinearPdeSystem,
    pub uncertain_inputs: Vec<LinearUncertainInputSpec>,
    pub physical_measurements: Vec<LinearGaussianMeasurementSpec>,
    pub derived_quantities: Vec<LinearPdeDerivedQuantitySpec>,
    pub pde_variance: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct LinearPdeDerivedQuantitySpec {
    pub name: String,
    pub operator: SparseRowOperator,
}

#[derive(Debug, Clone)]
pub struct LinearPdeDerivedMarginalResult {
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
}

#[derive(Debug, Clone)]
pub struct LinearPdeUqResult {
    pub posterior_mean: FeecVector,
    pub posterior_variance: FeecVector,
    pub prior_variance: FeecVector,
    pub derived_variances: BTreeMap<String, LinearPdeDerivedMarginalResult>,
    pub reduced_posterior_mean: FeecVector,
    pub reduced_posterior_variance: FeecVector,
    pub pde_residual_mean: FeecVector,
    pub debug: LinearPdeUqDebug,
}

pub struct LinearPdeJointPosterior {
    pub posterior: Gmrf,
    pub derived_quantities: BTreeMap<String, SparseRowOperator>,
    pub state_dimension: usize,
    pub joint_dimension: usize,
}

struct PreparedLinearPdeProblem {
    state_dimension: usize,
    joint_dimension: usize,
    state_mean: FeecVector,
    centered_residual_bias: FeecVector,
    pde_operator: GmrfSparseMatrix,
    prior_precision: GmrfSparseMatrix,
    posterior_precision: GmrfSparseMatrix,
    information: GmrfVector,
    derived_quantities: BTreeMap<String, SparseRowOperator>,
    input_representations: Vec<InputRepresentationDebug>,
}

pub fn solve_linear_pde_uq(problem: &LinearPdeUqProblem) -> Result<LinearPdeUqResult, String> {
    solve_linear_pde_uq_with_config(problem, &LinearPdeUqSolverConfig::default())
}

pub fn build_linear_pde_joint_posterior_with_config(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<LinearPdeJointPosterior, String> {
    let prepared = prepare_linear_pde_problem(problem, config)?;
    let posterior_precision =
        maybe_stabilize_precision(&prepared.posterior_precision, config.stabilize_precision)?;
    let (q_factor, _) =
        factorize_precision_with_diagnostics("posterior", &posterior_precision, config)?;
    let posterior = Gmrf::from_information_and_precision_with_sqrt(
        prepared.information,
        posterior_precision,
        q_factor,
    )
    .map_err(|err| err.to_string())?;

    Ok(LinearPdeJointPosterior {
        posterior,
        derived_quantities: prepared.derived_quantities,
        state_dimension: prepared.state_dimension,
        joint_dimension: prepared.joint_dimension,
    })
}

pub fn solve_linear_pde_uq_with_config(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<LinearPdeUqResult, String> {
    let prepared = prepare_linear_pde_problem(problem, config)?;
    let state_dim = prepared.state_dimension;
    let joint_dimension = prepared.joint_dimension;
    let state_mean = prepared.state_mean;
    let centered_residual_bias = prepared.centered_residual_bias;
    let pde_operator = prepared.pde_operator;
    let prior_gmrf_precision = prepared.prior_precision;
    let mut posterior_precision = prepared.posterior_precision;
    let information = prepared.information;
    let derived_quantities = prepared.derived_quantities;
    let (prior_factor, prior_factorization) =
        factorize_precision_with_diagnostics("prior", &prior_gmrf_precision, config)?;
    let mut prior = gmrf_from_zero_mean_precision(prior_gmrf_precision, prior_factor)
        .map_err(|err| format!("failed to build prior GMRF: {err}"))?;
    log_diagnostics(
        config,
        format_args!(
            "prior_variances_start mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );
    let prior_variances = estimate_variances(&mut prior, &config.variance, None)?;
    let prior_derived_variances =
        estimate_derived_variances(&mut prior, &derived_quantities, &config.variance, None)?;
    log_diagnostics(
        config,
        format_args!(
            "prior_variances_done mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );
    drop(prior);
    log_diagnostics(
        config,
        format_args!("prior_factor_released dimension={joint_dimension}"),
    );

    posterior_precision =
        maybe_stabilize_precision(&posterior_precision, config.stabilize_precision)?;
    let (q_factor, posterior_factorization) =
        factorize_precision_with_diagnostics("posterior", &posterior_precision, config)?;
    let mut posterior =
        Gmrf::from_information_and_precision_with_sqrt(information, posterior_precision, q_factor)
            .map_err(|err| err.to_string())?;
    log_diagnostics(
        config,
        format_args!(
            "posterior_variances_start mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );
    let posterior_variances =
        estimate_variances(&mut posterior, &config.variance, Some(&prior_variances))?;
    let posterior_derived_variances = estimate_derived_variances(
        &mut posterior,
        &derived_quantities,
        &config.variance,
        Some(&prior_derived_variances),
    )?;
    log_diagnostics(
        config,
        format_args!(
            "posterior_variances_done mode={} dimension={}",
            variance_mode_name(config.variance.mode),
            joint_dimension
        ),
    );

    let reduced_centered_mean = gmrf_vec_to_feec(&GmrfVector::from_vec(
        posterior.mean().as_slice()[0..state_dim].to_vec(),
    ));
    let reduced_posterior_mean = &reduced_centered_mean + &state_mean;
    let reduced_posterior_variance = gmrf_vec_to_feec(&GmrfVector::from_vec(
        posterior_variances.as_slice()[0..state_dim].to_vec(),
    ));
    let reduced_prior_variance = gmrf_vec_to_feec(&GmrfVector::from_vec(
        prior_variances.as_slice()[0..state_dim].to_vec(),
    ));

    let posterior_mean = lift_vector_with_layout(&problem.system.layout, &reduced_posterior_mean)?;
    let posterior_variance =
        lift_variances_with_layout(&problem.system.layout, &reduced_posterior_variance)?;
    let prior_variance =
        lift_variances_with_layout(&problem.system.layout, &reduced_prior_variance)?;
    let derived_variances =
        feec_derived_variances(&prior_derived_variances, &posterior_derived_variances);
    let pde_residual_mean =
        gmrf_vec_to_feec(&sparse_matvec(&pde_operator, posterior.mean())?) + centered_residual_bias;

    Ok(LinearPdeUqResult {
        posterior_mean,
        posterior_variance,
        prior_variance,
        derived_variances,
        reduced_posterior_mean,
        reduced_posterior_variance,
        pde_residual_mean,
        debug: LinearPdeUqDebug {
            input_representations: prepared.input_representations,
            joint_dimension,
            prior_factorization,
            posterior_factorization,
        },
    })
}

fn prepare_linear_pde_problem(
    problem: &LinearPdeUqProblem,
    config: &LinearPdeUqSolverConfig,
) -> Result<PreparedLinearPdeProblem, String> {
    validate_problem(problem)?;
    validate_solver_config(config)?;

    let state_dimension = problem.system.state_dimension();
    let residual_dim = problem.system.residual_dimension();
    let mut centered_residual_bias = FeecVector::from_vec(problem.system.residual_bias.clone());
    let state_operator = core_triplet_to_feec_csr(&problem.system.operator);
    let state_mean = FeecVector::from_vec(problem.state_prior.mean.clone());
    centered_residual_bias += &state_operator * &state_mean;

    let resolved = resolve_input_representations(&problem.uncertain_inputs, residual_dim)?;
    let collapsed_count = resolved
        .iter()
        .filter(|(_, representation, _)| *representation == ResolvedInputRepresentation::Collapsed)
        .count();
    if collapsed_count > 1 {
        return Err(
            "at most one uncertain input may be collapsed in v1; keep the remaining inputs latent"
                .to_string(),
        );
    }
    if collapsed_count == 0 && problem.pde_variance.is_none() {
        return Err(
            "pde_variance must be provided unless a single uncertain input is collapsed"
                .to_string(),
        );
    }
    if collapsed_count > 0 && problem.pde_variance.is_some() {
        return Err(
            "pde_variance cannot be combined with collapsed uncertain inputs in v1; keep the input latent or remove the scalar PDE variance".to_string(),
        );
    }

    let mut block_precisions = vec![problem.state_prior.precision.clone()];
    let mut latent_blocks = Vec::<(usize, FeecCsr)>::new();
    let mut input_representations = Vec::with_capacity(problem.uncertain_inputs.len());
    let mut joint_dimension = state_dimension;
    let mut collapsed_precision = None;

    for (input, representation, maybe_precision) in resolved {
        let operator = core_triplet_to_feec_csr(&input.operator);
        let mean = FeecVector::from_vec(input.prior.mean.clone());
        centered_residual_bias += &operator * &mean;
        input_representations.push(InputRepresentationDebug {
            name: input.name.clone(),
            representation,
        });

        match representation {
            ResolvedInputRepresentation::Latent => {
                let offset = joint_dimension;
                joint_dimension += input.prior.dimension();
                latent_blocks.push((offset, operator));
                block_precisions.push(input.prior.precision.clone());
            }
            ResolvedInputRepresentation::Collapsed => {
                let precision = maybe_precision.ok_or_else(|| {
                    format!(
                        "uncertain input `{}` was resolved as collapsed without a residual precision",
                        input.name
                    )
                })?;
                collapsed_precision = Some(core_triplet_to_feec_csr(&precision));
            }
        }
    }

    let derived_quantities = restrict_derived_quantities(
        &problem.derived_quantities,
        &problem.system.layout,
        joint_dimension,
    )?;

    let prior_precision = maybe_stabilize_precision(
        &block_diag_precision(&block_precisions),
        config.stabilize_precision,
    )?;
    let pde_operator = build_joint_operator(
        joint_dimension,
        &[(0, state_operator.clone())]
            .into_iter()
            .chain(
                latent_blocks
                    .iter()
                    .map(|(offset, operator)| (*offset, operator.clone())),
            )
            .collect::<Vec<_>>(),
    );
    let pde_bias = feec_vec_to_gmrf(&centered_residual_bias);
    let zero_observations = GmrfVector::zeros(residual_dim);

    let mut builder = LinearObservationStackBuilder::new(joint_dimension);
    if let Some(variance) = problem.pde_variance {
        builder
            .push_block(
                0,
                &pde_operator,
                zero_observations.as_slice(),
                pde_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())?;
    }

    for measurement in problem
        .system
        .boundary_measurements
        .iter()
        .chain(problem.physical_measurements.iter())
    {
        let centered =
            restrict_measurement_to_reduced(measurement, &problem.system.layout, &state_mean)?;
        builder
            .push_block(
                0,
                &feec_csr_to_gmrf(&centered.operator),
                centered.observations.as_slice(),
                centered.bias.as_slice(),
                centered.variance,
            )
            .map_err(|err| err.to_string())?;
    }

    let stacked = builder.finish();
    log_diagnostics(
        config,
        format_args!(
            "posterior_assembly_ready dimension={} observation_rows={} observation_nnz={}",
            joint_dimension,
            stacked.matrix.nrows(),
            stacked.matrix.nnz()
        ),
    );
    let (mut posterior_precision, mut information) = if stacked.matrix.nrows() == 0 {
        (prior_precision.clone(), GmrfVector::zeros(joint_dimension))
    } else {
        apply_gaussian_observations(
            &prior_precision,
            &stacked.matrix,
            &stacked.observations,
            Some(&stacked.bias),
            stacked.noise_variance,
        )
    };

    if let Some(collapsed_precision) = collapsed_precision {
        let (updated_precision, collapsed_information) = apply_gaussian_observations_with_precision(
            &posterior_precision,
            &pde_operator,
            &zero_observations,
            Some(&pde_bias),
            &feec_csr_to_gmrf(&collapsed_precision),
        );
        posterior_precision = updated_precision;
        information += collapsed_information;
    }

    let prior_precision = if stacked.matrix.nrows() == 0 {
        prior_precision.clone()
    } else {
        prior_precision
    };

    Ok(PreparedLinearPdeProblem {
        state_dimension,
        joint_dimension,
        state_mean,
        centered_residual_bias,
        pde_operator,
        prior_precision,
        posterior_precision,
        information,
        derived_quantities,
        input_representations,
    })
}

struct CenteredMeasurement {
    operator: FeecCsr,
    observations: FeecVector,
    bias: FeecVector,
    variance: f64,
}

fn validate_problem(problem: &LinearPdeUqProblem) -> Result<(), String> {
    problem.state_prior.validate()?;
    if problem.state_prior.dimension() != problem.system.state_dimension() {
        return Err(format!(
            "state prior dimension {} must match reduced state dimension {}",
            problem.state_prior.dimension(),
            problem.system.state_dimension()
        ));
    }
    if problem.state_prior.mean.len() != problem.system.state_dimension() {
        return Err("state prior mean must be defined on the reduced state".to_string());
    }
    if problem.system.residual_bias.len() != problem.system.residual_dimension() {
        return Err(format!(
            "residual bias length {} must match residual dimension {}",
            problem.system.residual_bias.len(),
            problem.system.residual_dimension()
        ));
    }
    if let Some(variance) = problem.pde_variance {
        if !variance.is_finite() || variance <= 0.0 {
            return Err("pde_variance must be finite and positive when provided".to_string());
        }
    }
    for input in &problem.uncertain_inputs {
        input.validate(problem.system.residual_dimension())?;
    }
    for measurement in problem
        .system
        .boundary_measurements
        .iter()
        .chain(problem.physical_measurements.iter())
    {
        measurement.validate(problem.system.layout.full_dimension)?;
    }
    for derived in &problem.derived_quantities {
        if derived.operator.ncols != problem.system.layout.full_dimension {
            return Err(format!(
                "derived operator `{}` column count {} must match full state dimension {}",
                derived.name, derived.operator.ncols, problem.system.layout.full_dimension
            ));
        }
    }
    Ok(())
}

fn validate_solver_config(config: &LinearPdeUqSolverConfig) -> Result<(), String> {
    match config.variance.mode {
        LinearPdeVarianceMode::Exact => Ok(()),
        LinearPdeVarianceMode::Rbmc | LinearPdeVarianceMode::RbmcClipped => {
            if config.variance.num_rbmc_probes == 0 {
                return Err("variance.num_rbmc_probes must be >= 1".to_string());
            }
            if config.variance.rbmc_batch_count == 0 {
                return Err("variance.rbmc_batch_count must be >= 1".to_string());
            }
            Ok(())
        }
    }
}

fn maybe_stabilize_precision(
    precision: &GmrfSparseMatrix,
    stabilize: bool,
) -> Result<GmrfSparseMatrix, String> {
    if stabilize {
        stabilize_spd_precision(precision)
    } else {
        Ok(precision.clone())
    }
}

fn resolve_input_representations(
    inputs: &[LinearUncertainInputSpec],
    residual_dimension: usize,
) -> Result<
    Vec<(
        LinearUncertainInputSpec,
        ResolvedInputRepresentation,
        Option<SparseTripletMatrix>,
    )>,
    String,
> {
    let mut forced_collapsed = Vec::new();
    let mut auto_candidates = Vec::new();
    let mut collapsed_available = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let available = available_collapsed_precision(input, residual_dimension);
        if matches!(input.preference, RepresentationPreference::ForceCollapsed)
            && available.is_none()
        {
            return Err(format!(
                "uncertain input `{}` cannot be forced collapsed because no sparse residual precision is available",
                input.name
            ));
        }
        if matches!(input.preference, RepresentationPreference::ForceCollapsed) {
            forced_collapsed.push(index);
        } else if matches!(input.preference, RepresentationPreference::Auto) && available.is_some()
        {
            auto_candidates.push(index);
        }
        collapsed_available.push(available);
    }

    if forced_collapsed.len() > 1 {
        return Err("only one uncertain input may be forced collapsed in v1".to_string());
    }

    let selected_auto = if forced_collapsed.is_empty() && auto_candidates.len() == 1 {
        Some(auto_candidates[0])
    } else {
        None
    };

    Ok(inputs
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, input)| {
            let representation =
                if forced_collapsed.contains(&index) || selected_auto == Some(index) {
                    ResolvedInputRepresentation::Collapsed
                } else {
                    ResolvedInputRepresentation::Latent
                };
            (input, representation, collapsed_available[index].clone())
        })
        .collect())
}

fn available_collapsed_precision(
    input: &LinearUncertainInputSpec,
    residual_dimension: usize,
) -> Option<SparseTripletMatrix> {
    if let Some(precision) = &input.collapsed_precision {
        return Some(precision.clone());
    }
    if input.operator.nrows() == residual_dimension
        && input.operator.ncols() == residual_dimension
        && input.prior.dimension() == residual_dimension
        && is_signed_identity(&input.operator)
    {
        return Some(input.prior.precision.clone());
    }
    None
}

fn is_signed_identity(matrix: &feg_core::SparseTripletMatrix) -> bool {
    if matrix.nrows() != matrix.ncols() {
        return false;
    }
    let mut seen = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row != col {
            return false;
        }
        if seen[row] != 0.0 {
            return false;
        }
        if value.abs() != 1.0 {
            return false;
        }
        seen[row] = value.abs();
    }
    seen.iter().all(|value| (*value - 1.0).abs() <= 1e-12)
}

fn build_joint_operator(dimension: usize, blocks: &[(usize, FeecCsr)]) -> GmrfSparseMatrix {
    let row_count = blocks.first().map(|(_, block)| block.nrows()).unwrap_or(0);
    let mut coo = GmrfCoo::new(row_count, dimension);
    for (offset, block) in blocks {
        for (row, col, value) in block.triplet_iter() {
            coo.push(row, offset + col, *value);
        }
    }
    GmrfSparseMatrix::from(&coo)
}

fn block_diag_precision(blocks: &[feg_core::SparseTripletMatrix]) -> GmrfSparseMatrix {
    let dimension = blocks.iter().map(|block| block.nrows()).sum();
    let mut coo = GmrfCoo::new(dimension, dimension);
    let mut offset = 0;
    for block in blocks {
        for (row, col, value) in block.triplet_iter() {
            coo.push(offset + row, offset + col, value);
        }
        offset += block.nrows();
    }
    GmrfSparseMatrix::from(&coo)
}

fn stabilize_spd_precision(precision: &GmrfSparseMatrix) -> Result<GmrfSparseMatrix, String> {
    if precision.cholesky_sqrt_lower().is_ok() {
        return Ok(precision.clone());
    }

    let symmetrized = symmetrize_precision(precision);
    if symmetrized.cholesky_sqrt_lower().is_ok() {
        return Ok(symmetrized);
    }

    let base = symmetrized;
    let (min_diag, max_abs_diag) = diagonal_stats(&base);
    let mut shift = if min_diag.is_finite() && min_diag <= 0.0 {
        (-min_diag) + max_abs_diag * 1e-8
    } else {
        max_abs_diag * 1e-12
    }
    .max(1e-10);
    let mut last_error = "precision matrix is not positive definite".to_string();
    for _ in 0..12 {
        let shifted = add_diagonal_shift(&base, shift);
        match shifted.cholesky_sqrt_lower() {
            Ok(_) => return Ok(shifted),
            Err(err) => {
                last_error = err.to_string();
                shift *= 10.0;
            }
        }
    }
    Err(last_error)
}

fn diagonal_stats(matrix: &GmrfSparseMatrix) -> (f64, f64) {
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    (min_diag, max_abs_diag.max(1.0))
}

fn symmetrize_precision(matrix: &GmrfSparseMatrix) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    GmrfSparseMatrix::from(&coo)
}

fn add_diagonal_shift(matrix: &GmrfSparseMatrix, shift: f64) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    GmrfSparseMatrix::from(&coo)
}

fn gmrf_from_zero_mean_precision(
    precision: GmrfSparseMatrix,
    factor: SparseCholeskyFactor,
) -> Result<Gmrf, String> {
    Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision)
        .map_err(|err| err.to_string())
        .map(|gmrf| gmrf.with_precision_sqrt(factor))
}

fn factorize_precision_with_diagnostics(
    label: &str,
    precision: &GmrfSparseMatrix,
    config: &LinearPdeUqSolverConfig,
) -> Result<(SparseCholeskyFactor, LinearPdeFactorizationDebug), String> {
    let matrix_nnz = precision.nnz();
    let matrix_lower_triangle_nnz = lower_triangle_nnz(precision);
    log_diagnostics(
        config,
        format_args!(
            "{label}_factorization_start dimension={} matrix_nnz={} lower_triangle_nnz={}",
            precision.nrows(),
            matrix_nnz,
            matrix_lower_triangle_nnz
        ),
    );
    let factor = precision
        .cholesky_sqrt_lower()
        .map_err(|err| err.to_string())?;
    let factorization = LinearPdeFactorizationDebug {
        dimension: precision.nrows(),
        matrix_nnz,
        matrix_lower_triangle_nnz,
        factor_nnz: factor.nnz(),
        fill_in_ratio_vs_lower_triangle: factor.nnz() as f64
            / matrix_lower_triangle_nnz.max(1) as f64,
        factor_numeric_values_mib: factor.nnz() as f64 * size_of::<f64>() as f64
            / (1024.0 * 1024.0),
    };
    log_diagnostics(
        config,
        format_args!(
            "{label}_factorization_done dimension={} matrix_nnz={} lower_triangle_nnz={} factor_nnz={} fill_in_vs_lower={:.3}x factor_values_mib={:.3}",
            factorization.dimension,
            factorization.matrix_nnz,
            factorization.matrix_lower_triangle_nnz,
            factorization.factor_nnz,
            factorization.fill_in_ratio_vs_lower_triangle,
            factorization.factor_numeric_values_mib
        ),
    );
    Ok((factor, factorization))
}

fn lower_triangle_nnz(matrix: &GmrfSparseMatrix) -> usize {
    matrix
        .triplet_iter()
        .filter(|(row, col, _)| row >= col)
        .count()
}

fn log_diagnostics(config: &LinearPdeUqSolverConfig, args: std::fmt::Arguments<'_>) {
    if config.log_diagnostics {
        eprintln!("[linear_pde] {args}");
    }
}

fn variance_mode_name(mode: LinearPdeVarianceMode) -> &'static str {
    match mode {
        LinearPdeVarianceMode::Exact => "exact",
        LinearPdeVarianceMode::Rbmc => "rbmc",
        LinearPdeVarianceMode::RbmcClipped => "rbmc-clipped",
    }
}

fn estimate_variances(
    gmrf: &mut Gmrf,
    config: &LinearPdeVarianceConfig,
    prior_variances: Option<&GmrfVector>,
) -> Result<GmrfVector, String> {
    let raw = match config.mode {
        LinearPdeVarianceMode::Exact => exact_variances(gmrf)?,
        LinearPdeVarianceMode::Rbmc | LinearPdeVarianceMode::RbmcClipped => {
            rbmc_variances(gmrf, config)?
        }
    };
    match (config.mode, prior_variances) {
        (LinearPdeVarianceMode::RbmcClipped, Some(prior)) => Ok(clip_vector_to_prior(prior, &raw)),
        _ => Ok(raw),
    }
}

fn estimate_derived_variances(
    gmrf: &mut Gmrf,
    derived_quantities: &BTreeMap<String, SparseRowOperator>,
    config: &LinearPdeVarianceConfig,
    prior_variances: Option<&BTreeMap<String, GmrfVector>>,
) -> Result<BTreeMap<String, GmrfVector>, String> {
    let mut derived_variances = BTreeMap::new();
    for (name, operator) in derived_quantities {
        let raw = match config.mode {
            LinearPdeVarianceMode::Exact => exact_transformed_variances(gmrf, operator)?,
            LinearPdeVarianceMode::Rbmc | LinearPdeVarianceMode::RbmcClipped => {
                rbmc_transformed_variances(gmrf, operator, config)?
            }
        };
        let variances = match (
            config.mode,
            prior_variances.and_then(|prior| prior.get(name)),
        ) {
            (LinearPdeVarianceMode::RbmcClipped, Some(prior)) => clip_vector_to_prior(prior, &raw),
            _ => raw,
        };
        derived_variances.insert(name.clone(), variances);
    }
    Ok(derived_variances)
}

fn exact_variances(gmrf: &mut Gmrf) -> Result<GmrfVector, String> {
    let constraints = GmrfDenseMatrix::zeros(0, gmrf.dimension());
    gmrf.exact_constrained_variance_decomposition(&constraints)
        .map(|decomposition| decomposition.unconstrained_diag)
        .map_err(|err| err.to_string())
}

fn exact_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
) -> Result<GmrfVector, String> {
    let constraints = GmrfDenseMatrix::zeros(0, gmrf.dimension());
    gmrf.exact_transformed_variance_decomposition(operator, &constraints)
        .map(|decomposition| decomposition.unconstrained_diag)
        .map_err(|err| err.to_string())
}

fn rbmc_variances(gmrf: &mut Gmrf, config: &LinearPdeVarianceConfig) -> Result<GmrfVector, String> {
    let batch_sizes = rbmc_batch_sizes(config.num_rbmc_probes, config.rbmc_batch_count);
    let mut batch_variances = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let batch_seed = config.rng_seed.wrapping_add(
            0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((batch_index as u64).wrapping_add(1)),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(batch_seed);
        let variances = gmrf
            .rbmc_variances(batch_size, &mut rng)
            .map_err(|err| err.to_string())?;
        batch_variances.push(stabilize_positive_variances(&variances));
    }
    weighted_average_vectors(&batch_variances, &batch_sizes)
}

fn rbmc_transformed_variances(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    config: &LinearPdeVarianceConfig,
) -> Result<GmrfVector, String> {
    let constraints = GmrfDenseMatrix::zeros(0, gmrf.dimension());
    let batch_sizes = rbmc_batch_sizes(config.num_rbmc_probes, config.rbmc_batch_count);
    let mut batch_variances = Vec::with_capacity(batch_sizes.len());
    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let batch_seed = config.rng_seed.wrapping_add(
            0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((batch_index as u64).wrapping_add(1)),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(batch_seed);
        let variances = gmrf
            .rbmc_transformed_variance_decomposition(operator, &constraints, batch_size, &mut rng)
            .map_err(|err| err.to_string())?;
        batch_variances.push(stabilize_positive_variances(&variances.unconstrained_diag));
    }
    weighted_average_vectors(&batch_variances, &batch_sizes)
}

fn rbmc_batch_sizes(num_probes: usize, batch_count: usize) -> Vec<usize> {
    let batches = batch_count.min(num_probes).max(1);
    let base = num_probes / batches;
    let remainder = num_probes % batches;
    (0..batches)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn stabilize_positive_variances(variances: &GmrfVector) -> GmrfVector {
    GmrfVector::from_iterator(
        variances.len(),
        variances.iter().map(|value| value.max(0.0)),
    )
}

fn clip_vector_to_prior(prior: &GmrfVector, posterior: &GmrfVector) -> GmrfVector {
    GmrfVector::from_iterator(
        prior.len(),
        (0..prior.len()).map(|i| posterior[i].max(0.0).min(prior[i].max(0.0))),
    )
}

fn feec_derived_variances(
    prior: &BTreeMap<String, GmrfVector>,
    posterior: &BTreeMap<String, GmrfVector>,
) -> BTreeMap<String, LinearPdeDerivedMarginalResult> {
    let mut derived = BTreeMap::new();
    for (name, prior_variance) in prior {
        let posterior_variance = posterior
            .get(name)
            .expect("posterior derived variances must align with prior names");
        derived.insert(
            name.clone(),
            LinearPdeDerivedMarginalResult {
                prior_variance: gmrf_vec_to_feec(prior_variance),
                posterior_variance: gmrf_vec_to_feec(posterior_variance),
            },
        );
    }
    derived
}

fn weighted_average_vectors(
    vectors: &[GmrfVector],
    weights: &[usize],
) -> Result<GmrfVector, String> {
    if vectors.is_empty() || weights.is_empty() || vectors.len() != weights.len() {
        return Err("rbmc batch vectors and weights must be non-empty and aligned".to_string());
    }
    let dimension = vectors[0].len();
    if vectors.iter().any(|vector| vector.len() != dimension) {
        return Err("rbmc batch vectors must have a consistent dimension".to_string());
    }
    let total_weight = weights.iter().sum::<usize>();
    if total_weight == 0 {
        return Err("rbmc batch weights must sum to a positive value".to_string());
    }

    let mut average = GmrfVector::zeros(dimension);
    for (vector, weight) in vectors.iter().zip(weights.iter().copied()) {
        average += vector * (weight as f64);
    }
    Ok(average / (total_weight as f64))
}

fn restrict_measurement_to_reduced(
    measurement: &LinearGaussianMeasurementSpec,
    layout: &StateLayout,
    reduced_state_mean: &FeecVector,
) -> Result<CenteredMeasurement, String> {
    let operator = core_triplet_to_feec_csr(&measurement.operator);
    let bias = FeecVector::from_vec(measurement.bias.clone());
    let (reduced_operator, folded_bias) =
        restrict_columns_and_fold_fixed(&operator, &bias, layout)?;
    let centered_bias = folded_bias + &reduced_operator * reduced_state_mean;
    Ok(CenteredMeasurement {
        operator: reduced_operator,
        observations: FeecVector::from_vec(measurement.observations.clone()),
        bias: centered_bias,
        variance: measurement.variance,
    })
}

fn restrict_derived_quantities(
    derived_quantities: &[LinearPdeDerivedQuantitySpec],
    layout: &StateLayout,
    joint_dimension: usize,
) -> Result<BTreeMap<String, SparseRowOperator>, String> {
    let mut restricted = BTreeMap::new();
    for derived in derived_quantities {
        if restricted.contains_key(&derived.name) {
            return Err(format!(
                "derived quantity names must be unique; duplicate `{}`",
                derived.name
            ));
        }
        let reduced = restrict_operator_to_reduced(&derived.operator, layout)?;
        restricted.insert(
            derived.name.clone(),
            state_operator_to_joint(reduced, joint_dimension)?,
        );
    }
    Ok(restricted)
}

fn restrict_operator_to_reduced(
    operator: &SparseRowOperator,
    layout: &StateLayout,
) -> Result<SparseRowOperator, String> {
    if operator.ncols != layout.full_dimension {
        return Err(format!(
            "derived operator column count {} does not match full state dimension {}",
            operator.ncols, layout.full_dimension
        ));
    }

    let reduced_map = reduced_index_map(layout);
    let mut rows = Vec::with_capacity(operator.nrows());
    for row in &operator.rows {
        let mut entries = BTreeMap::<usize, f64>::new();
        for (col, value) in row {
            if let Some(reduced_col) = reduced_map[*col] {
                *entries.entry(reduced_col).or_insert(0.0) += *value;
            }
        }
        rows.push(
            entries
                .into_iter()
                .filter(|(_, value)| value.abs() > 0.0)
                .collect(),
        );
    }
    SparseRowOperator::new(layout.reduced_dimension(), rows).map_err(|err| err.to_string())
}

fn state_operator_to_joint(
    operator: SparseRowOperator,
    joint_dimension: usize,
) -> Result<SparseRowOperator, String> {
    if operator.ncols > joint_dimension {
        return Err(format!(
            "reduced derived operator column count {} exceeds joint dimension {}",
            operator.ncols, joint_dimension
        ));
    }
    SparseRowOperator::new(joint_dimension, operator.rows).map_err(|err| err.to_string())
}

fn restrict_columns_and_fold_fixed(
    full_block: &FeecCsr,
    bias: &FeecVector,
    layout: &StateLayout,
) -> Result<(FeecCsr, FeecVector), String> {
    if full_block.ncols() != layout.full_dimension {
        return Err(format!(
            "full measurement block column count {} does not match layout dimension {}",
            full_block.ncols(),
            layout.full_dimension
        ));
    }
    if full_block.nrows() != bias.len() {
        return Err(format!(
            "measurement bias length {} does not match operator row count {}",
            bias.len(),
            full_block.nrows()
        ));
    }

    let mut reduced = FeecCoo::new(full_block.nrows(), layout.reduced_dimension());
    let mut reduced_bias = bias.clone();
    let reduced_map = reduced_index_map(layout);
    for (row, col, value) in full_block.triplet_iter() {
        if let Some(reduced_col) = reduced_map[col] {
            reduced.push(row, reduced_col, *value);
        } else if let Some(fixed) = layout.fixed_dofs.iter().find(|entry| entry.index == col) {
            reduced_bias[row] += *value * fixed.value;
        }
    }
    Ok((FeecCsr::from(&reduced), reduced_bias))
}

fn reduced_index_map(layout: &StateLayout) -> Vec<Option<usize>> {
    let mut map = vec![None; layout.full_dimension];
    for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
        map[full] = Some(reduced);
    }
    map
}

fn sparse_matvec(matrix: &GmrfSparseMatrix, vector: &GmrfVector) -> Result<GmrfVector, String> {
    if matrix.ncols() != vector.len() {
        return Err(format!(
            "sparse matvec expected vector length {}, got {}",
            matrix.ncols(),
            vector.len()
        ));
    }
    let mut out = GmrfVector::zeros(matrix.nrows());
    for (row, col, value) in matrix.triplet_iter() {
        out[row] += *value * vector[col];
    }
    Ok(out)
}

fn lift_variances_with_layout(
    layout: &StateLayout,
    reduced: &FeecVector,
) -> Result<FeecVector, String> {
    if reduced.len() != layout.reduced_dimension() {
        return Err(format!(
            "reduced variance length {} does not match layout reduced dimension {}",
            reduced.len(),
            layout.reduced_dimension()
        ));
    }
    let mut full = FeecVector::zeros(layout.full_dimension);
    for (reduced_index, full_index) in layout.active_dofs.iter().copied().enumerate() {
        full[full_index] = reduced[reduced_index];
    }
    Ok(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matern_0form::{build_matern_precision_0form, MaternConfig, MaternMassInverse};
    use feg_core::{
        BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, LinearGaussianMeasurementSpec,
        RepresentationPreference, SparseTripletMatrix,
    };
    use formoniq::assemble;
    use formoniq::problems::linear_uq::{
        build_reduced_hodge_laplace_1form_system, build_reduced_laplace_beltrami_system,
    };
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn max_abs_difference(lhs: &FeecVector, rhs: &FeecVector) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max)
    }

    fn to_core_triplets(matrix: &FeecCsr) -> SparseTripletMatrix {
        SparseTripletMatrix::from_triplets(
            matrix.nrows(),
            matrix.ncols(),
            matrix
                .triplet_iter()
                .map(|(row, col, value)| feg_core::SparseTriplet {
                    row,
                    col,
                    value: *value,
                }),
        )
    }

    fn diagonal_precision(dimension: usize, diagonal_value: f64) -> SparseTripletMatrix {
        SparseTripletMatrix::from_triplets(
            dimension,
            dimension,
            (0..dimension).map(|index| feg_core::SparseTriplet {
                row: index,
                col: index,
                value: diagonal_value,
            }),
        )
    }

    fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
        let mut coo = FeecCoo::new(lhs.nrows(), lhs.ncols());
        for (row, col, value) in lhs.triplet_iter() {
            coo.push(row, col, *value);
        }
        for (row, col, value) in rhs.triplet_iter() {
            coo.push(row, col, *value);
        }
        FeecCsr::from(&coo)
    }

    fn build_reduced_1form_whittle_prior(system: &ReducedLinearPdeSystem) -> SparseTripletMatrix {
        let operator = core_triplet_to_feec_csr(&system.operator);
        let mass = core_triplet_to_feec_csr(&system.state_mass);
        let mass_inverse = core_triplet_to_feec_csr(
            system
                .state_mass_inverse
                .as_ref()
                .expect("mixed 1-form reduced system should expose an NC1 projected inverse"),
        );
        let a = add_sparse(&operator, &mass);
        let precision = &a.transpose() * &(&mass_inverse * &a);
        to_core_triplets(&precision)
    }

    #[test]
    fn collapsed_and_latent_forcing_agree_for_identity_residual_map() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let forcing_mean = vec![1.0; system.residual_dimension()];
        let forcing_precision = diagonal_precision(system.residual_dimension(), 4.0);
        let residual_variance = 0.5;
        let collapsed_precision = diagonal_precision(system.residual_dimension(), 1.0 / 0.75);

        let build_problem =
            |preference, collapsed_precision: Option<SparseTripletMatrix>, pde_variance| {
                LinearPdeUqProblem {
                    state_prior: GaussianPriorSpec {
                        mean: vec![0.0; system.state_dimension()],
                        precision: to_core_triplets(&precision),
                    },
                    system: system.clone(),
                    uncertain_inputs: vec![LinearUncertainInputSpec {
                        name: "forcing".to_string(),
                        operator: system.forcing_operator.clone(),
                        prior: GaussianPriorSpec {
                            mean: forcing_mean.clone(),
                            precision: forcing_precision.clone(),
                        },
                        preference,
                        collapsed_precision,
                    }],
                    physical_measurements: Vec::new(),
                    derived_quantities: Vec::new(),
                    pde_variance,
                }
            };

        let collapsed = solve_linear_pde_uq(&build_problem(
            RepresentationPreference::Auto,
            Some(collapsed_precision),
            None,
        ))
        .expect("collapsed forcing solve should succeed");
        let latent = solve_linear_pde_uq(&build_problem(
            RepresentationPreference::ForceLatent,
            None,
            Some(residual_variance),
        ))
        .expect("latent forcing solve should succeed");

        assert!(
            max_abs_difference(&collapsed.posterior_mean, &latent.posterior_mean) <= 1e-8,
            "collapsed and latent forcing means should agree"
        );
        assert!(
            max_abs_difference(&collapsed.posterior_variance, &latent.posterior_variance) <= 1e-8,
            "collapsed and latent forcing variances should agree"
        );
    }

    #[test]
    fn soft_dirichlet_boundary_approaches_hard_boundary_for_0form() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let boundary_dofs =
            assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, |point| {
                point[1] == 1.0
            });

        let hard_boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
            "hard",
            boundary_dofs.clone(),
            vec![1.0; boundary_dofs.len()],
            BoundaryTreatment::HardEssential,
        ));
        let soft_boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
            "soft",
            boundary_dofs.clone(),
            vec![1.0; boundary_dofs.len()],
            BoundaryTreatment::SoftEssential { variance: 1e-8 },
        ));

        let hard_system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &hard_boundary)
                .expect("hard system should assemble");
        let soft_system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &soft_boundary)
                .expect("soft system should assemble");

        let hard_precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&hard_system.operator),
                mass: core_triplet_to_feec_csr(&hard_system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let soft_precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&soft_system.operator),
                mass: core_triplet_to_feec_csr(&soft_system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        let hard = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; hard_system.state_dimension()],
                precision: to_core_triplets(&hard_precision),
            },
            system: hard_system,
            uncertain_inputs: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
        })
        .expect("hard solve should succeed");
        let soft = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; soft_system.state_dimension()],
                precision: to_core_triplets(&soft_precision),
            },
            system: soft_system,
            uncertain_inputs: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
        })
        .expect("soft solve should succeed");

        for dof in boundary_dofs {
            assert!((hard.posterior_mean[dof] - 1.0).abs() < 1e-12);
            assert!((soft.posterior_mean[dof] - 1.0).abs() < 5e-3);
        }
    }

    #[test]
    fn uncertain_neumann_and_measurements_reduce_solution_variance_locally() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 3, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let precision = to_core_triplets(&precision);

        let measurement_vertex = 0usize;
        let measurement = LinearGaussianMeasurementSpec {
            name: "sensor".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                system.layout.full_dimension,
                [feg_core::SparseTriplet {
                    row: 0,
                    col: measurement_vertex,
                    value: 1.0,
                }],
            ),
            observations: vec![1.0],
            bias: vec![0.0],
            variance: 1e-6,
        };
        let neumann_precision = diagonal_precision(system.residual_dimension(), 1.0);

        let without_measurement = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: precision.clone(),
            },
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "neumann".to_string(),
                operator: system.neumann_operator.clone(),
                prior: GaussianPriorSpec {
                    mean: vec![0.0; system.residual_dimension()],
                    precision: neumann_precision.clone(),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
        })
        .expect("solve without measurement should succeed");
        let with_measurement = {
            let measurement_problem = LinearPdeUqProblem {
                state_prior: GaussianPriorSpec {
                    mean: vec![0.0; system.state_dimension()],
                    precision,
                },
                system: system.clone(),
                uncertain_inputs: vec![LinearUncertainInputSpec {
                    name: "neumann".to_string(),
                    operator: system.neumann_operator.clone(),
                    prior: GaussianPriorSpec {
                        mean: vec![0.0; system.residual_dimension()],
                        precision: neumann_precision,
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                }],
                physical_measurements: vec![measurement],
                derived_quantities: Vec::new(),
                pde_variance: Some(1e-8),
            };
            solve_linear_pde_uq(&measurement_problem)
        }
        .expect("solve with measurement should succeed");

        let min_variance = with_measurement
            .posterior_variance
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_variance = with_measurement
            .posterior_variance
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            with_measurement.posterior_variance[measurement_vertex]
                < without_measurement.posterior_variance[measurement_vertex]
        );
        assert!(
            max_variance - min_variance > 1e-8,
            "uncertain inputs plus physical measurements should produce a non-uniform posterior variance"
        );
    }

    #[test]
    fn identity_derived_quantity_matches_state_marginal_variances() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: to_core_triplets(&precision),
            },
            system: system.clone(),
            uncertain_inputs: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "identity".to_string(),
                operator: SparseRowOperator::identity(system.layout.full_dimension),
            }],
            pde_variance: Some(1e-6),
        })
        .expect("solve with identity derived quantity should succeed");

        let identity = result
            .derived_variances
            .get("identity")
            .expect("identity derived variance should be present");
        assert!(
            max_abs_difference(&identity.prior_variance, &result.prior_variance) <= 1e-10,
            "identity prior variance should match state prior variance"
        );
        assert!(
            max_abs_difference(&identity.posterior_variance, &result.posterior_variance) <= 1e-10,
            "identity posterior variance should match state posterior variance"
        );
    }

    #[test]
    fn joint_posterior_builder_exposes_restricted_identity_mean() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let problem = LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.5; system.state_dimension()],
                precision: to_core_triplets(&precision),
            },
            system,
            uncertain_inputs: Vec::new(),
            physical_measurements: Vec::new(),
            derived_quantities: vec![LinearPdeDerivedQuantitySpec {
                name: "identity".to_string(),
                operator: SparseRowOperator::identity(topology.nsimplices(0)),
            }],
            pde_variance: Some(1e-6),
        };
        let config = LinearPdeUqSolverConfig {
            variance: LinearPdeVarianceConfig::default(),
            stabilize_precision: true,
            log_diagnostics: false,
        };
        let result = solve_linear_pde_uq_with_config(&problem, &config)
            .expect("solve with identity derived quantity should succeed");
        let posterior = build_linear_pde_joint_posterior_with_config(&problem, &config)
            .expect("joint posterior build should succeed");
        let identity = posterior
            .derived_quantities
            .get("identity")
            .expect("identity derived operator should be present");
        let applied = identity
            .apply(posterior.posterior.mean_vector())
            .expect("identity derived operator should apply");
        let expected_centered_mean = &result.reduced_posterior_mean
            - &FeecVector::from_vec(problem.state_prior.mean.clone());

        assert_eq!(posterior.state_dimension, problem.system.state_dimension());
        assert_eq!(posterior.joint_dimension, posterior.posterior.dimension());
        assert_eq!(applied.len(), expected_centered_mean.len());
        assert!(
            max_abs_difference(
                &gmrf_vec_to_feec(&applied),
                &expected_centered_mean,
            ) <= 1e-10,
            "joint posterior builder should expose the same centered reduced posterior mean as the solver"
        );
    }

    #[test]
    fn mixed_1form_system_with_latent_loads_produces_finite_solution_outputs() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let state_dofs = topology
            .boundary_subcomplex_simplices(1)
            .into_iter()
            .take(1)
            .map(|simp| simp.kidx)
            .collect::<Vec<_>>();
        let auxiliary_dofs = topology
            .boundary_subcomplex_simplices(0)
            .into_iter()
            .take(1)
            .map(|simp| simp.kidx)
            .collect::<Vec<_>>();
        let boundary = BoundarySpec::default()
            .with_state_region(BoundaryRegionSpec::new(
                "soft-state",
                state_dofs.clone(),
                vec![0.0],
                BoundaryTreatment::SoftEssential { variance: 1e-6 },
            ))
            .with_auxiliary_region(BoundaryRegionSpec::new(
                "hard-aux",
                auxiliary_dofs,
                vec![0.0],
                BoundaryTreatment::HardEssential,
            ));
        let system = build_reduced_hodge_laplace_1form_system(&topology, &geometry, &boundary)
            .expect("mixed 1-form system should assemble");
        let result = solve_linear_pde_uq(&LinearPdeUqProblem {
            state_prior: GaussianPriorSpec {
                mean: vec![0.0; system.state_dimension()],
                precision: build_reduced_1form_whittle_prior(&system),
            },
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "forcing".to_string(),
                operator: system.forcing_operator.clone(),
                prior: GaussianPriorSpec {
                    mean: vec![0.0; system.residual_dimension()],
                    precision: diagonal_precision(system.residual_dimension(), 1.0),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: Some(1e-8),
        })
        .expect("mixed 1-form solve should succeed");

        assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
        assert!(result
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn rbmc_clipped_variances_stay_nonnegative_and_below_prior() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 3, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let measurement = LinearGaussianMeasurementSpec {
            name: "sensor".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                system.layout.full_dimension,
                [feg_core::SparseTriplet {
                    row: 0,
                    col: 0,
                    value: 1.0,
                }],
            ),
            observations: vec![1.0],
            bias: vec![0.0],
            variance: 1e-6,
        };
        let result = solve_linear_pde_uq_with_config(
            &LinearPdeUqProblem {
                state_prior: GaussianPriorSpec {
                    mean: vec![0.0; system.state_dimension()],
                    precision: to_core_triplets(&precision),
                },
                system,
                uncertain_inputs: Vec::new(),
                physical_measurements: vec![measurement],
                derived_quantities: Vec::new(),
                pde_variance: Some(1e-6),
            },
            &LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::RbmcClipped,
                    num_rbmc_probes: 24,
                    rbmc_batch_count: 4,
                    rng_seed: 11,
                },
                stabilize_precision: true,
                log_diagnostics: false,
            },
        )
        .expect("rbmc-clipped solve should succeed");

        assert!(result.prior_variance.iter().all(|value| *value >= 0.0));
        assert!(result.posterior_variance.iter().all(|value| *value >= 0.0));
        for (posterior, prior) in result
            .posterior_variance
            .iter()
            .zip(result.prior_variance.iter())
        {
            assert!(
                *posterior <= *prior + 1e-10,
                "clipped posterior variance should not exceed prior variance"
            );
        }
    }

    #[test]
    fn debug_reports_factorization_fill_stats() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let system =
            build_reduced_laplace_beltrami_system(&topology, &geometry, &BoundarySpec::default())
                .expect("0-form system should assemble");
        let precision = build_matern_precision_0form(
            &crate::matern_0form::LaplaceBeltrami0Form {
                laplacian: core_triplet_to_feec_csr(&system.operator),
                mass: core_triplet_to_feec_csr(&system.state_mass),
            },
            MaternConfig {
                kappa: 1.0,
                tau: 1.0,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let result = solve_linear_pde_uq_with_config(
            &LinearPdeUqProblem {
                state_prior: GaussianPriorSpec {
                    mean: vec![0.0; system.state_dimension()],
                    precision: to_core_triplets(&precision),
                },
                system,
                uncertain_inputs: Vec::new(),
                physical_measurements: Vec::new(),
                derived_quantities: Vec::new(),
                pde_variance: Some(1e-6),
            },
            &LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig::default(),
                stabilize_precision: true,
                log_diagnostics: false,
            },
        )
        .expect("diagnostic solve should succeed");

        let prior = result.debug.prior_factorization;
        let posterior = result.debug.posterior_factorization;
        assert_eq!(result.debug.joint_dimension, prior.dimension);
        assert_eq!(prior.dimension, posterior.dimension);
        assert!(prior.matrix_nnz >= prior.matrix_lower_triangle_nnz);
        assert!(posterior.matrix_nnz >= posterior.matrix_lower_triangle_nnz);
        assert!(prior.factor_nnz > 0);
        assert!(posterior.factor_nnz > 0);
        assert!(prior.fill_in_ratio_vs_lower_triangle > 0.0);
        assert!(posterior.fill_in_ratio_vs_lower_triangle > 0.0);
        assert!(prior.factor_numeric_values_mib > 0.0);
        assert!(posterior.factor_numeric_values_mib > 0.0);
    }

    #[test]
    fn auto_only_selects_collapsed_when_sparse_precision_is_available() {
        let input = LinearUncertainInputSpec {
            name: "nonlocal".to_string(),
            operator: SparseTripletMatrix::from_triplets(
                1,
                2,
                [feg_core::SparseTriplet {
                    row: 0,
                    col: 1,
                    value: 2.0,
                }],
            ),
            prior: GaussianPriorSpec {
                mean: vec![0.0, 0.0],
                precision: SparseTripletMatrix::new(2, 2),
            },
            preference: RepresentationPreference::Auto,
            collapsed_precision: None,
        };
        let resolved = resolve_input_representations(&[input], 1).expect("resolution should work");
        assert_eq!(
            resolved[0].1,
            ResolvedInputRepresentation::Latent,
            "Auto should keep inputs latent when no sparse collapsed precision is available"
        );
    }
}
