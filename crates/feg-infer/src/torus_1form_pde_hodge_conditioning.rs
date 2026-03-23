use crate::hodge_1form_conditioning::{
    build_coexact_1form_transform, build_exact_1form_transform,
    build_harmonic_restricted_precision, compute_posterior_bias_diagnostics, Hodge1FormBranchKind,
    Hodge1FormPosteriorBiasDiagnostics,
};
use crate::matern_0form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, MaternConfig as Matern0FormConfig,
    MaternMassInverse as Matern0FormMassInverse,
};
use crate::matern_1form::{
    build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig as Matern1FormConfig, MaternMassInverse as Matern1FormMassInverse,
};
use crate::matern_2form::{
    build_hodge_laplacian_2form, build_matern_precision_2form, MaternConfig as Matern2FormConfig,
    MaternMassInverse as Matern2FormMassInverse,
};
use crate::sparse::{dense_to_feec_csr, feec_csr_to_dense, gmrf_vec_to_feec};
use crate::torus_1form_conditioning::SurfaceVectorVarianceMode;
use crate::torus_1form_pde_conditioning::{
    assemble_torus_1form_pde_conditioning_result, build_rbmc_workspace,
    clip_rbmc_posterior_to_prior, default_torus_shell_resolution_1_mesh_path,
    estimate_transformed_rbmc_variances, exact_transformed_variances, invalid_data,
    prepare_torus_1form_pde_problem, run_prepared_torus_1form_pde_conditioning,
    split_ambient_estimates, split_component_estimates, validate_config,
    write_torus_1form_pde_conditioning_outputs, PreparedTorus1FormPdeProblem,
    SparseRowLinearOperator, Torus1FormAmbientVarianceEstimates, Torus1FormPdeConditioningConfig,
    Torus1FormPdeConditioningResult,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::ManifoldComplexExt;
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::DenseMatrix as GmrfDenseMatrix;
use gmrf_core::Gmrf;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Torus1FormPdeHodgeConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
}

impl Default for Torus1FormPdeHodgeConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_resolution_1_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            surface_vector_variance_mode: SurfaceVectorVarianceMode::Exact,
            num_rbmc_probes: 256,
            rbmc_batch_count: 8,
            rng_seed: 13,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeHodgeBranchResult {
    pub kind: Hodge1FormBranchKind,
    pub latent_dimension: usize,
    pub observation_count: usize,
    pub posterior_bias_diagnostics: Hodge1FormPosteriorBiasDiagnostics,
    pub conditioning: Torus1FormPdeConditioningResult,
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeHodgeConditioningResult {
    pub full: Torus1FormPdeConditioningResult,
    pub exact: Torus1FormPdeHodgeBranchResult,
    pub coexact: Torus1FormPdeHodgeBranchResult,
    pub harmonic: Torus1FormPdeHodgeBranchResult,
}

pub fn run_torus_1form_pde_hodge_conditioning(
    config: &Torus1FormPdeHodgeConditioningConfig,
) -> Result<Torus1FormPdeHodgeConditioningResult, Box<dyn Error>> {
    let base_config = config.to_base_config();
    validate_config(&base_config)?;

    let prepared = prepare_torus_1form_pde_problem(&base_config)?;
    let full = run_prepared_torus_1form_pde_conditioning(&prepared, &base_config)?;
    let harmonic_free_projection = build_harmonic_free_projection_operator(&prepared)?;

    let d0 = FeecCsr::from(&prepared.topology.exterior_derivative_operator(0));
    let d1 = FeecCsr::from(&prepared.topology.exterior_derivative_operator(1));

    let q1 = build_matern_precision_1form(
        &prepared.topology,
        &prepared.metric,
        &prepared.hodge,
        Matern1FormConfig {
            kappa: base_config.kappa,
            tau: base_config.tau,
            mass_inverse: Matern1FormMassInverse::Nc1ProjectedSparseInverse,
        },
    );

    let exact_transform = build_exact_1form_transform(&prepared.topology);
    let exact = run_sparse_branch(
        Hodge1FormBranchKind::Exact,
        &exact_transform,
        &build_matern_precision_0form(
            &build_laplace_beltrami_0form(&prepared.topology, &prepared.metric),
            Matern0FormConfig {
                kappa: base_config.kappa,
                tau: base_config.tau,
                mass_inverse: Matern0FormMassInverse::RowSumLumped,
            },
        ),
        &prepared,
        &base_config,
        &harmonic_free_projection,
        &d0,
        &d1,
    )?;

    let coexact_transform =
        build_coexact_1form_transform(&prepared.topology, &prepared.metric, &prepared.hodge.mass_u);
    let q2 = build_matern_precision_2form(
        &prepared.topology,
        &prepared.metric,
        &build_hodge_laplacian_2form(&prepared.topology, &prepared.metric)?,
        Matern2FormConfig {
            kappa: base_config.kappa,
            tau: base_config.tau,
            mass_inverse: Matern2FormMassInverse::ExactTopDegreeDiagonalOrProjectedNc2,
        },
    )?;
    let coexact = run_sparse_branch(
        Hodge1FormBranchKind::Coexact,
        &coexact_transform,
        &q2,
        &prepared,
        &base_config,
        &harmonic_free_projection,
        &d0,
        &d1,
    )?;

    let harmonic_precision =
        build_harmonic_restricted_precision(&q1, &prepared.harmonic_basis_orthonormal);
    let harmonic = run_dense_branch(
        Hodge1FormBranchKind::Harmonic,
        &prepared.harmonic_basis_orthonormal,
        &harmonic_precision,
        &prepared,
        &base_config,
        &harmonic_free_projection,
        &d0,
        &d1,
    )?;

    Ok(Torus1FormPdeHodgeConditioningResult {
        full,
        exact,
        coexact,
        harmonic,
    })
}

pub fn write_torus_1form_pde_hodge_conditioning_outputs(
    result: &Torus1FormPdeHodgeConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    write_torus_1form_pde_conditioning_outputs(&result.full, out_dir.join("full"))?;
    write_branch_outputs(&result.exact, out_dir)?;
    write_branch_outputs(&result.coexact, out_dir)?;
    write_branch_outputs(&result.harmonic, out_dir)?;
    write_comparison_summary(result, out_dir)?;

    Ok(())
}

impl Torus1FormPdeHodgeConditioningConfig {
    fn to_base_config(&self) -> Torus1FormPdeConditioningConfig {
        Torus1FormPdeConditioningConfig {
            mesh_path: self.mesh_path.clone(),
            kappa: self.kappa,
            tau: self.tau,
            noise_variance: self.noise_variance,
            surface_vector_variance_mode: self.surface_vector_variance_mode,
            num_rbmc_probes: self.num_rbmc_probes,
            rbmc_batch_count: self.rbmc_batch_count,
            rng_seed: self.rng_seed,
        }
    }
}

fn run_sparse_branch(
    kind: Hodge1FormBranchKind,
    ambient_transform: &FeecCsr,
    latent_precision: &FeecCsr,
    prepared: &PreparedTorus1FormPdeProblem,
    config: &Torus1FormPdeConditioningConfig,
    harmonic_free_projection: &SparseRowLinearOperator,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Result<Torus1FormPdeHodgeBranchResult, Box<dyn Error>> {
    let ambient_operator =
        SparseRowLinearOperator::from_sparse_matrix(ambient_transform).map_err(invalid_data)?;
    let latent_observation_matrix = &prepared.system_matrix * ambient_transform;
    run_branch(
        kind,
        &ambient_operator,
        &latent_observation_matrix,
        latent_precision,
        prepared,
        config,
        harmonic_free_projection,
        d0,
        d1,
    )
}

fn run_dense_branch(
    kind: Hodge1FormBranchKind,
    ambient_transform: &FeecMatrix,
    latent_precision: &FeecCsr,
    prepared: &PreparedTorus1FormPdeProblem,
    config: &Torus1FormPdeConditioningConfig,
    harmonic_free_projection: &SparseRowLinearOperator,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Result<Torus1FormPdeHodgeBranchResult, Box<dyn Error>> {
    let ambient_operator =
        SparseRowLinearOperator::from_dense_matrix(ambient_transform, 0.0).map_err(invalid_data)?;
    let latent_observation_dense = feec_csr_to_dense(&prepared.system_matrix) * ambient_transform;
    let latent_observation_matrix = dense_to_feec_csr(&latent_observation_dense, 0.0);
    run_branch(
        kind,
        &ambient_operator,
        &latent_observation_matrix,
        latent_precision,
        prepared,
        config,
        harmonic_free_projection,
        d0,
        d1,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_branch(
    kind: Hodge1FormBranchKind,
    ambient_operator: &SparseRowLinearOperator,
    latent_observation_matrix: &FeecCsr,
    latent_precision: &FeecCsr,
    prepared: &PreparedTorus1FormPdeProblem,
    config: &Torus1FormPdeConditioningConfig,
    harmonic_free_projection: &SparseRowLinearOperator,
    d0: &FeecCsr,
    d1: &FeecCsr,
) -> Result<Torus1FormPdeHodgeBranchResult, Box<dyn Error>> {
    let latent_dimension = latent_precision.nrows();
    let empty_constraints = GmrfDenseMatrix::zeros(0, latent_dimension);
    let cell_count = prepared.cell_geometry.theta.len();

    let harmonic_free_operator =
        SparseRowLinearOperator::compose(harmonic_free_projection, ambient_operator)
            .map_err(invalid_data)?;
    let reconstructed_operator = SparseRowLinearOperator::compose(
        &prepared.reconstructed_stacked_operator,
        ambient_operator,
    )
    .map_err(invalid_data)?;
    let surface_vector_operator = SparseRowLinearOperator::compose(
        &prepared.surface_vector_stacked_operator,
        ambient_operator,
    )
    .map_err(invalid_data)?;
    let smoothed_operator =
        SparseRowLinearOperator::compose(&prepared.smoothed_stacked_operator, ambient_operator)
            .map_err(invalid_data)?;
    let circulation_operator =
        SparseRowLinearOperator::compose(&prepared.circulation_operator, ambient_operator)
            .map_err(invalid_data)?;

    let prior_precision = feec_csr_to_gmrf(latent_precision);
    let observations = feec_vec_to_gmrf(&prepared.rhs);
    let mut prior_workspace = build_rbmc_workspace(&prior_precision, &empty_constraints)?;
    let prior_variance = exact_transformed_variances(&mut prior_workspace, ambient_operator)?;
    let prior_harmonic_free_variance =
        exact_transformed_variances(&mut prior_workspace, &harmonic_free_operator)?;
    let reconstructed_prior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut prior_workspace,
            &reconstructed_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?,
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_prior = surface_vector_estimates(
        &mut prior_workspace,
        &surface_vector_operator,
        config,
        cell_count,
        None,
    )?;
    let smoothed_prior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut prior_workspace,
            &smoothed_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x2000),
        )?,
        cell_count,
    )
    .map_err(invalid_data)?;
    let circulation_prior = estimate_transformed_rbmc_variances(
        &mut prior_workspace,
        &circulation_operator,
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?;

    let (posterior_precision, information) = apply_gaussian_observations(
        &prior_precision,
        &feec_csr_to_gmrf(latent_observation_matrix),
        &observations,
        None,
        config.noise_variance,
    );
    let posterior = Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let latent_posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let posterior_mean = ambient_operator
        .apply_feec(&latent_posterior_mean)
        .map_err(invalid_data)?;

    let mut posterior_workspace = build_rbmc_workspace(&posterior_precision, &empty_constraints)?;
    let posterior_variance =
        exact_transformed_variances(&mut posterior_workspace, ambient_operator)?;
    let posterior_harmonic_free_variance =
        exact_transformed_variances(&mut posterior_workspace, &harmonic_free_operator)?;
    let reconstructed_posterior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut posterior_workspace,
            &reconstructed_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?,
        cell_count,
    )
    .map_err(invalid_data)?;
    let surface_vector_posterior = surface_vector_estimates(
        &mut posterior_workspace,
        &surface_vector_operator,
        config,
        cell_count,
        Some(&surface_vector_prior),
    )?;
    let smoothed_posterior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut posterior_workspace,
            &smoothed_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x2000),
        )?,
        cell_count,
    )
    .map_err(invalid_data)?;
    let circulation_posterior = estimate_transformed_rbmc_variances(
        &mut posterior_workspace,
        &circulation_operator,
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?;

    let conditioning = assemble_torus_1form_pde_conditioning_result(
        prepared,
        config,
        posterior_mean,
        &prior_variance.unconstrained,
        &posterior_variance.unconstrained,
        &prior_harmonic_free_variance.unconstrained,
        &posterior_harmonic_free_variance.unconstrained,
        &reconstructed_prior,
        &reconstructed_posterior,
        &surface_vector_prior,
        &surface_vector_posterior,
        &smoothed_prior,
        &smoothed_posterior,
        &circulation_prior,
        &circulation_posterior,
    )?;
    let posterior_bias_diagnostics = compute_posterior_bias_diagnostics(
        &conditioning.posterior_mean,
        &prepared.hodge.mass_u,
        d0,
        d1,
    );

    Ok(Torus1FormPdeHodgeBranchResult {
        kind,
        latent_dimension,
        observation_count: prepared.rhs.len(),
        posterior_bias_diagnostics,
        conditioning,
    })
}

fn surface_vector_estimates(
    workspace: &mut crate::torus_1form_pde_conditioning::RbmcWorkspace,
    operator: &SparseRowLinearOperator,
    config: &Torus1FormPdeConditioningConfig,
    cell_count: usize,
    prior: Option<&Torus1FormAmbientVarianceEstimates>,
) -> Result<Torus1FormAmbientVarianceEstimates, Box<dyn Error>> {
    let estimates = match config.surface_vector_variance_mode {
        SurfaceVectorVarianceMode::Exact => exact_transformed_variances(workspace, operator)?,
        SurfaceVectorVarianceMode::Rbmc | SurfaceVectorVarianceMode::RbmcClipped => {
            estimate_transformed_rbmc_variances(
                workspace,
                operator,
                config.num_rbmc_probes,
                config.rbmc_batch_count,
                config.rng_seed.wrapping_add(0x1800),
            )?
        }
    };
    let ambient = split_ambient_estimates(estimates, cell_count).map_err(invalid_data)?;
    Ok(
        if config.surface_vector_variance_mode == SurfaceVectorVarianceMode::RbmcClipped {
            if let Some(prior) = prior {
                clip_rbmc_posterior_to_prior(prior, &ambient)
            } else {
                ambient
            }
        } else {
            ambient
        },
    )
}

fn build_harmonic_free_projection_operator(
    prepared: &PreparedTorus1FormPdeProblem,
) -> Result<SparseRowLinearOperator, Box<dyn Error>> {
    let dim = prepared.hodge.mass_u.nrows();
    if prepared.harmonic_basis_orthonormal.ncols() == 0 {
        return SparseRowLinearOperator::from_dense_matrix(&FeecMatrix::identity(dim, dim), 0.0)
            .map_err(|err| invalid_data(err).into());
    }

    let mass_dense = feec_csr_to_dense(&prepared.hodge.mass_u);
    let projector = FeecMatrix::identity(dim, dim)
        - &prepared.harmonic_basis_orthonormal
            * (prepared.harmonic_basis_orthonormal.transpose() * mass_dense);
    SparseRowLinearOperator::from_dense_matrix(&projector, 0.0)
        .map_err(|err| invalid_data(err).into())
}

fn write_branch_outputs(
    branch: &Torus1FormPdeHodgeBranchResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let branch_dir = out_dir.join(branch.kind.as_str());
    write_torus_1form_pde_conditioning_outputs(&branch.conditioning, &branch_dir)?;

    let summary_path = branch_dir.join("summary.txt");
    let mut writer = OpenOptions::new().append(true).open(summary_path)?;
    writeln!(writer, "branch={}", branch.kind.as_str())?;
    writeln!(writer, "latent_dimension={}", branch.latent_dimension)?;
    writeln!(writer, "observation_count={}", branch.observation_count)?;
    writeln!(
        writer,
        "curl_residual_norm_posterior_mean={}",
        branch.posterior_bias_diagnostics.curl_residual_norm
    )?;
    writeln!(
        writer,
        "curl_residual_relative_posterior_mean={}",
        branch.posterior_bias_diagnostics.curl_residual_relative
    )?;
    writeln!(
        writer,
        "coclosed_residual_norm_posterior_mean={}",
        branch.posterior_bias_diagnostics.coclosed_residual_norm
    )?;
    writeln!(
        writer,
        "coclosed_residual_relative_posterior_mean={}",
        branch.posterior_bias_diagnostics.coclosed_residual_relative
    )?;

    Ok(())
}

fn write_comparison_summary(
    result: &Torus1FormPdeHodgeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("comparison_summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Torus 1-form Matérn PDE Hodge-split conditioning")?;
    write_result_summary(&mut writer, "full", None, None, &result.full)?;
    write_result_summary(
        &mut writer,
        "exact",
        Some(result.exact.latent_dimension),
        Some(&result.exact.posterior_bias_diagnostics),
        &result.exact.conditioning,
    )?;
    write_result_summary(
        &mut writer,
        "coexact",
        Some(result.coexact.latent_dimension),
        Some(&result.coexact.posterior_bias_diagnostics),
        &result.coexact.conditioning,
    )?;
    write_result_summary(
        &mut writer,
        "harmonic",
        Some(result.harmonic.latent_dimension),
        Some(&result.harmonic.posterior_bias_diagnostics),
        &result.harmonic.conditioning,
    )?;
    Ok(())
}

fn write_result_summary(
    writer: &mut impl Write,
    label: &str,
    latent_dimension: Option<usize>,
    diagnostics: Option<&Hodge1FormPosteriorBiasDiagnostics>,
    result: &Torus1FormPdeConditioningResult,
) -> io::Result<()> {
    writeln!(writer, "[{label}]")?;
    if let Some(latent_dimension) = latent_dimension {
        writeln!(writer, "latent_dimension={latent_dimension}")?;
    }
    writeln!(
        writer,
        "posterior_relative_residual_norm={}",
        result.posterior_relative_residual_norm
    )?;
    writeln!(writer, "l2_error={}", result.l2_error)?;
    writeln!(writer, "hd_error={}", result.hd_error)?;
    writeln!(
        writer,
        "edge_variance_ratio_mean={}",
        mean(&result.variance_ratio)
    )?;
    writeln!(
        writer,
        "surface_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.surface_vector.trace.ratio)
    )?;
    if let Some(diagnostics) = diagnostics {
        writeln!(
            writer,
            "curl_residual_relative_posterior_mean={}",
            diagnostics.curl_residual_relative
        )?;
        writeln!(
            writer,
            "coclosed_residual_relative_posterior_mean={}",
            diagnostics.coclosed_residual_relative
        )?;
    }
    Ok(())
}

fn mean(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
