use crate::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    infer_torus_radii,
};
use crate::matern_1form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, build_matern_system_matrix_1form,
    feec_csr_to_gmrf, feec_vec_to_gmrf, MaternConfig, MaternMassInverse,
};
use crate::torus_1form_conditioning::{
    SurfaceVectorVarianceMode, Torus1FormAmbientVarianceFields, Torus1FormVarianceComponentFields,
    Torus1FormVarianceFieldSet,
};
use crate::util::convert_whittle_params_to_matern;
use crate::vtk::{
    write_1cochain_vtk_fields, write_1form_vector_proxy_vtk_fields,
    write_top_cell_scalar_vtk_fields,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::{field::EmbeddedDiffFormClosure, field::ExteriorField, ExteriorElement};
use faer::linalg::solvers::Solve;
use faer::Side;
use formoniq::fe::fe_l2_error;
use formoniq::io::{sample_1form_cell_vectors, write_top_cell_vtk_fields};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::{
    DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{Gmrf, GmrfError};
use manifold::{
    geometry::coord::{
        mesh::MeshCoords,
        simplex::{barycenter_local, SimplexHandleExt},
        CoordRef,
    },
    topology::complex::Complex,
};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;
const EXACT_VARIANCE_TOLERANCE: f64 = 1e-10;
const DEFAULT_NUM_RBMC_PROBES: usize = 256;
const DEFAULT_RBMC_BATCH_COUNT: usize = 8;
const DEFAULT_RNG_SEED: u64 = 13;
const DEFAULT_MAJOR_RADIUS: f64 = 1.0;
const DEFAULT_MINOR_RADIUS: f64 = 0.3;
const SMOOTHING_BANDWIDTH_SCALE: f64 = 0.5;
const SMOOTHING_CUTOFF_SCALE: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
}

impl Default for Torus1FormPdeConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_resolution_1_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            surface_vector_variance_mode: SurfaceVectorVarianceMode::Exact,
            num_rbmc_probes: DEFAULT_NUM_RBMC_PROBES,
            rbmc_batch_count: DEFAULT_RBMC_BATCH_COUNT,
            rng_seed: DEFAULT_RNG_SEED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeVarianceFields {
    pub reconstructed: Torus1FormVarianceComponentFields,
    pub surface_vector: Torus1FormAmbientVarianceFields,
    pub smoothed: Torus1FormVarianceComponentFields,
    pub circulation: Torus1FormVarianceFieldSet,
}

#[derive(Debug, Clone)]
pub struct Torus1FormPdeConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub edge_theta: FeecVector,
    pub edge_phi: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
    pub effective_range: f64,
    pub truth: FeecVector,
    pub rhs: FeecVector,
    pub posterior_mean: FeecVector,
    pub posterior_rhs: FeecVector,
    pub pde_residual: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub variance_ratio: FeecVector,
    pub harmonic_free_truth: FeecVector,
    pub harmonic_free_posterior_mean: FeecVector,
    pub harmonic_free_absolute_mean_error: FeecVector,
    pub harmonic_free_prior_variance: FeecVector,
    pub harmonic_free_posterior_variance: FeecVector,
    pub harmonic_free_variance_reduction: FeecVector,
    pub harmonic_free_variance_ratio: FeecVector,
    pub harmonic_coefficients_truth: [f64; 2],
    pub harmonic_coefficients_posterior_mean: [f64; 2],
    pub l2_error: f64,
    pub hd_error: f64,
    pub truth_residual_norm: f64,
    pub truth_relative_residual_norm: f64,
    pub posterior_residual_norm: f64,
    pub posterior_relative_residual_norm: f64,
    pub variance_fields: Torus1FormPdeVarianceFields,
}

#[derive(Clone)]
struct SparseRowLinearOperator {
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
}

struct TorusEdgeGeometry {
    major_radius: f64,
    minor_radius: f64,
    theta: Vec<f64>,
    phi: Vec<f64>,
    toroidal_alignment_sq: Vec<f64>,
}

struct TorusCellGeometry {
    major_radius: f64,
    minor_radius: f64,
    theta: Vec<f64>,
    phi: Vec<f64>,
}

struct RbmcVarianceEstimates {
    unconstrained: GmrfVector,
    harmonic_free: GmrfVector,
}

struct ConstraintVarianceCorrection {
    covariance_times_constraint_t: GmrfDenseMatrix,
    schur_inverse: GmrfDenseMatrix,
}

struct RbmcWorkspace {
    gmrf: Gmrf,
    constraint_correction: Option<ConstraintVarianceCorrection>,
}

struct Torus1FormVarianceComponentEstimates {
    toroidal: RbmcVarianceEstimates,
    poloidal: RbmcVarianceEstimates,
}

struct Torus1FormAmbientVarianceEstimates {
    x: RbmcVarianceEstimates,
    y: RbmcVarianceEstimates,
    z: RbmcVarianceEstimates,
}

pub fn default_torus_shell_resolution_1_mesh_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_resolution_1.msh")
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

    fn apply(&self, input: &GmrfVector) -> GmrfVector {
        GmrfVector::from_iterator(
            self.nrows(),
            self.rows.iter().map(|row| {
                row.iter()
                    .map(|(col, value)| *value * input[*col])
                    .sum::<f64>()
            }),
        )
    }

    fn apply_transpose(&self, input: &GmrfVector) -> GmrfVector {
        let mut out = GmrfVector::zeros(self.ncols);
        for (row_index, row) in self.rows.iter().enumerate() {
            let weight = input[row_index];
            if weight == 0.0 {
                continue;
            }
            for (col, value) in row {
                out[*col] += weight * *value;
            }
        }
        out
    }

    fn stack(operators: &[&SparseRowLinearOperator]) -> Result<Self, String> {
        let Some(first) = operators.first() else {
            return Err("at least one operator is required for stacking".to_string());
        };
        let ncols = first.ncols;
        if operators.iter().any(|operator| operator.ncols != ncols) {
            return Err("all stacked operators must have the same column count".to_string());
        }

        let mut rows = Vec::new();
        for operator in operators {
            rows.extend(operator.rows.iter().cloned());
        }
        Ok(Self { ncols, rows })
    }

    fn compose(
        left: &SparseRowLinearOperator,
        right: &SparseRowLinearOperator,
    ) -> Result<Self, String> {
        if left.ncols != right.nrows() {
            return Err("operator dimensions are incompatible for composition".to_string());
        }

        let mut rows = Vec::with_capacity(left.nrows());
        for left_row in &left.rows {
            let mut combined = BTreeMap::<usize, f64>::new();
            for (intermediate, weight) in left_row {
                for (col, value) in &right.rows[*intermediate] {
                    *combined.entry(*col).or_insert(0.0) += *weight * *value;
                }
            }
            let row = combined
                .into_iter()
                .filter_map(|(col, value)| (value.abs() > EPS).then_some((col, value)))
                .collect::<Vec<_>>();
            rows.push(row);
        }

        Ok(Self {
            ncols: right.ncols,
            rows,
        })
    }
}

pub fn run_torus_1form_pde_conditioning(
    config: &Torus1FormPdeConditioningConfig,
) -> Result<Torus1FormPdeConditioningResult, Box<dyn Error>> {
    validate_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let geometry = build_torus_edge_geometry(&topology, &coords)?;
    let cell_geometry = build_torus_cell_geometry(
        &topology,
        &coords,
        geometry.major_radius,
        geometry.minor_radius,
    )
    .map_err(invalid_data)?;
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let system_matrix = build_matern_system_matrix_1form(&hodge, config.kappa);

    let harmonic_basis =
        build_analytic_torus_harmonic_basis(&topology, &coords, &metric).map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis(&harmonic_basis, &hodge.mass_u).map_err(invalid_data)?;
    let harmonic_constraints =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .map_err(invalid_data)?;

    let (u_exact, dif_solution_exact) = build_torus_convergence_fields();
    let truth_cochain = cochain_projection(&u_exact, &topology, &coords, None);
    let truth = truth_cochain.coeffs.clone();
    let rhs = &system_matrix * &truth;

    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let observation_matrix = feec_csr_to_gmrf(&system_matrix);
    let observations = feec_vec_to_gmrf(&rhs);
    let (posterior_precision, information) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        config.noise_variance,
    );
    let posterior =
        Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let absolute_mean_error = absolute_difference(&posterior_mean, &truth);

    let mut prior_workspace = build_rbmc_workspace(&q_prior, &harmonic_constraints)?;
    let prior_latent_variances =
        exact_latent_variances(&mut prior_workspace, &harmonic_constraints)?;
    let mut posterior_workspace =
        build_rbmc_workspace(&posterior_precision, &harmonic_constraints)?;
    let posterior_latent_variances =
        exact_latent_variances(&mut posterior_workspace, &harmonic_constraints)?;

    let smoothing_bandwidth = SMOOTHING_BANDWIDTH_SCALE
        * convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2).2;
    let smoothing_cutoff = SMOOTHING_CUTOFF_SCALE
        * convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2).2;
    let (_nu, _variance, effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    let toroidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, true)
            .map_err(invalid_data)?;
    let poloidal_operator =
        build_reconstructed_component_operator(&topology, &coords, &cell_geometry, false)
            .map_err(invalid_data)?;
    let surface_x_operator =
        build_embedded_component_operator(&topology, &coords, 0).map_err(invalid_data)?;
    let surface_y_operator =
        build_embedded_component_operator(&topology, &coords, 1).map_err(invalid_data)?;
    let surface_z_operator =
        build_embedded_component_operator(&topology, &coords, 2).map_err(invalid_data)?;
    let reconstructed_stacked_operator =
        SparseRowLinearOperator::stack(&[&toroidal_operator, &poloidal_operator])
            .map_err(invalid_data)?;
    let surface_vector_stacked_operator =
        SparseRowLinearOperator::stack(&[&surface_x_operator, &surface_y_operator, &surface_z_operator])
            .map_err(invalid_data)?;
    let smoothing_operator =
        build_gaussian_smoothing_operator(&cell_geometry, smoothing_bandwidth, smoothing_cutoff)
            .map_err(invalid_data)?;
    let smoothed_toroidal_operator =
        SparseRowLinearOperator::compose(&smoothing_operator, &toroidal_operator)
            .map_err(invalid_data)?;
    let smoothed_poloidal_operator =
        SparseRowLinearOperator::compose(&smoothing_operator, &poloidal_operator)
            .map_err(invalid_data)?;
    let smoothed_stacked_operator =
        SparseRowLinearOperator::stack(&[&smoothed_toroidal_operator, &smoothed_poloidal_operator])
            .map_err(invalid_data)?;
    let circulation_operator =
        build_local_circulation_operator(&topology, hodge.mass_u.nrows()).map_err(invalid_data)?;

    let reconstructed_prior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut prior_workspace,
            &reconstructed_stacked_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?,
        cell_geometry.theta.len(),
    )
    .map_err(invalid_data)?;
    let surface_vector_prior_estimates = match config.surface_vector_variance_mode {
        SurfaceVectorVarianceMode::Exact => {
            exact_transformed_variances(&mut prior_workspace, &surface_vector_stacked_operator)?
        }
        SurfaceVectorVarianceMode::Rbmc | SurfaceVectorVarianceMode::RbmcClipped => {
            estimate_transformed_rbmc_variances(
                &mut prior_workspace,
                &surface_vector_stacked_operator,
                config.num_rbmc_probes,
                config.rbmc_batch_count,
                config.rng_seed.wrapping_add(0x1800),
            )?
        }
    };
    let surface_vector_prior =
        split_ambient_estimates(surface_vector_prior_estimates, cell_geometry.theta.len())
            .map_err(invalid_data)?;
    let smoothed_prior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut prior_workspace,
            &smoothed_stacked_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x2000),
        )?,
        cell_geometry.theta.len(),
    )
    .map_err(invalid_data)?;
    let circulation_prior = estimate_transformed_rbmc_variances(
        &mut prior_workspace,
        &circulation_operator,
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?;

    let reconstructed_posterior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut posterior_workspace,
            &reconstructed_stacked_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x1000),
        )?,
        cell_geometry.theta.len(),
    )
    .map_err(invalid_data)?;
    let surface_vector_posterior_estimates = match config.surface_vector_variance_mode {
        SurfaceVectorVarianceMode::Exact => {
            exact_transformed_variances(&mut posterior_workspace, &surface_vector_stacked_operator)?
        }
        SurfaceVectorVarianceMode::Rbmc | SurfaceVectorVarianceMode::RbmcClipped => {
            estimate_transformed_rbmc_variances(
                &mut posterior_workspace,
                &surface_vector_stacked_operator,
                config.num_rbmc_probes,
                config.rbmc_batch_count,
                config.rng_seed.wrapping_add(0x1800),
            )?
        }
    };
    let surface_vector_posterior = if config.surface_vector_variance_mode
        == SurfaceVectorVarianceMode::RbmcClipped
    {
        clip_rbmc_posterior_to_prior(
            &surface_vector_prior,
            &split_ambient_estimates(surface_vector_posterior_estimates, cell_geometry.theta.len())
                .map_err(invalid_data)?,
        )
    } else {
        split_ambient_estimates(surface_vector_posterior_estimates, cell_geometry.theta.len())
            .map_err(invalid_data)?
    };
    let smoothed_posterior = split_component_estimates(
        estimate_transformed_rbmc_variances(
            &mut posterior_workspace,
            &smoothed_stacked_operator,
            config.num_rbmc_probes,
            config.rbmc_batch_count,
            config.rng_seed.wrapping_add(0x2000),
        )?,
        cell_geometry.theta.len(),
    )
    .map_err(invalid_data)?;
    let circulation_posterior = estimate_transformed_rbmc_variances(
        &mut posterior_workspace,
        &circulation_operator,
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed.wrapping_add(0x3000),
    )?;

    let prior_variance = gmrf_vec_to_feec(&prior_latent_variances.unconstrained);
    let posterior_variance = gmrf_vec_to_feec(&posterior_latent_variances.unconstrained);
    let variance_reduction = &prior_variance - &posterior_variance;
    let variance_ratio = ratio_vector(&posterior_variance, &prior_variance);

    let harmonic_free_truth = remove_harmonic_content(&truth, &harmonic_basis_orthonormal, &hodge.mass_u);
    let harmonic_free_posterior_mean =
        remove_harmonic_content(&posterior_mean, &harmonic_basis_orthonormal, &hodge.mass_u);
    let harmonic_free_absolute_mean_error =
        absolute_difference(&harmonic_free_posterior_mean, &harmonic_free_truth);
    let harmonic_free_prior_variance = gmrf_vec_to_feec(&prior_latent_variances.harmonic_free);
    let harmonic_free_posterior_variance =
        gmrf_vec_to_feec(&posterior_latent_variances.harmonic_free);
    let harmonic_free_variance_reduction =
        &harmonic_free_prior_variance - &harmonic_free_posterior_variance;
    let harmonic_free_variance_ratio =
        ratio_vector(&harmonic_free_posterior_variance, &harmonic_free_prior_variance);

    let harmonic_coefficients_truth =
        harmonic_coefficients(&truth, &harmonic_basis_orthonormal, &hodge.mass_u)
            .map_err(invalid_data)?;
    let harmonic_coefficients_posterior_mean =
        harmonic_coefficients(&posterior_mean, &harmonic_basis_orthonormal, &hodge.mass_u)
            .map_err(invalid_data)?;

    let posterior_mean_cochain = Cochain::new(1, posterior_mean.clone());
    let posterior_dif = posterior_mean_cochain.dif(&topology);
    let l2_error = fe_l2_error(&posterior_mean_cochain, &u_exact, &topology, &coords);
    let hd_error = fe_l2_error(&posterior_dif, &dif_solution_exact, &topology, &coords);

    let truth_rhs = &system_matrix * &truth;
    let posterior_rhs = &system_matrix * &posterior_mean;
    let truth_residual = &truth_rhs - &rhs;
    let pde_residual = &posterior_rhs - &rhs;
    let rhs_norm = rhs.norm().max(EPS);

    let variance_fields = Torus1FormPdeVarianceFields {
        reconstructed: build_component_field_set(&reconstructed_prior, &reconstructed_posterior),
        surface_vector: build_ambient_field_set(&surface_vector_prior, &surface_vector_posterior),
        smoothed: build_component_field_set(&smoothed_prior, &smoothed_posterior),
        circulation: build_variance_field_set(&circulation_prior, &circulation_posterior),
    };

    Ok(Torus1FormPdeConditioningResult {
        topology,
        coords,
        edge_theta: FeecVector::from_vec(geometry.theta),
        edge_phi: FeecVector::from_vec(geometry.phi),
        toroidal_alignment_sq: FeecVector::from_vec(geometry.toroidal_alignment_sq),
        major_radius: geometry.major_radius,
        minor_radius: geometry.minor_radius,
        surface_vector_variance_mode: config.surface_vector_variance_mode,
        num_rbmc_probes: config.num_rbmc_probes,
        rbmc_batch_count: config.rbmc_batch_count,
        rng_seed: config.rng_seed,
        effective_range,
        truth,
        rhs,
        posterior_mean,
        posterior_rhs,
        pde_residual: pde_residual.clone(),
        absolute_mean_error,
        prior_variance,
        posterior_variance,
        variance_reduction,
        variance_ratio,
        harmonic_free_truth,
        harmonic_free_posterior_mean,
        harmonic_free_absolute_mean_error,
        harmonic_free_prior_variance,
        harmonic_free_posterior_variance,
        harmonic_free_variance_reduction,
        harmonic_free_variance_ratio,
        harmonic_coefficients_truth,
        harmonic_coefficients_posterior_mean,
        l2_error,
        hd_error,
        truth_residual_norm: truth_residual.norm(),
        truth_relative_residual_norm: truth_residual.norm() / rhs_norm,
        posterior_residual_norm: pde_residual.norm(),
        posterior_relative_residual_norm: pde_residual.norm() / rhs_norm,
        variance_fields,
    })
}

pub fn write_torus_1form_pde_conditioning_outputs(
    result: &Torus1FormPdeConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    write_overall_summary(result, out_dir)?;
    write_edge_fields_vtk(result, out_dir)?;
    write_edge_csv(result, out_dir)?;
    write_surface_vector_vtk(result, out_dir)?;
    write_variance_field_vtks(result, out_dir)?;

    Ok(())
}

fn validate_config(config: &Torus1FormPdeConditioningConfig) -> Result<(), Box<dyn Error>> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive").into());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive").into());
    }
    if config.num_rbmc_probes == 0 {
        return Err(invalid_input("num_rbmc_probes must be >= 1").into());
    }
    if config.rbmc_batch_count == 0 {
        return Err(invalid_input("rbmc_batch_count must be >= 1").into());
    }
    Ok(())
}

fn build_torus_convergence_fields() -> (EmbeddedDiffFormClosure, EmbeddedDiffFormClosure) {
    let u_exact = EmbeddedDiffFormClosure::ambient_one_form(
        move |p: CoordRef| {
            let (theta, phi, rho_val) = torus_angles(p, DEFAULT_MAJOR_RADIUS);

            let a_theta = 2.0 * (2.0 * theta).cos() * (3.0 * phi).cos()
                - 2.0 * DEFAULT_MINOR_RADIUS * theta.cos() / rho_val * (2.0 * phi).cos();

            let a_phi = -3.0 * (2.0 * theta).sin() * (3.0 * phi).sin()
                - rho_val / DEFAULT_MINOR_RADIUS * theta.sin() * (2.0 * phi).sin();

            chart_one_form_to_xyz(p, DEFAULT_MAJOR_RADIUS, a_theta, a_phi)
        },
        3,
        2,
    );

    let dif_solution_exact = EmbeddedDiffFormClosure::ambient_k_form(
        move |p: CoordRef| {
            let (theta, phi, rho_val) = torus_angles(p, DEFAULT_MAJOR_RADIUS);

            let coeff_theta_phi = (theta.sin().powi(2)
                - (rho_val / DEFAULT_MINOR_RADIUS + 4.0 * DEFAULT_MINOR_RADIUS / rho_val)
                    * theta.cos())
                * (2.0 * phi).sin();

            chart_two_form_to_xyz(p, DEFAULT_MAJOR_RADIUS, coeff_theta_phi)
        },
        3,
        2,
        2,
    );

    (u_exact, dif_solution_exact)
}

fn torus_angles(p: CoordRef, major_radius: f64) -> (f64, f64, f64) {
    let x = p[0];
    let y = p[1];
    let z = p[2];

    let s = (x * x + y * y).sqrt();

    let phi = y.atan2(x);
    let theta = z.atan2(s - major_radius);
    let rho = s;

    (theta, phi, rho)
}

fn torus_covectors(p: CoordRef, major_radius: f64) -> (FeecVector, FeecVector) {
    let x = p[0];
    let y = p[1];
    let z = p[2];

    let s = (x * x + y * y).sqrt();
    let q = (s - major_radius).powi(2) + z * z;

    let dtheta_x = -z * x / (s * q);
    let dtheta_y = -z * y / (s * q);
    let dtheta_z = (s - major_radius) / q;

    let dphi_x = -y / (s * s);
    let dphi_y = x / (s * s);
    let dphi_z = 0.0;

    let dphi = FeecVector::from_vec(vec![dphi_x, dphi_y, dphi_z]);
    let dtheta = FeecVector::from_vec(vec![dtheta_x, dtheta_y, dtheta_z]);
    (dtheta, dphi)
}

fn chart_one_form_to_xyz(
    p: CoordRef,
    major_radius: f64,
    a_theta: f64,
    a_phi: f64,
) -> FeecVector {
    let (dtheta, dphi) = torus_covectors(p, major_radius);
    a_theta * dtheta + a_phi * dphi
}

fn chart_two_form_to_xyz(
    p: CoordRef,
    major_radius: f64,
    coeff_theta_phi: f64,
) -> FeecVector {
    let (dtheta, dphi) = torus_covectors(p, major_radius);
    coeff_theta_phi
        * ExteriorElement::line(dtheta)
            .wedge(&ExteriorElement::line(dphi))
            .into_coeffs()
}

fn build_rbmc_workspace(
    precision: &GmrfSparseMatrix,
    harmonic_constraints: &GmrfDenseMatrix,
) -> Result<RbmcWorkspace, Box<dyn Error>> {
    let q_factor = precision.cholesky_sqrt_lower()?;
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?
            .with_precision_sqrt(q_factor);
    let constraint_correction =
        build_constraint_variance_correction(&mut gmrf, harmonic_constraints)?;
    Ok(RbmcWorkspace {
        gmrf,
        constraint_correction,
    })
}

fn build_constraint_variance_correction(
    gmrf: &mut Gmrf,
    harmonic_constraints: &GmrfDenseMatrix,
) -> Result<Option<ConstraintVarianceCorrection>, GmrfError> {
    if harmonic_constraints.nrows() == 0 {
        return Ok(None);
    }

    let covariance_times_constraint_t = covariance_times_constraint_t(gmrf, harmonic_constraints)?;
    let schur = schur_complement(harmonic_constraints, &covariance_times_constraint_t);
    let schur_inverse = invert_spd_dense(&schur)?;

    Ok(Some(ConstraintVarianceCorrection {
        covariance_times_constraint_t,
        schur_inverse,
    }))
}

fn exact_latent_variances(
    workspace: &mut RbmcWorkspace,
    harmonic_constraints: &GmrfDenseMatrix,
) -> Result<RbmcVarianceEstimates, GmrfError> {
    let decomposition = workspace
        .gmrf
        .exact_constrained_variance_decomposition(harmonic_constraints)?;
    Ok(RbmcVarianceEstimates {
        unconstrained: decomposition.unconstrained_diag,
        harmonic_free: decomposition.constrained_diag,
    })
}

fn exact_transformed_variances(
    workspace: &mut RbmcWorkspace,
    operator: &SparseRowLinearOperator,
) -> Result<RbmcVarianceEstimates, GmrfError> {
    let mut unconstrained = GmrfVector::zeros(operator.nrows());
    for (row_index, row) in operator.rows.iter().enumerate() {
        let rhs = sparse_row_rhs(row, operator.ncols);
        let solved = workspace.gmrf.solve_precision(&rhs)?;
        let value = row
            .iter()
            .map(|(state_index, weight)| *weight * solved[*state_index])
            .sum::<f64>();
        unconstrained[row_index] = clamp_small_negative_variance(
            value,
            rhs.norm().max(1.0),
            "transformed unconstrained marginal variance must be nonnegative",
        )?;
    }

    let harmonic_free = if let Some(correction) = &workspace.constraint_correction {
        let removed = transformed_constraint_correction_diag(operator, correction);
        let mut constrained = GmrfVector::zeros(operator.nrows());
        for i in 0..operator.nrows() {
            let scale = unconstrained[i].abs().max(1.0);
            let max_removed = unconstrained[i] + EXACT_VARIANCE_TOLERANCE * scale;
            if removed[i] > max_removed {
                return Err(GmrfError::NumericalInstability(
                    "transformed removed marginal variance exceeded unconstrained variance",
                ));
            }
            constrained[i] = clamp_small_negative_variance(
                unconstrained[i] - removed[i].min(unconstrained[i]),
                scale,
                "transformed constrained marginal variance must be nonnegative",
            )?;
        }
        constrained
    } else {
        unconstrained.clone()
    };

    Ok(RbmcVarianceEstimates {
        unconstrained,
        harmonic_free,
    })
}

fn estimate_transformed_rbmc_variances(
    workspace: &mut RbmcWorkspace,
    operator: &SparseRowLinearOperator,
    num_rbmc_probes: usize,
    rbmc_batch_count: usize,
    rng_seed: u64,
) -> Result<RbmcVarianceEstimates, Box<dyn Error>> {
    let batch_sizes = rbmc_batch_sizes(num_rbmc_probes, rbmc_batch_count);
    let mut batch_estimates = Vec::with_capacity(batch_sizes.len());
    for (batch_idx, batch_size) in batch_sizes.iter().copied().enumerate() {
        let batch_seed = rng_seed.wrapping_add(
            0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((batch_idx as u64).wrapping_add(1)),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(batch_seed);
        let batch_raw =
            transformed_rbmc_variances_batch(&mut workspace.gmrf, operator, batch_size, &mut rng)?;
        let (batch_stabilized, _floor_hits) = stabilize_positive_variances(&batch_raw);
        batch_estimates.push(batch_stabilized);
    }

    let unconstrained =
        weighted_average_vectors(&batch_estimates, &batch_sizes).map_err(invalid_data)?;
    let removed_diag = if let Some(correction) = &workspace.constraint_correction {
        transformed_constraint_correction_diag(operator, correction)
    } else {
        GmrfVector::zeros(operator.nrows())
    };
    let (harmonic_free, _floor_hits) =
        stabilize_harmonic_free_variances(&unconstrained, &removed_diag);

    Ok(RbmcVarianceEstimates {
        unconstrained,
        harmonic_free,
    })
}

fn transformed_rbmc_variances_batch(
    gmrf: &mut Gmrf,
    operator: &SparseRowLinearOperator,
    num_samples: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<GmrfVector, GmrfError> {
    if num_samples == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one RBMC probe is required",
        ));
    }

    let output_dim = operator.nrows();
    let mut variances = GmrfVector::zeros(output_dim);
    for _ in 0..num_samples {
        let probe = GmrfVector::from_fn(output_dim, |_| rng.sample(StandardNormal));
        let rhs = operator.apply_transpose(&probe);
        let solved = gmrf.solve_precision(&rhs)?;
        let projected = operator.apply(&solved);
        variances += projected.component_mul(&probe);
    }

    Ok(variances / (num_samples as f64))
}

fn sparse_row_rhs(row: &[(usize, f64)], dimension: usize) -> GmrfVector {
    let mut rhs = GmrfVector::zeros(dimension);
    for (col, value) in row {
        rhs[*col] = *value;
    }
    rhs
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

fn invert_spd_dense(matrix: &GmrfDenseMatrix) -> Result<GmrfDenseMatrix, GmrfError> {
    let factor = matrix
        .clone()
        .llt(Side::Lower)
        .map_err(|_| GmrfError::SingularConstraintSystem)?;
    let dim = matrix.nrows();
    let mut columns = Vec::with_capacity(dim);
    for j in 0..dim {
        let mut rhs = GmrfVector::zeros(dim);
        rhs[j] = 1.0;
        factor.solve_in_place(rhs.as_col_mut().as_mat_mut());
        columns.push(rhs);
    }
    Ok(GmrfDenseMatrix::from_fn(dim, dim, |i, j| columns[j][i]))
}

fn transformed_constraint_correction_diag(
    operator: &SparseRowLinearOperator,
    correction: &ConstraintVarianceCorrection,
) -> GmrfVector {
    GmrfVector::from_iterator(
        operator.nrows(),
        operator.rows.iter().map(|row| {
            let mut g = GmrfVector::zeros(correction.schur_inverse.nrows());
            for (constraint_idx, column) in correction
                .covariance_times_constraint_t
                .as_ref()
                .col_iter()
                .enumerate()
            {
                let column = column
                    .try_as_col_major()
                    .expect("dense matrix is column-major");
                g[constraint_idx] = row
                    .iter()
                    .map(|(state_idx, value)| *value * column.as_slice()[*state_idx])
                    .sum::<f64>();
            }
            quadratic_form_dense(&correction.schur_inverse, &g)
        }),
    )
}

fn quadratic_form_dense(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> f64 {
    let applied = dense_matvec(matrix, vector);
    vector.dot(&applied)
}

fn split_component_estimates(
    stacked: RbmcVarianceEstimates,
    cell_count: usize,
) -> Result<Torus1FormVarianceComponentEstimates, String> {
    if stacked.unconstrained.len() != 2 * cell_count
        || stacked.harmonic_free.len() != 2 * cell_count
    {
        return Err(
            "stacked component estimates must contain toroidal and poloidal blocks".to_string(),
        );
    }

    Ok(Torus1FormVarianceComponentEstimates {
        toroidal: RbmcVarianceEstimates {
            unconstrained: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.unconstrained[i]),
            ),
            harmonic_free: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.harmonic_free[i]),
            ),
        },
        poloidal: RbmcVarianceEstimates {
            unconstrained: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.unconstrained[cell_count + i]),
            ),
            harmonic_free: GmrfVector::from_iterator(
                cell_count,
                (0..cell_count).map(|i| stacked.harmonic_free[cell_count + i]),
            ),
        },
    })
}

fn split_ambient_estimates(
    stacked: RbmcVarianceEstimates,
    cell_count: usize,
) -> Result<Torus1FormAmbientVarianceEstimates, String> {
    if stacked.unconstrained.len() != 3 * cell_count
        || stacked.harmonic_free.len() != 3 * cell_count
    {
        return Err(
            "stacked ambient estimates must contain x, y, and z component blocks".to_string(),
        );
    }

    let block = |values: &GmrfVector, block_index: usize| {
        GmrfVector::from_iterator(
            cell_count,
            (0..cell_count).map(|i| values[block_index * cell_count + i]),
        )
    };

    Ok(Torus1FormAmbientVarianceEstimates {
        x: RbmcVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 0),
            harmonic_free: block(&stacked.harmonic_free, 0),
        },
        y: RbmcVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 1),
            harmonic_free: block(&stacked.harmonic_free, 1),
        },
        z: RbmcVarianceEstimates {
            unconstrained: block(&stacked.unconstrained, 2),
            harmonic_free: block(&stacked.harmonic_free, 2),
        },
    })
}

fn clip_rbmc_posterior_to_prior(
    prior: &Torus1FormAmbientVarianceEstimates,
    posterior: &Torus1FormAmbientVarianceEstimates,
) -> Torus1FormAmbientVarianceEstimates {
    Torus1FormAmbientVarianceEstimates {
        x: clip_rbmc_estimates_to_prior(&prior.x, &posterior.x),
        y: clip_rbmc_estimates_to_prior(&prior.y, &posterior.y),
        z: clip_rbmc_estimates_to_prior(&prior.z, &posterior.z),
    }
}

fn clip_rbmc_estimates_to_prior(
    prior: &RbmcVarianceEstimates,
    posterior: &RbmcVarianceEstimates,
) -> RbmcVarianceEstimates {
    RbmcVarianceEstimates {
        unconstrained: clip_vector_to_prior(&prior.unconstrained, &posterior.unconstrained),
        harmonic_free: clip_vector_to_prior(&prior.harmonic_free, &posterior.harmonic_free),
    }
}

fn clip_vector_to_prior(prior: &GmrfVector, posterior: &GmrfVector) -> GmrfVector {
    GmrfVector::from_iterator(
        prior.len(),
        (0..prior.len()).map(|i| posterior[i].max(0.0).min(prior[i].max(0.0))),
    )
}

fn rbmc_batch_sizes(num_probes: usize, batch_count: usize) -> Vec<usize> {
    let batch_count = batch_count.min(num_probes).max(1);
    let base = num_probes / batch_count;
    let remainder = num_probes % batch_count;
    (0..batch_count)
        .map(|batch_idx| base + usize::from(batch_idx < remainder))
        .collect()
}

fn weighted_average_vectors(
    vectors: &[GmrfVector],
    weights: &[usize],
) -> Result<GmrfVector, String> {
    if vectors.is_empty() || vectors.len() != weights.len() {
        return Err("RBMC batches and weights must be nonempty and aligned".to_string());
    }
    let dim = vectors[0].len();
    if weights.iter().all(|weight| *weight == 0) {
        return Err("at least one RBMC probe is required".to_string());
    }
    if vectors.iter().any(|vec| vec.len() != dim) {
        return Err("RBMC batch dimensions must agree".to_string());
    }

    let total_weight = weights.iter().sum::<usize>() as f64;
    let mut out = GmrfVector::zeros(dim);
    for (vec, weight) in vectors.iter().zip(weights.iter().copied()) {
        let scale = weight as f64 / total_weight;
        for i in 0..dim {
            out[i] += scale * vec[i];
        }
    }
    Ok(out)
}

fn stabilize_positive_variances(variances: &GmrfVector) -> (GmrfVector, usize) {
    let positive_sum = variances
        .iter()
        .copied()
        .filter(|value| *value > EPS)
        .sum::<f64>();
    let positive_count = variances.iter().filter(|value| **value > EPS).count();
    let positive_mean = if positive_count > 0 {
        positive_sum / positive_count as f64
    } else {
        1.0
    };
    let floor = (positive_mean.abs().max(1.0)) * 1e-12;

    let mut hits = 0_usize;
    let stabilized = GmrfVector::from_iterator(
        variances.len(),
        variances.iter().copied().map(|value| {
            if value > floor {
                value
            } else {
                hits += 1;
                floor
            }
        }),
    );
    (stabilized, hits)
}

fn stabilize_harmonic_free_variances(
    unconstrained: &GmrfVector,
    removed: &GmrfVector,
) -> (GmrfVector, usize) {
    let positive_sum = unconstrained
        .iter()
        .copied()
        .filter(|value| *value > EPS)
        .sum::<f64>();
    let positive_count = unconstrained.iter().filter(|value| **value > EPS).count();
    let positive_mean = if positive_count > 0 {
        positive_sum / positive_count as f64
    } else {
        1.0
    };
    let floor = (positive_mean.abs().max(1.0)) * 1e-12;

    let mut hits = 0_usize;
    let stabilized = GmrfVector::from_iterator(
        unconstrained.len(),
        (0..unconstrained.len()).map(|i| {
            let raw = unconstrained[i] - removed[i];
            if raw > floor {
                raw
            } else {
                hits += 1;
                floor
            }
        }),
    );
    (stabilized, hits)
}

fn build_torus_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<TorusEdgeGeometry, io::Error> {
    let (major_radius, minor_radius) = infer_torus_radii(coords).map_err(invalid_data)?;
    let edge_skeleton = topology.skeleton(1);

    let mut theta = Vec::with_capacity(edge_skeleton.len());
    let mut phi = Vec::with_capacity(edge_skeleton.len());
    let mut toroidal_alignment_sq = Vec::with_capacity(edge_skeleton.len());

    for edge in edge_skeleton.handle_iter() {
        let v0 = coords.coord(edge.vertices[0]);
        let v1 = coords.coord(edge.vertices[1]);
        let midpoint = (v0 + v1) / 2.0;
        let rho = (midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1])
            .sqrt()
            .max(EPS);
        let midpoint_theta = midpoint[2].atan2(rho - major_radius);
        let midpoint_phi = midpoint[1].atan2(midpoint[0]);

        let tangent = v1 - v0;
        let tangent_norm = tangent.norm();
        let alignment_sq = if tangent_norm <= EPS {
            0.0
        } else {
            let e_phi =
                FeecVector::from_column_slice(&[-midpoint_phi.sin(), midpoint_phi.cos(), 0.0]);
            let unit_tangent = tangent / tangent_norm;
            unit_tangent.dot(&e_phi).powi(2).clamp(0.0, 1.0)
        };

        theta.push(midpoint_theta);
        phi.push(midpoint_phi);
        toroidal_alignment_sq.push(alignment_sq);
    }

    Ok(TorusEdgeGeometry {
        major_radius,
        minor_radius,
        theta,
        phi,
        toroidal_alignment_sq,
    })
}

fn build_torus_cell_geometry(
    topology: &Complex,
    coords: &MeshCoords,
    major_radius: f64,
    minor_radius: f64,
) -> Result<TorusCellGeometry, String> {
    let mut theta = Vec::with_capacity(topology.cells().len());
    let mut phi = Vec::with_capacity(topology.cells().len());

    for cell in topology.cells().handle_iter() {
        let barycenter = cell.coord_simplex(coords).barycenter();
        let x = barycenter[0];
        let y = barycenter[1];
        let z = barycenter[2];
        let rho = (x * x + y * y).sqrt().max(EPS);
        theta.push(z.atan2(rho - major_radius));
        phi.push(y.atan2(x));
    }

    Ok(TorusCellGeometry {
        major_radius,
        minor_radius,
        theta,
        phi,
    })
}

fn build_reconstructed_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    cell_geometry: &TorusCellGeometry,
    toroidal_component: bool,
) -> Result<SparseRowLinearOperator, String> {
    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(cell_skeleton.len());

    for (cell_index, cell) in cell_skeleton.handle_iter().enumerate() {
        let theta = cell_geometry.theta[cell_index];
        let phi = cell_geometry.phi[cell_index];
        let direction = if toroidal_component {
            [-phi.sin(), phi.cos(), 0.0]
        } else {
            [
                -theta.sin() * phi.cos(),
                -theta.sin() * phi.sin(),
                theta.cos(),
            ]
        };

        let cell_coords = cell.coord_simplex(coords);
        let jacobian_pinv = cell_coords.inv_linear_transform();
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let local_value = lsf.at_point(&bary_local).into_grade1();
            let ambient_value = if topo_dim == coords.dim() {
                local_value
            } else {
                jacobian_pinv.transpose() * local_value
            };
            let coefficient = ambient_value[0] * direction[0]
                + ambient_value[1] * direction[1]
                + ambient_value[2] * direction[2];
            if coefficient.abs() > EPS {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(topology.skeleton(1).len(), rows)
}

fn build_embedded_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowLinearOperator, String> {
    if component_index >= coords.dim() {
        return Err(format!(
            "ambient component index {} is out of range for coordinate dimension {}",
            component_index,
            coords.dim()
        ));
    }

    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(cell_skeleton.len());

    for cell in cell_skeleton.handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let jacobian_pinv = cell_coords.inv_linear_transform();
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(1) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let local_value = lsf.at_point(&bary_local).into_grade1();
            let ambient_value = if topo_dim == coords.dim() {
                local_value
            } else {
                jacobian_pinv.transpose() * local_value
            };
            let coefficient = ambient_value[component_index];
            if coefficient.abs() > EPS {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(topology.skeleton(1).len(), rows)
}

fn build_gaussian_smoothing_operator(
    cell_geometry: &TorusCellGeometry,
    bandwidth: f64,
    cutoff: f64,
) -> Result<SparseRowLinearOperator, String> {
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err("smoothing bandwidth must be finite and positive".to_string());
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err("smoothing cutoff must be finite and positive".to_string());
    }

    let mut rows = Vec::with_capacity(cell_geometry.theta.len());
    for row_index in 0..cell_geometry.theta.len() {
        let mut row = Vec::new();
        let mut weight_sum = 0.0;
        for col_index in 0..cell_geometry.theta.len() {
            let distance = intrinsic_torus_distance(
                cell_geometry.major_radius,
                cell_geometry.minor_radius,
                cell_geometry.theta[row_index],
                cell_geometry.phi[row_index],
                cell_geometry.theta[col_index],
                cell_geometry.phi[col_index],
            );
            if distance > cutoff {
                continue;
            }
            let weight = (-0.5 * (distance / bandwidth).powi(2)).exp();
            if weight <= EPS {
                continue;
            }
            row.push((col_index, weight));
            weight_sum += weight;
        }

        if weight_sum <= EPS {
            row.push((row_index, 1.0));
            weight_sum = 1.0;
        }
        for (_, weight) in row.iter_mut() {
            *weight /= weight_sum;
        }
        rows.push(row);
    }

    SparseRowLinearOperator::new(cell_geometry.theta.len(), rows)
}

fn build_local_circulation_operator(
    topology: &Complex,
    edge_count: usize,
) -> Result<SparseRowLinearOperator, String> {
    let boundary = topology.boundary_operator(topology.dim());
    let mut rows = vec![Vec::new(); topology.cells().len()];
    for (edge_index, cell_index, value) in boundary.triplet_iter() {
        if edge_index >= edge_count || cell_index >= rows.len() {
            return Err("invalid face-boundary incidence entry".to_string());
        }
        if value.abs() > EPS {
            rows[cell_index].push((edge_index, *value));
        }
    }
    SparseRowLinearOperator::new(edge_count, rows)
}

fn mass_orthonormalize_harmonic_basis(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<FeecMatrix, String> {
    if harmonic_basis.ncols() == 0 {
        return Err("harmonic basis must contain at least one column".to_string());
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
                "harmonic basis column {j} became singular during orthonormalization"
            ));
        }
        column /= norm_sq.sqrt();
        columns.push(column);
    }

    Ok(FeecMatrix::from_columns(&columns))
}

fn harmonic_coefficients(
    field: &FeecVector,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<[f64; 2], String> {
    if harmonic_basis_orthonormal.ncols() != 2 {
        return Err(format!(
            "expected exactly two harmonic basis vectors on the torus, found {}",
            harmonic_basis_orthonormal.ncols()
        ));
    }

    Ok([
        mass_inner_product(
            &harmonic_basis_orthonormal.column(0).into_owned(),
            field,
            mass_u,
        ),
        mass_inner_product(
            &harmonic_basis_orthonormal.column(1).into_owned(),
            field,
            mass_u,
        ),
    ])
}

fn remove_harmonic_content(
    field: &FeecVector,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
) -> FeecVector {
    let mut harmonic_free = field.clone();
    for j in 0..harmonic_basis_orthonormal.ncols() {
        let basis_col = harmonic_basis_orthonormal.column(j).into_owned();
        let coeff = mass_inner_product(&basis_col, field, mass_u);
        harmonic_free -= basis_col.scale(coeff);
    }
    harmonic_free
}

fn mass_inner_product(lhs: &FeecVector, rhs: &FeecVector, mass_u: &FeecCsr) -> f64 {
    let weighted_rhs = mass_u * rhs;
    lhs.dot(&weighted_rhs)
}

fn build_variance_field_set(
    prior: &RbmcVarianceEstimates,
    posterior: &RbmcVarianceEstimates,
) -> Torus1FormVarianceFieldSet {
    let prior = gmrf_vec_to_feec(&prior.unconstrained);
    let posterior = gmrf_vec_to_feec(&posterior.unconstrained);
    let ratio = ratio_vector(&posterior, &prior);
    Torus1FormVarianceFieldSet {
        prior,
        posterior,
        ratio,
    }
}

fn build_component_field_set(
    prior: &Torus1FormVarianceComponentEstimates,
    posterior: &Torus1FormVarianceComponentEstimates,
) -> Torus1FormVarianceComponentFields {
    let toroidal = build_variance_field_set(&prior.toroidal, &posterior.toroidal);
    let poloidal = build_variance_field_set(&prior.poloidal, &posterior.poloidal);
    let trace_prior = &toroidal.prior + &poloidal.prior;
    let trace_posterior = &toroidal.posterior + &poloidal.posterior;
    let trace_ratio = ratio_vector(&trace_posterior, &trace_prior);
    Torus1FormVarianceComponentFields {
        toroidal,
        poloidal,
        trace: Torus1FormVarianceFieldSet {
            prior: trace_prior,
            posterior: trace_posterior,
            ratio: trace_ratio,
        },
    }
}

fn build_ambient_field_set(
    prior: &Torus1FormAmbientVarianceEstimates,
    posterior: &Torus1FormAmbientVarianceEstimates,
) -> Torus1FormAmbientVarianceFields {
    let x = build_variance_field_set(&prior.x, &posterior.x);
    let y = build_variance_field_set(&prior.y, &posterior.y);
    let z = build_variance_field_set(&prior.z, &posterior.z);
    let trace_prior = &x.prior + &y.prior + &z.prior;
    let trace_posterior = &x.posterior + &y.posterior + &z.posterior;
    let trace_ratio = ratio_vector(&trace_posterior, &trace_prior);

    Torus1FormAmbientVarianceFields {
        x,
        y,
        z,
        trace: Torus1FormVarianceFieldSet {
            prior: trace_prior,
            posterior: trace_posterior,
            ratio: trace_ratio,
        },
    }
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

fn absolute_difference(lhs: &FeecVector, rhs: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(lhs.len(), (0..lhs.len()).map(|i| (lhs[i] - rhs[i]).abs()))
}

fn intrinsic_torus_distance(
    major_radius: f64,
    minor_radius: f64,
    theta: f64,
    phi: f64,
    theta_ref: f64,
    phi_ref: f64,
) -> f64 {
    let delta_theta = wrap_angle_difference(theta, theta_ref);
    let delta_phi = wrap_angle_difference(phi, phi_ref);
    let phi_scale = major_radius + minor_radius * ((theta + theta_ref) * 0.5).cos();
    ((minor_radius * delta_theta).powi(2) + (phi_scale * delta_phi).powi(2)).sqrt()
}

fn wrap_angle_difference(angle: f64, reference: f64) -> f64 {
    let mut delta = angle - reference;
    while delta <= -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    delta
}

fn covariance_times_constraint_t(
    posterior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
) -> Result<GmrfDenseMatrix, GmrfError> {
    let state_dim = posterior.dimension();
    let constraint_dim = constraint_matrix.nrows();
    let mut columns = Vec::with_capacity(constraint_dim);
    for row in 0..constraint_dim {
        let rhs = dense_row_as_vector(constraint_matrix, row);
        let solved = posterior.solve_precision(&rhs)?;
        columns.push(solved);
    }

    Ok(GmrfDenseMatrix::from_fn(
        state_dim,
        constraint_dim,
        |i, j| columns[j][i],
    ))
}

fn dense_matvec(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> GmrfVector {
    let mut out = GmrfVector::zeros(matrix.nrows());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let xj = vector[j];
        if xj == 0.0 {
            continue;
        }
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        for (i, value) in col.as_slice().iter().enumerate() {
            out[i] += *value * xj;
        }
    }
    out
}

fn dense_row_as_vector(matrix: &GmrfDenseMatrix, row: usize) -> GmrfVector {
    let mut out = GmrfVector::zeros(matrix.ncols());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        out[j] = col.as_slice()[row];
    }
    out
}

fn schur_complement(
    constraint_matrix: &GmrfDenseMatrix,
    covariance_times_constraint_t: &GmrfDenseMatrix,
) -> GmrfDenseMatrix {
    let mut schur = GmrfDenseMatrix::zeros(constraint_matrix.nrows(), constraint_matrix.nrows());
    for i in 0..constraint_matrix.nrows() {
        for j in 0..constraint_matrix.nrows() {
            let mut sum = 0.0;
            for k in 0..constraint_matrix.ncols() {
                sum += constraint_matrix[(i, k)] * covariance_times_constraint_t[(k, j)];
            }
            schur[(i, j)] = sum;
        }
    }
    schur
}

fn gmrf_vec_to_feec(vec: &GmrfVector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

fn mean(values: &FeecVector) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn max_value(values: &FeecVector) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

fn write_overall_summary(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Torus 1-form Matérn PDE conditioning")?;
    writeln!(writer, "major_radius={}", result.major_radius)?;
    writeln!(writer, "minor_radius={}", result.minor_radius)?;
    writeln!(writer, "num_rbmc_probes={}", result.num_rbmc_probes)?;
    writeln!(writer, "rbmc_batch_count={}", result.rbmc_batch_count)?;
    writeln!(writer, "rng_seed={}", result.rng_seed)?;
    writeln!(
        writer,
        "surface_vector_variance_mode={}",
        result.surface_vector_variance_mode.as_str()
    )?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "harmonic_coefficients_truth={},{}",
        result.harmonic_coefficients_truth[0], result.harmonic_coefficients_truth[1]
    )?;
    writeln!(
        writer,
        "harmonic_coefficients_posterior_mean={},{}",
        result.harmonic_coefficients_posterior_mean[0],
        result.harmonic_coefficients_posterior_mean[1]
    )?;
    writeln!(writer, "l2_error={}", result.l2_error)?;
    writeln!(writer, "hd_error={}", result.hd_error)?;
    writeln!(writer, "truth_residual_norm={}", result.truth_residual_norm)?;
    writeln!(
        writer,
        "truth_relative_residual_norm={}",
        result.truth_relative_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_residual_norm={}",
        result.posterior_residual_norm
    )?;
    writeln!(
        writer,
        "posterior_relative_residual_norm={}",
        result.posterior_relative_residual_norm
    )?;
    writeln!(writer, "edge_mean_abs_error={}", mean(&result.absolute_mean_error))?;
    writeln!(
        writer,
        "edge_max_abs_error={}",
        max_value(&result.absolute_mean_error)
    )?;
    writeln!(writer, "edge_variance_ratio_mean={}", mean(&result.variance_ratio))?;
    writeln!(
        writer,
        "harmonic_free_edge_variance_ratio_mean={}",
        mean(&result.harmonic_free_variance_ratio)
    )?;
    writeln!(
        writer,
        "surface_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.surface_vector.trace.ratio)
    )?;
    writeln!(
        writer,
        "reconstructed_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.reconstructed.trace.ratio)
    )?;
    writeln!(
        writer,
        "smoothed_trace_variance_ratio_mean={}",
        mean(&result.variance_fields.smoothed.trace.ratio)
    )?;
    writeln!(
        writer,
        "circulation_variance_ratio_mean={}",
        mean(&result.variance_fields.circulation.ratio)
    )?;
    Ok(())
}

fn write_edge_fields_vtk(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let truth = Cochain::new(1, result.truth.clone());
    let rhs = Cochain::new(1, result.rhs.clone());
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_rhs = Cochain::new(1, result.posterior_rhs.clone());
    let pde_residual = Cochain::new(1, result.pde_residual.clone());
    let absolute_mean_error = Cochain::new(1, result.absolute_mean_error.clone());
    let prior_variance = Cochain::new(1, result.prior_variance.clone());
    let posterior_variance = Cochain::new(1, result.posterior_variance.clone());
    let variance_reduction = Cochain::new(1, result.variance_reduction.clone());
    let variance_ratio = Cochain::new(1, result.variance_ratio.clone());
    let harmonic_free_truth = Cochain::new(1, result.harmonic_free_truth.clone());
    let harmonic_free_posterior_mean = Cochain::new(1, result.harmonic_free_posterior_mean.clone());
    let harmonic_free_absolute_mean_error =
        Cochain::new(1, result.harmonic_free_absolute_mean_error.clone());
    let harmonic_free_prior_variance =
        Cochain::new(1, result.harmonic_free_prior_variance.clone());
    let harmonic_free_posterior_variance =
        Cochain::new(1, result.harmonic_free_posterior_variance.clone());
    let harmonic_free_variance_reduction =
        Cochain::new(1, result.harmonic_free_variance_reduction.clone());
    let harmonic_free_variance_ratio =
        Cochain::new(1, result.harmonic_free_variance_ratio.clone());
    let edge_theta = Cochain::new(1, result.edge_theta.clone());
    let edge_phi = Cochain::new(1, result.edge_phi.clone());
    let toroidal_alignment_sq = Cochain::new(1, result.toroidal_alignment_sq.clone());

    write_1cochain_vtk_fields(
        out_dir.join("fields.vtk"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("rhs", &rhs),
            ("posterior_mean", &posterior_mean),
            ("posterior_rhs", &posterior_rhs),
            ("pde_residual", &pde_residual),
            ("absolute_mean_error", &absolute_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
            ("variance_ratio", &variance_ratio),
            ("harmonic_free_truth", &harmonic_free_truth),
            (
                "harmonic_free_posterior_mean",
                &harmonic_free_posterior_mean,
            ),
            (
                "harmonic_free_absolute_mean_error",
                &harmonic_free_absolute_mean_error,
            ),
            (
                "harmonic_free_prior_variance",
                &harmonic_free_prior_variance,
            ),
            (
                "harmonic_free_posterior_variance",
                &harmonic_free_posterior_variance,
            ),
            (
                "harmonic_free_variance_reduction",
                &harmonic_free_variance_reduction,
            ),
            (
                "harmonic_free_variance_ratio",
                &harmonic_free_variance_ratio,
            ),
            ("edge_theta", &edge_theta),
            ("edge_phi", &edge_phi),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;

    write_1form_vector_proxy_vtk_fields(
        out_dir.join("posterior_mean_vector.vtk"),
        &result.coords,
        &result.topology,
        "posterior_mean_vector",
        &posterior_mean,
        &[
            ("truth", &truth),
            ("absolute_mean_error", &absolute_mean_error),
            ("posterior_variance", &posterior_variance),
            ("pde_residual", &pde_residual),
        ],
    )?;

    Ok(())
}

fn write_surface_vector_vtk(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let posterior_mean = Cochain::new(1, result.posterior_mean.clone());
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let posterior_mean_magnitude = vector_magnitudes(&posterior_mean_vectors);
    let surface = &result.variance_fields.surface_vector;
    let posterior_variance_vectors = ambient_variance_vectors(surface, false);
    let prior_variance_vectors = ambient_variance_vectors(surface, true);
    let posterior_marginal_std = surface.trace.posterior.map(|value| value.max(0.0).sqrt());

    write_top_cell_vtk_fields(
        out_dir.join("posterior_mean_surface_vector.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "posterior_mean_surface_vector",
                posterior_mean_vectors.as_slice(),
            ),
            (
                "posterior_directional_variance",
                posterior_variance_vectors.as_slice(),
            ),
            (
                "prior_directional_variance",
                prior_variance_vectors.as_slice(),
            ),
        ],
        &[
            ("magnitude", posterior_mean_magnitude.as_slice()),
            ("marginal_variance", surface.trace.posterior.as_slice()),
            ("marginal_std", posterior_marginal_std.as_slice()),
            ("prior_marginal_variance", surface.trace.prior.as_slice()),
            ("marginal_variance_ratio", surface.trace.ratio.as_slice()),
        ],
    )?;

    Ok(())
}

fn write_edge_csv(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("edge_fields.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "edge_index,theta,phi,toroidal_alignment_sq,truth,rhs,posterior_mean,posterior_rhs,pde_residual,absolute_mean_error,prior_variance,posterior_variance,variance_reduction,variance_ratio,harmonic_free_truth,harmonic_free_posterior_mean,harmonic_free_absolute_mean_error,harmonic_free_prior_variance,harmonic_free_posterior_variance,harmonic_free_variance_reduction,harmonic_free_variance_ratio"
    )?;

    for edge_index in 0..result.truth.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            edge_index,
            result.edge_theta[edge_index],
            result.edge_phi[edge_index],
            result.toroidal_alignment_sq[edge_index],
            result.truth[edge_index],
            result.rhs[edge_index],
            result.posterior_mean[edge_index],
            result.posterior_rhs[edge_index],
            result.pde_residual[edge_index],
            result.absolute_mean_error[edge_index],
            result.prior_variance[edge_index],
            result.posterior_variance[edge_index],
            result.variance_reduction[edge_index],
            result.variance_ratio[edge_index],
            result.harmonic_free_truth[edge_index],
            result.harmonic_free_posterior_mean[edge_index],
            result.harmonic_free_absolute_mean_error[edge_index],
            result.harmonic_free_prior_variance[edge_index],
            result.harmonic_free_posterior_variance[edge_index],
            result.harmonic_free_variance_reduction[edge_index],
            result.harmonic_free_variance_ratio[edge_index],
        )?;
    }

    Ok(())
}

fn write_variance_field_vtks(
    result: &Torus1FormPdeConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    write_top_cell_scalar_vtk_fields(
        out_dir.join("reconstructed_component_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_toroidal",
                result
                    .variance_fields
                    .reconstructed
                    .toroidal
                    .ratio
                    .as_slice(),
            ),
            (
                "prior_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_poloidal",
                result
                    .variance_fields
                    .reconstructed
                    .poloidal
                    .ratio
                    .as_slice(),
            ),
            (
                "trace_prior",
                result.variance_fields.reconstructed.trace.prior.as_slice(),
            ),
            (
                "trace_post",
                result
                    .variance_fields
                    .reconstructed
                    .trace
                    .posterior
                    .as_slice(),
            ),
            (
                "trace_ratio",
                result.variance_fields.reconstructed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    write_top_cell_scalar_vtk_fields(
        out_dir.join("smoothed_component_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "smoothed_prior_toroidal",
                result.variance_fields.smoothed.toroidal.prior.as_slice(),
            ),
            (
                "smoothed_post_toroidal",
                result
                    .variance_fields
                    .smoothed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_toroidal",
                result.variance_fields.smoothed.toroidal.ratio.as_slice(),
            ),
            (
                "smoothed_prior_poloidal",
                result.variance_fields.smoothed.poloidal.prior.as_slice(),
            ),
            (
                "smoothed_post_poloidal",
                result
                    .variance_fields
                    .smoothed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_poloidal",
                result.variance_fields.smoothed.poloidal.ratio.as_slice(),
            ),
            (
                "smoothed_trace_prior",
                result.variance_fields.smoothed.trace.prior.as_slice(),
            ),
            (
                "smoothed_trace_post",
                result.variance_fields.smoothed.trace.posterior.as_slice(),
            ),
            (
                "smoothed_trace_ratio",
                result.variance_fields.smoothed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    write_top_cell_scalar_vtk_fields(
        out_dir.join("circulation_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_circulation",
                result.variance_fields.circulation.prior.as_slice(),
            ),
            (
                "post_circulation",
                result.variance_fields.circulation.posterior.as_slice(),
            ),
            (
                "ratio_circulation",
                result.variance_fields.circulation.ratio.as_slice(),
            ),
        ],
    )?;
    Ok(())
}

fn ambient_variance_vectors(
    surface: &Torus1FormAmbientVarianceFields,
    use_prior: bool,
) -> Vec<[f64; 3]> {
    let x = if use_prior {
        &surface.x.prior
    } else {
        &surface.x.posterior
    };
    let y = if use_prior {
        &surface.y.prior
    } else {
        &surface.y.posterior
    };
    let z = if use_prior {
        &surface.z.prior
    } else {
        &surface.z.posterior
    };
    (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect()
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
        .collect()
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
