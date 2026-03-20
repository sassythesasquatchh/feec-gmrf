use crate::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    infer_torus_radii,
};
use crate::matern_1form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use crate::util::convert_whittle_params_to_matern;
use crate::vtk::{
    write_1cochain_vtk_fields, write_1form_vector_proxy_vtk_fields,
    write_top_cell_scalar_vtk_fields,
};
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use ddf::whitney::lsf::WhitneyLsf;
use exterior::field::{EmbeddedDiffFormClosure, ExteriorField};
use faer::linalg::solvers::Solve;
use faer::Side;
use formoniq::io::{sample_1form_cell_vectors, write_top_cell_vtk_fields};
use gmrf_core::observation::{
    apply_gaussian_observations, ht_weighted_observations, observation_selector,
};
use gmrf_core::types::{
    DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{Gmrf, GmrfError};
use manifold::{
    geometry::coord::{
        mesh::MeshCoords,
        simplex::{barycenter_local, SimplexHandleExt},
    },
    topology::complex::Complex,
};
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;

const EPS: f64 = 1e-12;
const EXACT_VARIANCE_TOLERANCE: f64 = 1e-10;
const HARMONIC_TOROIDAL_SCALE: f64 = 0.75;
const HARMONIC_POLOIDAL_SCALE: f64 = -0.50;
const TOROIDAL_ALIGNMENT_MIN: f64 = 0.8;
const POLOIDAL_ALIGNMENT_MAX: f64 = 0.2;
const FAR_RADIUS_SCALE: f64 = 2.0;
const DEFAULT_NUM_RBMC_PROBES: usize = 256;
const DEFAULT_RBMC_BATCH_COUNT: usize = 8;
const DEFAULT_RNG_SEED: u64 = 13;
const OBSERVATION_PROFILE_DISTANCE_SCALES: &[f64] =
    &[0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00];
const SMOOTHING_BANDWIDTH_SCALE: f64 = 0.5;
const SMOOTHING_CUTOFF_SCALE: f64 = 1.0;
const VARIANCE_OBJECT_EDGE_ALL: &str = "edge_all";
const VARIANCE_OBJECT_EDGE_COMPATIBLE: &str = "edge_compatible";
const VARIANCE_OBJECT_EDGE_TRANSVERSE: &str = "edge_transverse";
const VARIANCE_OBJECT_COMPONENT_MATCHED: &str = "component_matched";
const VARIANCE_OBJECT_COMPONENT_ORTHOGONAL: &str = "component_orthogonal";
const VARIANCE_OBJECT_COMPONENT_TRACE: &str = "component_trace";
const VARIANCE_OBJECT_SMOOTHED_MATCHED: &str = "smoothed_matched";
const VARIANCE_OBJECT_SMOOTHED_ORTHOGONAL: &str = "smoothed_orthogonal";
const VARIANCE_OBJECT_SMOOTHED_TRACE: &str = "smoothed_trace";
const VARIANCE_OBJECT_CIRCULATION: &str = "circulation";

const DEFAULT_OBSERVATION_TARGETS: [Torus1FormObservationTarget; 6] = [
    Torus1FormObservationTarget {
        theta: -0.50,
        phi: -2.75,
        direction: ObservationDirection::Toroidal,
    },
    Torus1FormObservationTarget {
        theta: 1.50,
        phi: -1.75,
        direction: ObservationDirection::Toroidal,
    },
    Torus1FormObservationTarget {
        theta: 2.00,
        phi: 1.75,
        direction: ObservationDirection::Toroidal,
    },
    Torus1FormObservationTarget {
        theta: -3.00,
        phi: -0.75,
        direction: ObservationDirection::Poloidal,
    },
    Torus1FormObservationTarget {
        theta: 0.75,
        phi: -0.75,
        direction: ObservationDirection::Poloidal,
    },
    Torus1FormObservationTarget {
        theta: 1.25,
        phi: -1.75,
        direction: ObservationDirection::Poloidal,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationDirection {
    Toroidal,
    Poloidal,
}

impl ObservationDirection {
    fn matches_alignment(self, toroidal_alignment_sq: f64) -> bool {
        match self {
            Self::Toroidal => toroidal_alignment_sq >= TOROIDAL_ALIGNMENT_MIN,
            Self::Poloidal => toroidal_alignment_sq <= POLOIDAL_ALIGNMENT_MAX,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Toroidal => "toroidal",
            Self::Poloidal => "poloidal",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SurfaceVectorVarianceMode {
    #[default]
    Exact,
    Rbmc,
    RbmcClipped,
}

impl SurfaceVectorVarianceMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Rbmc => "rbmc",
            Self::RbmcClipped => "rbmc-clipped",
        }
    }
}

impl std::str::FromStr for SurfaceVectorVarianceMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "exact" => Ok(Self::Exact),
            "rbmc" => Ok(Self::Rbmc),
            "rbmc-clipped" => Ok(Self::RbmcClipped),
            _ => Err(format!(
                "invalid surface-vector variance mode `{value}`; expected one of: exact, rbmc, rbmc-clipped"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Torus1FormObservationTarget {
    pub theta: f64,
    pub phi: f64,
    pub direction: ObservationDirection,
}

#[derive(Debug, Clone)]
pub struct Torus1FormConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
    pub neighbourhood_radius_scale: f64,
    pub observation_targets: Vec<Torus1FormObservationTarget>,
}

impl Default for Torus1FormConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../meshes/torus_shell.msh"),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            surface_vector_variance_mode: SurfaceVectorVarianceMode::Exact,
            num_rbmc_probes: DEFAULT_NUM_RBMC_PROBES,
            rbmc_batch_count: DEFAULT_RBMC_BATCH_COUNT,
            rng_seed: DEFAULT_RNG_SEED,
            neighbourhood_radius_scale: 0.75,
            observation_targets: DEFAULT_OBSERVATION_TARGETS.to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Torus1FormSelectedObservation {
    pub observation_index: usize,
    pub edge_index: usize,
    pub target_theta: f64,
    pub target_phi: f64,
    pub direction: ObservationDirection,
    pub edge_theta: f64,
    pub edge_phi: f64,
    pub toroidal_alignment_sq: f64,
    pub selection_distance: f64,
    pub used_fallback: bool,
}

#[derive(Debug, Clone)]
pub struct Torus1FormObservationSummary {
    pub observation_index: usize,
    pub edge_index: usize,
    pub direction: ObservationDirection,
    pub used_fallback: bool,
    pub target_theta: f64,
    pub target_phi: f64,
    pub edge_theta: f64,
    pub edge_phi: f64,
    pub observation_value: f64,
    pub posterior_mean_at_observation: f64,
    pub abs_error_at_observation: f64,
    pub prior_variance_at_observation: f64,
    pub posterior_variance_at_observation: f64,
    pub harmonic_free_truth_at_observation: f64,
    pub harmonic_free_posterior_mean_at_observation: f64,
    pub harmonic_free_abs_error_at_observation: f64,
    pub harmonic_free_prior_variance_at_observation: f64,
    pub harmonic_free_posterior_variance_at_observation: f64,
}

#[derive(Debug, Clone)]
pub struct RegionSummary {
    pub count: usize,
    pub mean_abs_error: f64,
    pub harmonic_free_mean_abs_error: f64,
    pub prior_variance_mean: f64,
    pub posterior_variance_mean: f64,
    pub variance_reduction_mean: f64,
    pub variance_ratio_mean: f64,
    pub harmonic_free_prior_variance_mean: f64,
    pub harmonic_free_posterior_variance_mean: f64,
    pub harmonic_free_variance_reduction_mean: f64,
    pub harmonic_free_variance_ratio_mean: f64,
}

#[derive(Debug, Clone)]
pub struct ObservedSummary {
    pub count: usize,
    pub max_abs_error: f64,
    pub mean_abs_error: f64,
    pub harmonic_free_mean_abs_error: f64,
    pub prior_variance_mean: f64,
    pub posterior_variance_mean: f64,
    pub variance_reduction_mean: f64,
    pub variance_ratio_mean: f64,
    pub harmonic_free_prior_variance_mean: f64,
    pub harmonic_free_posterior_variance_mean: f64,
    pub harmonic_free_variance_reduction_mean: f64,
    pub harmonic_free_variance_ratio_mean: f64,
}

#[derive(Debug, Clone)]
pub struct Torus1FormBranchSummary {
    pub observed: ObservedSummary,
    pub near: RegionSummary,
    pub far: RegionSummary,
}

#[derive(Debug, Clone)]
pub struct Torus1FormVarianceFieldSet {
    pub prior: FeecVector,
    pub posterior: FeecVector,
    pub ratio: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Torus1FormVarianceComponentFields {
    pub toroidal: Torus1FormVarianceFieldSet,
    pub poloidal: Torus1FormVarianceFieldSet,
    pub trace: Torus1FormVarianceFieldSet,
}

#[derive(Debug, Clone)]
pub struct Torus1FormAmbientVarianceFields {
    pub x: Torus1FormVarianceFieldSet,
    pub y: Torus1FormVarianceFieldSet,
    pub z: Torus1FormVarianceFieldSet,
    pub trace: Torus1FormVarianceFieldSet,
}

#[derive(Debug, Clone)]
pub struct Torus1FormVariancePatternSummaryRow {
    pub object: &'static str,
    pub observation_count: usize,
    pub very_local_ratio: f64,
    pub local_ratio: f64,
    pub range_ratio: f64,
    pub far_ratio: f64,
    pub localization_auc: f64,
    pub monotonicity_score: f64,
    pub very_local_orientation_contrast: f64,
    pub local_orientation_contrast: f64,
}

#[derive(Debug, Clone)]
pub struct Torus1FormVariancePatternShellProfileRow {
    pub object: &'static str,
    pub observation_index: usize,
    pub observation_direction: ObservationDirection,
    pub distance_min_scale: f64,
    pub distance_max_scale: f64,
    pub distance_min: f64,
    pub distance_max: f64,
    pub shell_mid_scale: f64,
    pub count: usize,
    pub mean_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct Torus1FormVariancePatternReport {
    pub shell_distance_scales: Vec<f64>,
    pub smoothing_bandwidth: f64,
    pub smoothing_cutoff: f64,
    pub reconstructed: Torus1FormVarianceComponentFields,
    pub surface_vector: Torus1FormAmbientVarianceFields,
    pub smoothed: Torus1FormVarianceComponentFields,
    pub circulation: Torus1FormVarianceFieldSet,
    pub summary_rows: Vec<Torus1FormVariancePatternSummaryRow>,
    pub shell_profile_rows: Vec<Torus1FormVariancePatternShellProfileRow>,
}

#[derive(Debug, Clone)]
pub struct Torus1FormBranchResult {
    pub name: &'static str,
    pub truth: FeecVector,
    pub posterior_mean: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub harmonic_free_truth: FeecVector,
    pub harmonic_free_posterior_mean: FeecVector,
    pub harmonic_free_absolute_mean_error: FeecVector,
    pub harmonic_free_prior_variance: FeecVector,
    pub harmonic_free_posterior_variance: FeecVector,
    pub harmonic_free_variance_reduction: FeecVector,
    pub observed_mask: FeecVector,
    pub nearest_observation_value: FeecVector,
    pub nearest_observation_distance: FeecVector,
    pub observation_values: Vec<f64>,
    pub observation_summaries: Vec<Torus1FormObservationSummary>,
    pub harmonic_coefficients_truth: [f64; 2],
    pub harmonic_coefficients_posterior_mean: [f64; 2],
    pub summary: Torus1FormBranchSummary,
    pub variance_pattern: Torus1FormVariancePatternReport,
}

pub struct Torus1FormConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub edge_theta: FeecVector,
    pub edge_phi: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
    pub observation_targets: Vec<Torus1FormObservationTarget>,
    pub selected_observations: Vec<Torus1FormSelectedObservation>,
    pub observation_indices: Vec<usize>,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub surface_vector_variance_mode: SurfaceVectorVarianceMode,
    pub num_rbmc_probes: usize,
    pub rbmc_batch_count: usize,
    pub rng_seed: u64,
    pub effective_range: f64,
    pub neighbourhood_radius: f64,
    pub far_radius: f64,
    pub harmonic_free_constrained: Torus1FormBranchResult,
    pub full_unconstrained: Torus1FormBranchResult,
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

#[derive(Clone)]
struct SparseRowLinearOperator {
    ncols: usize,
    rows: Vec<Vec<(usize, f64)>>,
}

struct VariancePatternSharedData {
    major_radius: f64,
    minor_radius: f64,
    cell_theta: FeecVector,
    cell_phi: FeecVector,
    smoothing_bandwidth: f64,
    smoothing_cutoff: f64,
    reconstructed_prior: Torus1FormVarianceComponentEstimates,
    reconstructed_posterior: Torus1FormVarianceComponentEstimates,
    surface_vector_prior: Torus1FormAmbientVarianceEstimates,
    surface_vector_posterior: Torus1FormAmbientVarianceEstimates,
    smoothed_prior: Torus1FormVarianceComponentEstimates,
    smoothed_posterior: Torus1FormVarianceComponentEstimates,
    circulation_prior: RbmcVarianceEstimates,
    circulation_posterior: RbmcVarianceEstimates,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservationOrientationRelation {
    Compatible,
    Oblique,
    Transverse,
}

impl ObservationOrientationRelation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Compatible => "compatible",
            Self::Oblique => "oblique",
            Self::Transverse => "transverse",
        }
    }
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

    #[cfg(test)]
    fn identity(size: usize) -> Self {
        Self {
            ncols: size,
            rows: (0..size).map(|i| vec![(i, 1.0)]).collect(),
        }
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

pub fn run_torus_1form_conditioning(
    config: &Torus1FormConditioningConfig,
) -> Result<Torus1FormConditioningResult, Box<dyn Error>> {
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
    let harmonic_basis =
        build_analytic_torus_harmonic_basis(&topology, &coords, &metric).map_err(invalid_data)?;
    let harmonic_basis_orthonormal =
        mass_orthonormalize_harmonic_basis(&harmonic_basis, &hodge.mass_u).map_err(invalid_data)?;
    let harmonic_constraints =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .map_err(invalid_data)?;
    let constraint_rhs = GmrfVector::zeros(harmonic_constraints.nrows());

    let seed = build_local_seed_cochain(
        &topology,
        &coords,
        geometry.major_radius,
        geometry.minor_radius,
    );
    let truth_harmonic_free =
        remove_harmonic_content(&seed.coeffs, &harmonic_basis_orthonormal, &hodge.mass_u);
    let harmonic_toroidal = harmonic_basis_orthonormal.column(0).into_owned();
    let harmonic_poloidal = harmonic_basis_orthonormal.column(1).into_owned();
    let truth_full = &truth_harmonic_free
        + harmonic_toroidal.scale(HARMONIC_TOROIDAL_SCALE)
        + harmonic_poloidal.scale(HARMONIC_POLOIDAL_SCALE);

    let selected_observations =
        select_observation_edges(&geometry, &config.observation_targets).map_err(invalid_data)?;
    let observation_indices = selected_observations
        .iter()
        .map(|selected| selected.edge_index)
        .collect::<Vec<_>>();
    let observation_matrix = observation_selector(hodge.mass_u.nrows(), &observation_indices);
    let observed_mask = build_observed_mask(hodge.mass_u.nrows(), &observation_indices);
    let nearest_observation_slots =
        build_nearest_observation_slots(&geometry, &observation_indices);
    let nearest_observation_distance = build_nearest_observation_distance_field(
        &geometry,
        &observation_indices,
        &nearest_observation_slots,
    );
    let edge_theta = FeecVector::from_vec(geometry.theta.clone());
    let edge_phi = FeecVector::from_vec(geometry.phi.clone());
    let toroidal_alignment_sq = FeecVector::from_vec(geometry.toroidal_alignment_sq.clone());

    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let zero_observations = GmrfVector::zeros(observation_indices.len());
    let (posterior_precision, _) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &zero_observations,
        None,
        config.noise_variance,
    );
    let (_nu, _variance, effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);
    let neighbourhood_radius = config.neighbourhood_radius_scale * effective_range;
    let far_radius = FAR_RADIUS_SCALE * effective_range;
    let smoothing_bandwidth = SMOOTHING_BANDWIDTH_SCALE * effective_range;
    let smoothing_cutoff = SMOOTHING_CUTOFF_SCALE * effective_range;

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
    let surface_vector_stacked_operator = SparseRowLinearOperator::stack(&[
        &surface_x_operator,
        &surface_y_operator,
        &surface_z_operator,
    ])
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

    let mut prior_workspace = build_rbmc_workspace(&q_prior, &harmonic_constraints)?;
    let prior_latent_variances =
        exact_latent_variances(&mut prior_workspace, &harmonic_constraints)?;
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

    let mut posterior_workspace =
        build_rbmc_workspace(&posterior_precision, &harmonic_constraints)?;
    let posterior_latent_variances =
        exact_latent_variances(&mut posterior_workspace, &harmonic_constraints)?;
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
    let surface_vector_posterior_estimates =
        if config.surface_vector_variance_mode == SurfaceVectorVarianceMode::RbmcClipped {
            clip_rbmc_posterior_to_prior(
                &surface_vector_prior,
                &split_ambient_estimates(
                    surface_vector_posterior_estimates,
                    cell_geometry.theta.len(),
                )
                .map_err(invalid_data)?,
            )
        } else {
            split_ambient_estimates(
                surface_vector_posterior_estimates,
                cell_geometry.theta.len(),
            )
            .map_err(invalid_data)?
        };
    let surface_vector_posterior = surface_vector_posterior_estimates;
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

    let variance_pattern_shared = VariancePatternSharedData {
        major_radius: geometry.major_radius,
        minor_radius: geometry.minor_radius,
        cell_theta: FeecVector::from_vec(cell_geometry.theta),
        cell_phi: FeecVector::from_vec(cell_geometry.phi),
        smoothing_bandwidth,
        smoothing_cutoff,
        reconstructed_prior,
        reconstructed_posterior,
        surface_vector_prior,
        surface_vector_posterior,
        smoothed_prior,
        smoothed_posterior,
        circulation_prior,
        circulation_posterior,
    };

    let harmonic_free_constrained = build_branch_result(
        "harmonic_free_constrained",
        &truth_harmonic_free,
        &posterior_precision,
        &observation_matrix,
        config.noise_variance,
        &harmonic_constraints,
        &constraint_rhs,
        &prior_latent_variances,
        &posterior_latent_variances,
        true,
        &edge_theta,
        &edge_phi,
        &toroidal_alignment_sq,
        &variance_pattern_shared,
        &harmonic_basis_orthonormal,
        &hodge.mass_u,
        &selected_observations,
        &nearest_observation_slots,
        &nearest_observation_distance,
        &observed_mask,
        effective_range,
        neighbourhood_radius,
        far_radius,
    )?;

    let full_unconstrained = build_branch_result(
        "full_unconstrained",
        &truth_full,
        &posterior_precision,
        &observation_matrix,
        config.noise_variance,
        &harmonic_constraints,
        &constraint_rhs,
        &prior_latent_variances,
        &posterior_latent_variances,
        false,
        &edge_theta,
        &edge_phi,
        &toroidal_alignment_sq,
        &variance_pattern_shared,
        &harmonic_basis_orthonormal,
        &hodge.mass_u,
        &selected_observations,
        &nearest_observation_slots,
        &nearest_observation_distance,
        &observed_mask,
        effective_range,
        neighbourhood_radius,
        far_radius,
    )?;

    Ok(Torus1FormConditioningResult {
        topology,
        coords,
        edge_theta,
        edge_phi,
        toroidal_alignment_sq,
        observation_targets: config.observation_targets.clone(),
        selected_observations,
        observation_indices,
        major_radius: geometry.major_radius,
        minor_radius: geometry.minor_radius,
        surface_vector_variance_mode: config.surface_vector_variance_mode,
        num_rbmc_probes: config.num_rbmc_probes,
        rbmc_batch_count: config.rbmc_batch_count,
        rng_seed: config.rng_seed,
        effective_range,
        neighbourhood_radius,
        far_radius,
        harmonic_free_constrained,
        full_unconstrained,
    })
}

pub fn write_torus_1form_conditioning_outputs(
    result: &Torus1FormConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    write_selected_observations_csv(result, out_dir)?;
    write_overall_summary(result, out_dir)?;
    write_branch_outputs(result, &result.harmonic_free_constrained, out_dir)?;
    write_branch_outputs(result, &result.full_unconstrained, out_dir)?;

    Ok(())
}

fn build_branch_result(
    name: &'static str,
    truth: &FeecVector,
    posterior_precision: &GmrfSparseMatrix,
    observation_matrix: &GmrfSparseMatrix,
    noise_variance: f64,
    harmonic_constraints: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
    prior_variances: &RbmcVarianceEstimates,
    posterior_variances: &RbmcVarianceEstimates,
    enforce_harmonic_constraints: bool,
    edge_theta: &FeecVector,
    edge_phi: &FeecVector,
    toroidal_alignment_sq: &FeecVector,
    variance_pattern_shared: &VariancePatternSharedData,
    harmonic_basis_orthonormal: &FeecMatrix,
    mass_u: &FeecCsr,
    selected_observations: &[Torus1FormSelectedObservation],
    nearest_observation_slots: &[usize],
    nearest_observation_distance: &FeecVector,
    observed_mask: &FeecVector,
    effective_range: f64,
    neighbourhood_radius: f64,
    far_radius: f64,
) -> Result<Torus1FormBranchResult, Box<dyn Error>> {
    let truth_gmrf = feec_vec_to_gmrf(truth);
    let observation_values = (&*observation_matrix * &truth_gmrf)
        .iter()
        .copied()
        .collect::<Vec<_>>();
    let information = ht_weighted_observations(
        observation_matrix,
        &GmrfVector::from_vec(observation_values.clone()),
        1.0 / noise_variance,
    );

    let mut posterior =
        Gmrf::from_information_and_precision(information, posterior_precision.clone())?;
    let posterior_mean = if enforce_harmonic_constraints {
        constrained_mean(&mut posterior, harmonic_constraints, constraint_rhs)?
    } else {
        posterior.mean().clone()
    };
    let posterior_variance = if enforce_harmonic_constraints {
        gmrf_vec_to_feec(&posterior_variances.harmonic_free)
    } else {
        gmrf_vec_to_feec(&posterior_variances.unconstrained)
    };
    let prior_variance = if enforce_harmonic_constraints {
        gmrf_vec_to_feec(&prior_variances.harmonic_free)
    } else {
        gmrf_vec_to_feec(&prior_variances.unconstrained)
    };

    let posterior_mean = gmrf_vec_to_feec(&posterior_mean);
    let absolute_mean_error = absolute_difference(&posterior_mean, truth);
    let variance_reduction = &prior_variance - &posterior_variance;

    let harmonic_free_truth = remove_harmonic_content(truth, harmonic_basis_orthonormal, mass_u);
    let harmonic_free_posterior_mean =
        remove_harmonic_content(&posterior_mean, harmonic_basis_orthonormal, mass_u);
    let harmonic_free_absolute_mean_error =
        absolute_difference(&harmonic_free_posterior_mean, &harmonic_free_truth);
    let harmonic_free_prior_variance = gmrf_vec_to_feec(&prior_variances.harmonic_free);
    let harmonic_free_posterior_variance = gmrf_vec_to_feec(&posterior_variances.harmonic_free);
    let harmonic_free_variance_reduction =
        &harmonic_free_prior_variance - &harmonic_free_posterior_variance;

    let nearest_observation_value =
        build_nearest_observation_value_field(nearest_observation_slots, &observation_values);
    let harmonic_coefficients_truth =
        harmonic_coefficients(truth, harmonic_basis_orthonormal, mass_u).map_err(invalid_data)?;
    let harmonic_coefficients_posterior_mean =
        harmonic_coefficients(&posterior_mean, harmonic_basis_orthonormal, mass_u)
            .map_err(invalid_data)?;

    let observation_summaries = build_observation_summaries(
        selected_observations,
        &observation_values,
        &posterior_mean,
        &absolute_mean_error,
        &prior_variance,
        &posterior_variance,
        &harmonic_free_truth,
        &harmonic_free_posterior_mean,
        &harmonic_free_absolute_mean_error,
        &harmonic_free_prior_variance,
        &harmonic_free_posterior_variance,
    );
    let summary = build_branch_summary(
        selected_observations,
        nearest_observation_distance,
        &absolute_mean_error,
        &harmonic_free_absolute_mean_error,
        &prior_variance,
        &posterior_variance,
        &harmonic_free_prior_variance,
        &harmonic_free_posterior_variance,
        neighbourhood_radius,
        far_radius,
    )
    .map_err(invalid_data)?;
    let variance_pattern = build_variance_pattern_report(
        enforce_harmonic_constraints,
        edge_theta,
        edge_phi,
        toroidal_alignment_sq,
        selected_observations,
        variance_pattern_shared,
        &prior_variance,
        &posterior_variance,
        effective_range,
    )
    .map_err(invalid_data)?;

    Ok(Torus1FormBranchResult {
        name,
        truth: truth.clone(),
        posterior_mean,
        absolute_mean_error,
        prior_variance,
        posterior_variance,
        variance_reduction,
        harmonic_free_truth,
        harmonic_free_posterior_mean,
        harmonic_free_absolute_mean_error,
        harmonic_free_prior_variance,
        harmonic_free_posterior_variance,
        harmonic_free_variance_reduction,
        observed_mask: observed_mask.clone(),
        nearest_observation_value,
        nearest_observation_distance: nearest_observation_distance.clone(),
        observation_values,
        observation_summaries,
        harmonic_coefficients_truth,
        harmonic_coefficients_posterior_mean,
        summary,
        variance_pattern,
    })
}

fn validate_config(config: &Torus1FormConditioningConfig) -> Result<(), Box<dyn Error>> {
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
    if !config.neighbourhood_radius_scale.is_finite() || config.neighbourhood_radius_scale <= 0.0 {
        return Err(invalid_input("neighbourhood_radius_scale must be finite and positive").into());
    }
    if config.observation_targets.is_empty() {
        return Err(invalid_input("at least one observation target is required").into());
    }
    Ok(())
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

#[cfg(test)]
fn estimate_latent_rbmc_variances(
    workspace: &mut RbmcWorkspace,
    dimension: usize,
    num_rbmc_probes: usize,
    rbmc_batch_count: usize,
    rng_seed: u64,
) -> Result<RbmcVarianceEstimates, Box<dyn Error>> {
    let operator = SparseRowLinearOperator::identity(dimension);
    estimate_transformed_rbmc_variances(
        workspace,
        &operator,
        num_rbmc_probes,
        rbmc_batch_count,
        rng_seed,
    )
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

fn classify_orientation_relation(
    observation_direction: ObservationDirection,
    toroidal_alignment_sq: f64,
) -> ObservationOrientationRelation {
    if observation_direction.matches_alignment(toroidal_alignment_sq) {
        return ObservationOrientationRelation::Compatible;
    }

    let transverse = match observation_direction {
        ObservationDirection::Toroidal => toroidal_alignment_sq <= POLOIDAL_ALIGNMENT_MAX,
        ObservationDirection::Poloidal => toroidal_alignment_sq >= TOROIDAL_ALIGNMENT_MIN,
    };
    if transverse {
        ObservationOrientationRelation::Transverse
    } else {
        ObservationOrientationRelation::Oblique
    }
}

fn orientation_relation_filter_matches(
    filter: Option<ObservationOrientationRelation>,
    relation: ObservationOrientationRelation,
) -> bool {
    match filter {
        Some(expected) => expected == relation,
        None => true,
    }
}

fn summarize_values(values: &[f64]) -> (usize, f64, f64, f64) {
    if values.is_empty() {
        return (0, f64::NAN, f64::NAN, f64::NAN);
    }

    let mut sum = 0.0;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for value in values.iter().copied() {
        sum += value;
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }

    (
        values.len(),
        sum / values.len() as f64,
        min_value,
        max_value,
    )
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

fn build_local_seed_cochain(
    topology: &Complex,
    coords: &MeshCoords,
    major_radius: f64,
    minor_radius: f64,
) -> Cochain {
    let seed = EmbeddedDiffFormClosure::ambient_one_form(
        move |p| {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            let rho = (x * x + y * y).sqrt().max(EPS);
            let theta = z.atan2(rho - major_radius);
            let phi = y.atan2(x);
            let a = 0.7 * (2.0 * phi - theta).cos() + 0.2 * (3.0 * theta).sin();
            let b = -0.5 * (phi + theta).sin() + 0.3 * (2.0 * theta).cos();

            let toroidal = toroidal_covector(x, y);
            let poloidal = poloidal_covector(x, y, z, rho, major_radius, minor_radius);
            FeecVector::from_column_slice(&[
                a * toroidal[0] + b * poloidal[0],
                a * toroidal[1] + b * poloidal[1],
                a * toroidal[2] + b * poloidal[2],
            ])
        },
        coords.dim(),
        topology.dim(),
    );
    cochain_projection(&seed, topology, coords, None)
}

fn toroidal_covector(x: f64, y: f64) -> [f64; 3] {
    let rho2 = (x * x + y * y).max(EPS);
    [-y / rho2, x / rho2, 0.0]
}

fn poloidal_covector(
    x: f64,
    y: f64,
    z: f64,
    rho: f64,
    major_radius: f64,
    minor_radius: f64,
) -> [f64; 3] {
    [
        -z * x / (minor_radius * rho * rho),
        -z * y / (minor_radius * rho * rho),
        (rho - major_radius) / (minor_radius * rho),
    ]
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

fn select_observation_edges(
    geometry: &TorusEdgeGeometry,
    targets: &[Torus1FormObservationTarget],
) -> Result<Vec<Torus1FormSelectedObservation>, String> {
    let mut used = HashSet::with_capacity(targets.len());
    let mut selected = Vec::with_capacity(targets.len());

    for (observation_index, target) in targets.iter().copied().enumerate() {
        let mut best_matching = None::<(usize, f64)>;
        let mut best_fallback = None::<(usize, f64)>;

        for edge_index in 0..geometry.theta.len() {
            if used.contains(&edge_index) {
                continue;
            }

            let distance = intrinsic_torus_distance(
                geometry.major_radius,
                geometry.minor_radius,
                geometry.theta[edge_index],
                geometry.phi[edge_index],
                target.theta,
                target.phi,
            );

            update_best_candidate(&mut best_fallback, edge_index, distance);
            if target
                .direction
                .matches_alignment(geometry.toroidal_alignment_sq[edge_index])
            {
                update_best_candidate(&mut best_matching, edge_index, distance);
            }
        }

        let (edge_index, selection_distance, used_fallback) =
            if let Some((edge_index, selection_distance)) = best_matching {
                (edge_index, selection_distance, false)
            } else if let Some((edge_index, selection_distance)) = best_fallback {
                (edge_index, selection_distance, true)
            } else {
                return Err("failed to find a unique observation edge".to_string());
            };

        used.insert(edge_index);
        selected.push(Torus1FormSelectedObservation {
            observation_index,
            edge_index,
            target_theta: target.theta,
            target_phi: target.phi,
            direction: target.direction,
            edge_theta: geometry.theta[edge_index],
            edge_phi: geometry.phi[edge_index],
            toroidal_alignment_sq: geometry.toroidal_alignment_sq[edge_index],
            selection_distance,
            used_fallback,
        });
    }

    Ok(selected)
}

fn update_best_candidate(best: &mut Option<(usize, f64)>, edge_index: usize, distance: f64) {
    match best {
        Some((_, best_distance)) if distance >= *best_distance => {}
        _ => *best = Some((edge_index, distance)),
    }
}

fn build_nearest_observation_slots(
    geometry: &TorusEdgeGeometry,
    observation_indices: &[usize],
) -> Vec<usize> {
    (0..geometry.theta.len())
        .map(|edge_index| {
            observation_indices
                .iter()
                .enumerate()
                .min_by(|(_, lhs_idx), (_, rhs_idx)| {
                    let lhs_distance = intrinsic_torus_distance(
                        geometry.major_radius,
                        geometry.minor_radius,
                        geometry.theta[edge_index],
                        geometry.phi[edge_index],
                        geometry.theta[**lhs_idx],
                        geometry.phi[**lhs_idx],
                    );
                    let rhs_distance = intrinsic_torus_distance(
                        geometry.major_radius,
                        geometry.minor_radius,
                        geometry.theta[edge_index],
                        geometry.phi[edge_index],
                        geometry.theta[**rhs_idx],
                        geometry.phi[**rhs_idx],
                    );
                    lhs_distance
                        .partial_cmp(&rhs_distance)
                        .expect("intrinsic distances should be finite")
                })
                .map(|(slot, _)| slot)
                .expect("at least one observation edge is required")
        })
        .collect()
}

fn build_nearest_observation_distance_field(
    geometry: &TorusEdgeGeometry,
    observation_indices: &[usize],
    nearest_observation_slots: &[usize],
) -> FeecVector {
    FeecVector::from_iterator(
        geometry.theta.len(),
        (0..geometry.theta.len()).map(|edge_index| {
            let slot = nearest_observation_slots[edge_index];
            let obs_edge = observation_indices[slot];
            intrinsic_torus_distance(
                geometry.major_radius,
                geometry.minor_radius,
                geometry.theta[edge_index],
                geometry.phi[edge_index],
                geometry.theta[obs_edge],
                geometry.phi[obs_edge],
            )
        }),
    )
}

fn build_nearest_observation_value_field(
    nearest_observation_slots: &[usize],
    observation_values: &[f64],
) -> FeecVector {
    FeecVector::from_iterator(
        nearest_observation_slots.len(),
        nearest_observation_slots
            .iter()
            .map(|slot| observation_values[*slot]),
    )
}

fn build_observed_mask(dimension: usize, observation_indices: &[usize]) -> FeecVector {
    let mut mask = FeecVector::zeros(dimension);
    for &idx in observation_indices {
        mask[idx] = 1.0;
    }
    mask
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
    while delta <= -PI {
        delta += 2.0 * PI;
    }
    while delta > PI {
        delta -= 2.0 * PI;
    }
    delta
}

fn absolute_difference(lhs: &FeecVector, rhs: &FeecVector) -> FeecVector {
    FeecVector::from_iterator(lhs.len(), (0..lhs.len()).map(|i| (lhs[i] - rhs[i]).abs()))
}

fn build_observation_summaries(
    selected_observations: &[Torus1FormSelectedObservation],
    observation_values: &[f64],
    posterior_mean: &FeecVector,
    absolute_mean_error: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    harmonic_free_truth: &FeecVector,
    harmonic_free_posterior_mean: &FeecVector,
    harmonic_free_absolute_mean_error: &FeecVector,
    harmonic_free_prior_variance: &FeecVector,
    harmonic_free_posterior_variance: &FeecVector,
) -> Vec<Torus1FormObservationSummary> {
    selected_observations
        .iter()
        .zip(observation_values.iter().copied())
        .map(|(selected, observation_value)| {
            let edge_index = selected.edge_index;
            Torus1FormObservationSummary {
                observation_index: selected.observation_index,
                edge_index,
                direction: selected.direction,
                used_fallback: selected.used_fallback,
                target_theta: selected.target_theta,
                target_phi: selected.target_phi,
                edge_theta: selected.edge_theta,
                edge_phi: selected.edge_phi,
                observation_value,
                posterior_mean_at_observation: posterior_mean[edge_index],
                abs_error_at_observation: absolute_mean_error[edge_index],
                prior_variance_at_observation: prior_variance[edge_index],
                posterior_variance_at_observation: posterior_variance[edge_index],
                harmonic_free_truth_at_observation: harmonic_free_truth[edge_index],
                harmonic_free_posterior_mean_at_observation: harmonic_free_posterior_mean
                    [edge_index],
                harmonic_free_abs_error_at_observation: harmonic_free_absolute_mean_error
                    [edge_index],
                harmonic_free_prior_variance_at_observation: harmonic_free_prior_variance
                    [edge_index],
                harmonic_free_posterior_variance_at_observation: harmonic_free_posterior_variance
                    [edge_index],
            }
        })
        .collect()
}

fn build_branch_summary(
    selected_observations: &[Torus1FormSelectedObservation],
    nearest_observation_distance: &FeecVector,
    absolute_mean_error: &FeecVector,
    harmonic_free_absolute_mean_error: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    harmonic_free_prior_variance: &FeecVector,
    harmonic_free_posterior_variance: &FeecVector,
    neighbourhood_radius: f64,
    far_radius: f64,
) -> Result<Torus1FormBranchSummary, String> {
    let observed_indices = selected_observations
        .iter()
        .map(|selected| selected.edge_index)
        .collect::<Vec<_>>();
    let near_indices = (0..nearest_observation_distance.len())
        .filter(|idx| nearest_observation_distance[*idx] <= neighbourhood_radius)
        .collect::<Vec<_>>();
    let far_indices = (0..nearest_observation_distance.len())
        .filter(|idx| nearest_observation_distance[*idx] > far_radius)
        .collect::<Vec<_>>();

    Ok(Torus1FormBranchSummary {
        observed: summarize_observed(
            &observed_indices,
            absolute_mean_error,
            harmonic_free_absolute_mean_error,
            prior_variance,
            posterior_variance,
            harmonic_free_prior_variance,
            harmonic_free_posterior_variance,
        )?,
        near: summarize_region(
            &near_indices,
            absolute_mean_error,
            harmonic_free_absolute_mean_error,
            prior_variance,
            posterior_variance,
            harmonic_free_prior_variance,
            harmonic_free_posterior_variance,
        )?,
        far: summarize_region(
            &far_indices,
            absolute_mean_error,
            harmonic_free_absolute_mean_error,
            prior_variance,
            posterior_variance,
            harmonic_free_prior_variance,
            harmonic_free_posterior_variance,
        )?,
    })
}

fn summarize_region(
    indices: &[usize],
    absolute_mean_error: &FeecVector,
    harmonic_free_absolute_mean_error: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    harmonic_free_prior_variance: &FeecVector,
    harmonic_free_posterior_variance: &FeecVector,
) -> Result<RegionSummary, String> {
    if indices.is_empty() {
        return Ok(empty_region_summary());
    }

    let count = indices.len() as f64;
    let mut mean_abs_error = 0.0;
    let mut harmonic_free_mean_abs_error = 0.0;
    let mut prior_variance_mean = 0.0;
    let mut posterior_variance_mean = 0.0;
    let mut variance_reduction_mean = 0.0;
    let mut variance_ratio_mean = 0.0;
    let mut harmonic_free_prior_variance_mean = 0.0;
    let mut harmonic_free_posterior_variance_mean = 0.0;
    let mut harmonic_free_variance_reduction_mean = 0.0;
    let mut harmonic_free_variance_ratio_mean = 0.0;

    for &idx in indices {
        mean_abs_error += absolute_mean_error[idx];
        harmonic_free_mean_abs_error += harmonic_free_absolute_mean_error[idx];
        prior_variance_mean += prior_variance[idx];
        posterior_variance_mean += posterior_variance[idx];
        variance_reduction_mean += prior_variance[idx] - posterior_variance[idx];
        variance_ratio_mean += safe_ratio(posterior_variance[idx], prior_variance[idx]);
        harmonic_free_prior_variance_mean += harmonic_free_prior_variance[idx];
        harmonic_free_posterior_variance_mean += harmonic_free_posterior_variance[idx];
        harmonic_free_variance_reduction_mean +=
            harmonic_free_prior_variance[idx] - harmonic_free_posterior_variance[idx];
        harmonic_free_variance_ratio_mean += safe_ratio(
            harmonic_free_posterior_variance[idx],
            harmonic_free_prior_variance[idx],
        );
    }

    Ok(RegionSummary {
        count: indices.len(),
        mean_abs_error: mean_abs_error / count,
        harmonic_free_mean_abs_error: harmonic_free_mean_abs_error / count,
        prior_variance_mean: prior_variance_mean / count,
        posterior_variance_mean: posterior_variance_mean / count,
        variance_reduction_mean: variance_reduction_mean / count,
        variance_ratio_mean: variance_ratio_mean / count,
        harmonic_free_prior_variance_mean: harmonic_free_prior_variance_mean / count,
        harmonic_free_posterior_variance_mean: harmonic_free_posterior_variance_mean / count,
        harmonic_free_variance_reduction_mean: harmonic_free_variance_reduction_mean / count,
        harmonic_free_variance_ratio_mean: harmonic_free_variance_ratio_mean / count,
    })
}

fn empty_region_summary() -> RegionSummary {
    RegionSummary {
        count: 0,
        mean_abs_error: f64::NAN,
        harmonic_free_mean_abs_error: f64::NAN,
        prior_variance_mean: f64::NAN,
        posterior_variance_mean: f64::NAN,
        variance_reduction_mean: f64::NAN,
        variance_ratio_mean: f64::NAN,
        harmonic_free_prior_variance_mean: f64::NAN,
        harmonic_free_posterior_variance_mean: f64::NAN,
        harmonic_free_variance_reduction_mean: f64::NAN,
        harmonic_free_variance_ratio_mean: f64::NAN,
    }
}

fn summarize_observed(
    indices: &[usize],
    absolute_mean_error: &FeecVector,
    harmonic_free_absolute_mean_error: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    harmonic_free_prior_variance: &FeecVector,
    harmonic_free_posterior_variance: &FeecVector,
) -> Result<ObservedSummary, String> {
    let region = summarize_region(
        indices,
        absolute_mean_error,
        harmonic_free_absolute_mean_error,
        prior_variance,
        posterior_variance,
        harmonic_free_prior_variance,
        harmonic_free_posterior_variance,
    )?;
    let max_abs_error = indices
        .iter()
        .map(|idx| absolute_mean_error[*idx])
        .fold(0.0_f64, f64::max);

    Ok(ObservedSummary {
        count: region.count,
        max_abs_error,
        mean_abs_error: region.mean_abs_error,
        harmonic_free_mean_abs_error: region.harmonic_free_mean_abs_error,
        prior_variance_mean: region.prior_variance_mean,
        posterior_variance_mean: region.posterior_variance_mean,
        variance_reduction_mean: region.variance_reduction_mean,
        variance_ratio_mean: region.variance_ratio_mean,
        harmonic_free_prior_variance_mean: region.harmonic_free_prior_variance_mean,
        harmonic_free_posterior_variance_mean: region.harmonic_free_posterior_variance_mean,
        harmonic_free_variance_reduction_mean: region.harmonic_free_variance_reduction_mean,
        harmonic_free_variance_ratio_mean: region.harmonic_free_variance_ratio_mean,
    })
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn build_variance_pattern_report(
    enforce_harmonic_constraints: bool,
    edge_theta: &FeecVector,
    edge_phi: &FeecVector,
    toroidal_alignment_sq: &FeecVector,
    selected_observations: &[Torus1FormSelectedObservation],
    shared: &VariancePatternSharedData,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    effective_range: f64,
) -> Result<Torus1FormVariancePatternReport, String> {
    let reconstructed = build_component_field_set(
        &shared.reconstructed_prior,
        &shared.reconstructed_posterior,
        enforce_harmonic_constraints,
    );
    let surface_vector = build_ambient_field_set(
        &shared.surface_vector_prior,
        &shared.surface_vector_posterior,
        enforce_harmonic_constraints,
    );
    let smoothed = build_component_field_set(
        &shared.smoothed_prior,
        &shared.smoothed_posterior,
        enforce_harmonic_constraints,
    );
    let circulation = build_variance_field_set(
        &shared.circulation_prior,
        &shared.circulation_posterior,
        enforce_harmonic_constraints,
    );

    let edge_all_profiles = build_edge_shell_profile_rows(
        VARIANCE_OBJECT_EDGE_ALL,
        selected_observations,
        edge_theta,
        edge_phi,
        toroidal_alignment_sq,
        prior_variance,
        posterior_variance,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        None,
    );
    let edge_compatible_profiles = build_edge_shell_profile_rows(
        VARIANCE_OBJECT_EDGE_COMPATIBLE,
        selected_observations,
        edge_theta,
        edge_phi,
        toroidal_alignment_sq,
        prior_variance,
        posterior_variance,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        Some(ObservationOrientationRelation::Compatible),
    );
    let edge_transverse_profiles = build_edge_shell_profile_rows(
        VARIANCE_OBJECT_EDGE_TRANSVERSE,
        selected_observations,
        edge_theta,
        edge_phi,
        toroidal_alignment_sq,
        prior_variance,
        posterior_variance,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        Some(ObservationOrientationRelation::Transverse),
    );

    let component_matched_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_COMPONENT_MATCHED,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |selected, cell_index| match selected.direction {
            ObservationDirection::Toroidal => Some((
                reconstructed.toroidal.prior[cell_index],
                reconstructed.toroidal.posterior[cell_index],
            )),
            ObservationDirection::Poloidal => Some((
                reconstructed.poloidal.prior[cell_index],
                reconstructed.poloidal.posterior[cell_index],
            )),
        },
    );
    let component_orthogonal_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_COMPONENT_ORTHOGONAL,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |selected, cell_index| match selected.direction {
            ObservationDirection::Toroidal => Some((
                reconstructed.poloidal.prior[cell_index],
                reconstructed.poloidal.posterior[cell_index],
            )),
            ObservationDirection::Poloidal => Some((
                reconstructed.toroidal.prior[cell_index],
                reconstructed.toroidal.posterior[cell_index],
            )),
        },
    );
    let component_trace_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_COMPONENT_TRACE,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |_, cell_index| {
            Some((
                reconstructed.trace.prior[cell_index],
                reconstructed.trace.posterior[cell_index],
            ))
        },
    );
    let smoothed_matched_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_SMOOTHED_MATCHED,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |selected, cell_index| match selected.direction {
            ObservationDirection::Toroidal => Some((
                smoothed.toroidal.prior[cell_index],
                smoothed.toroidal.posterior[cell_index],
            )),
            ObservationDirection::Poloidal => Some((
                smoothed.poloidal.prior[cell_index],
                smoothed.poloidal.posterior[cell_index],
            )),
        },
    );
    let smoothed_orthogonal_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_SMOOTHED_ORTHOGONAL,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |selected, cell_index| match selected.direction {
            ObservationDirection::Toroidal => Some((
                smoothed.poloidal.prior[cell_index],
                smoothed.poloidal.posterior[cell_index],
            )),
            ObservationDirection::Poloidal => Some((
                smoothed.toroidal.prior[cell_index],
                smoothed.toroidal.posterior[cell_index],
            )),
        },
    );
    let smoothed_trace_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_SMOOTHED_TRACE,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |_, cell_index| {
            Some((
                smoothed.trace.prior[cell_index],
                smoothed.trace.posterior[cell_index],
            ))
        },
    );
    let circulation_profiles = build_domain_shell_profile_rows(
        VARIANCE_OBJECT_CIRCULATION,
        selected_observations,
        &shared.cell_theta,
        &shared.cell_phi,
        shared.major_radius,
        shared.minor_radius,
        effective_range,
        |_, cell_index| {
            Some((
                circulation.prior[cell_index],
                circulation.posterior[cell_index],
            ))
        },
    );

    let shell_profile_rows = [
        edge_all_profiles.clone(),
        edge_compatible_profiles.clone(),
        edge_transverse_profiles.clone(),
        component_matched_profiles.clone(),
        component_orthogonal_profiles.clone(),
        component_trace_profiles.clone(),
        smoothed_matched_profiles.clone(),
        smoothed_orthogonal_profiles.clone(),
        smoothed_trace_profiles.clone(),
        circulation_profiles.clone(),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();

    let summary_rows = vec![
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_EDGE_ALL,
            selected_observations.len(),
            &edge_all_profiles,
            Some((&edge_compatible_profiles, &edge_transverse_profiles)),
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_EDGE_COMPATIBLE,
            selected_observations.len(),
            &edge_compatible_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_EDGE_TRANSVERSE,
            selected_observations.len(),
            &edge_transverse_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_COMPONENT_MATCHED,
            selected_observations.len(),
            &component_matched_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_COMPONENT_ORTHOGONAL,
            selected_observations.len(),
            &component_orthogonal_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_COMPONENT_TRACE,
            selected_observations.len(),
            &component_trace_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_SMOOTHED_MATCHED,
            selected_observations.len(),
            &smoothed_matched_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_SMOOTHED_ORTHOGONAL,
            selected_observations.len(),
            &smoothed_orthogonal_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_SMOOTHED_TRACE,
            selected_observations.len(),
            &smoothed_trace_profiles,
            None,
        ),
        build_variance_pattern_summary_row(
            VARIANCE_OBJECT_CIRCULATION,
            selected_observations.len(),
            &circulation_profiles,
            None,
        ),
    ];

    Ok(Torus1FormVariancePatternReport {
        shell_distance_scales: OBSERVATION_PROFILE_DISTANCE_SCALES.to_vec(),
        smoothing_bandwidth: shared.smoothing_bandwidth,
        smoothing_cutoff: shared.smoothing_cutoff,
        reconstructed,
        surface_vector,
        smoothed,
        circulation,
        summary_rows,
        shell_profile_rows,
    })
}

fn build_variance_field_set(
    prior: &RbmcVarianceEstimates,
    posterior: &RbmcVarianceEstimates,
    constrained: bool,
) -> Torus1FormVarianceFieldSet {
    let prior = gmrf_vec_to_feec(if constrained {
        &prior.harmonic_free
    } else {
        &prior.unconstrained
    });
    let posterior = gmrf_vec_to_feec(if constrained {
        &posterior.harmonic_free
    } else {
        &posterior.unconstrained
    });
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
    constrained: bool,
) -> Torus1FormVarianceComponentFields {
    let toroidal = build_variance_field_set(&prior.toroidal, &posterior.toroidal, constrained);
    let poloidal = build_variance_field_set(&prior.poloidal, &posterior.poloidal, constrained);
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
    constrained: bool,
) -> Torus1FormAmbientVarianceFields {
    let x = build_variance_field_set(&prior.x, &posterior.x, constrained);
    let y = build_variance_field_set(&prior.y, &posterior.y, constrained);
    let z = build_variance_field_set(&prior.z, &posterior.z, constrained);
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

fn build_edge_shell_profile_rows(
    object: &'static str,
    selected_observations: &[Torus1FormSelectedObservation],
    edge_theta: &FeecVector,
    edge_phi: &FeecVector,
    toroidal_alignment_sq: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    major_radius: f64,
    minor_radius: f64,
    effective_range: f64,
    relation_filter: Option<ObservationOrientationRelation>,
) -> Vec<Torus1FormVariancePatternShellProfileRow> {
    build_domain_shell_profile_rows(
        object,
        selected_observations,
        edge_theta,
        edge_phi,
        major_radius,
        minor_radius,
        effective_range,
        |selected, edge_index| {
            if edge_index == selected.edge_index {
                return None;
            }
            let relation = classify_orientation_relation(
                selected.direction,
                toroidal_alignment_sq[edge_index],
            );
            if !orientation_relation_filter_matches(relation_filter, relation) {
                return None;
            }
            Some((prior_variance[edge_index], posterior_variance[edge_index]))
        },
    )
}

fn build_domain_shell_profile_rows<F>(
    object: &'static str,
    selected_observations: &[Torus1FormSelectedObservation],
    domain_theta: &FeecVector,
    domain_phi: &FeecVector,
    major_radius: f64,
    minor_radius: f64,
    effective_range: f64,
    mut value_at: F,
) -> Vec<Torus1FormVariancePatternShellProfileRow>
where
    F: FnMut(&Torus1FormSelectedObservation, usize) -> Option<(f64, f64)>,
{
    let mut rows = Vec::new();
    let mut previous_scale = 0.0;
    for &distance_max_scale in OBSERVATION_PROFILE_DISTANCE_SCALES {
        let distance_min_scale = previous_scale;
        previous_scale = distance_max_scale;
        let distance_min = distance_min_scale * effective_range;
        let distance_max = distance_max_scale * effective_range;
        let shell_mid_scale = 0.5 * (distance_min_scale + distance_max_scale);

        for selected in selected_observations {
            let mut count = 0_usize;
            let mut prior_sum = 0.0;
            let mut posterior_sum = 0.0;
            for item_index in 0..domain_theta.len() {
                let (prior, posterior) = match value_at(selected, item_index) {
                    Some(value) => value,
                    None => continue,
                };
                let distance = intrinsic_torus_distance(
                    major_radius,
                    minor_radius,
                    domain_theta[item_index],
                    domain_phi[item_index],
                    selected.edge_theta,
                    selected.edge_phi,
                );
                if distance <= distance_min || distance > distance_max {
                    continue;
                }
                prior_sum += prior;
                posterior_sum += posterior;
                count += 1;
            }

            rows.push(Torus1FormVariancePatternShellProfileRow {
                object,
                observation_index: selected.observation_index,
                observation_direction: selected.direction,
                distance_min_scale,
                distance_max_scale,
                distance_min,
                distance_max,
                shell_mid_scale,
                count,
                mean_ratio: safe_ratio(posterior_sum, prior_sum),
            });
        }
    }

    rows
}

fn build_variance_pattern_summary_row(
    object: &'static str,
    observation_count: usize,
    profiles: &[Torus1FormVariancePatternShellProfileRow],
    contrast_profiles: Option<(
        &[Torus1FormVariancePatternShellProfileRow],
        &[Torus1FormVariancePatternShellProfileRow],
    )>,
) -> Torus1FormVariancePatternSummaryRow {
    let very_local_ratio = weighted_profile_mean(profiles, |row| row.distance_max_scale <= 0.20);
    let local_ratio = weighted_profile_mean(profiles, |row| row.distance_max_scale <= 0.50);
    let range_ratio = weighted_profile_mean(profiles, |row| row.distance_max_scale <= 1.00);
    let far_ratio = weighted_profile_mean(profiles, |row| {
        row.distance_min_scale >= 1.50 && row.distance_max_scale <= 2.00
    });
    let localization_auc = localization_auc(profiles);
    let monotonicity_score = shell_monotonicity_score(profiles);
    let (very_local_orientation_contrast, local_orientation_contrast) =
        if let Some((compatible_profiles, transverse_profiles)) = contrast_profiles {
            (
                weighted_profile_mean(transverse_profiles, |row| row.distance_max_scale <= 0.20)
                    - weighted_profile_mean(compatible_profiles, |row| {
                        row.distance_max_scale <= 0.20
                    }),
                weighted_profile_mean(transverse_profiles, |row| {
                    row.distance_min_scale >= 0.20 && row.distance_max_scale <= 0.50
                }) - weighted_profile_mean(compatible_profiles, |row| {
                    row.distance_min_scale >= 0.20 && row.distance_max_scale <= 0.50
                }),
            )
        } else {
            (f64::NAN, f64::NAN)
        };

    Torus1FormVariancePatternSummaryRow {
        object,
        observation_count,
        very_local_ratio,
        local_ratio,
        range_ratio,
        far_ratio,
        localization_auc,
        monotonicity_score,
        very_local_orientation_contrast,
        local_orientation_contrast,
    }
}

fn weighted_profile_mean<F>(
    profiles: &[Torus1FormVariancePatternShellProfileRow],
    predicate: F,
) -> f64
where
    F: Fn(&Torus1FormVariancePatternShellProfileRow) -> bool,
{
    let mut sum = 0.0;
    let mut weight = 0.0;
    for row in profiles
        .iter()
        .filter(|row| predicate(row) && row.count > 0)
    {
        sum += row.mean_ratio * row.count as f64;
        weight += row.count as f64;
    }
    if weight <= 0.0 {
        f64::NAN
    } else {
        sum / weight
    }
}

fn localization_auc(profiles: &[Torus1FormVariancePatternShellProfileRow]) -> f64 {
    let mut sum = 0.0;
    let mut weight = 0.0;
    for row in profiles.iter().filter(|row| row.count > 0) {
        let shell_width = row.distance_max_scale - row.distance_min_scale;
        let row_weight = shell_width * row.count as f64;
        sum += row_weight * (1.0 - row.mean_ratio);
        weight += row_weight;
    }
    if weight <= 0.0 {
        f64::NAN
    } else {
        sum / weight
    }
}

fn shell_monotonicity_score(profiles: &[Torus1FormVariancePatternShellProfileRow]) -> f64 {
    let mut midpoints = Vec::new();
    let mut means = Vec::new();
    let mut previous_scale = 0.0;
    for &distance_max_scale in OBSERVATION_PROFILE_DISTANCE_SCALES {
        let distance_min_scale = previous_scale;
        previous_scale = distance_max_scale;
        let shell_rows = profiles.iter().filter(|row| {
            (row.distance_min_scale - distance_min_scale).abs() <= EPS
                && (row.distance_max_scale - distance_max_scale).abs() <= EPS
                && row.count > 0
        });

        let mut weighted_sum = 0.0;
        let mut weight = 0.0;
        for row in shell_rows {
            weighted_sum += row.mean_ratio * row.count as f64;
            weight += row.count as f64;
        }
        if weight <= 0.0 {
            continue;
        }
        midpoints.push(0.5 * (distance_min_scale + distance_max_scale));
        means.push(weighted_sum / weight);
    }

    spearman_rank_correlation(&midpoints, &means)
}

fn spearman_rank_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }
    let xr = average_ranks(xs);
    let yr = average_ranks(ys);
    pearson_correlation(&xr, &yr)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|(_, lhs), (_, rhs)| lhs.partial_cmp(rhs).unwrap());

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0_usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && (indexed[end].1 - indexed[start].1).abs() <= EPS {
            end += 1;
        }
        let avg_rank = 0.5 * ((start + 1) as f64 + end as f64);
        for position in start..end {
            ranks[indexed[position].0] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn pearson_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f64::NAN;
    }

    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= EPS || var_y <= EPS {
        f64::NAN
    } else {
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

fn constrained_mean(
    posterior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
) -> Result<GmrfVector, GmrfError> {
    if constraint_matrix.ncols() != posterior.dimension() {
        return Err(GmrfError::DimensionMismatch(
            "constraint matrix columns must match latent dimension",
        ));
    }
    if constraint_matrix.nrows() != constraint_rhs.len() {
        return Err(GmrfError::DimensionMismatch(
            "constraint rhs length must match constraint matrix rows",
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return Ok(posterior.mean().clone());
    }

    let covariance_times_constraint_t =
        covariance_times_constraint_t(posterior, constraint_matrix)?;
    let predicted_constraints = dense_matvec(constraint_matrix, posterior.mean());
    let mut lagrange_rhs = constraint_rhs - &predicted_constraints;

    let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
    let schur_factor = schur
        .llt(Side::Lower)
        .map_err(|_| GmrfError::SingularConstraintSystem)?;
    schur_factor.solve_in_place(lagrange_rhs.as_col_mut().as_mat_mut());

    let correction = dense_matvec(&covariance_times_constraint_t, &lagrange_rhs);
    Ok(posterior.mean() + correction)
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

fn write_branch_outputs(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let branch_dir = out_dir.join(branch.name);
    fs::create_dir_all(&branch_dir)?;

    let truth = Cochain::new(1, branch.truth.clone());
    let posterior_mean = Cochain::new(1, branch.posterior_mean.clone());
    let absolute_mean_error = Cochain::new(1, branch.absolute_mean_error.clone());
    let prior_variance = Cochain::new(1, branch.prior_variance.clone());
    let posterior_variance = Cochain::new(1, branch.posterior_variance.clone());
    let variance_reduction = Cochain::new(1, branch.variance_reduction.clone());
    let observed_mask = Cochain::new(1, branch.observed_mask.clone());
    let nearest_observation_value = Cochain::new(1, branch.nearest_observation_value.clone());
    let nearest_observation_distance = Cochain::new(1, branch.nearest_observation_distance.clone());
    let harmonic_free_truth = Cochain::new(1, branch.harmonic_free_truth.clone());
    let harmonic_free_posterior_mean = Cochain::new(1, branch.harmonic_free_posterior_mean.clone());
    let harmonic_free_absolute_mean_error =
        Cochain::new(1, branch.harmonic_free_absolute_mean_error.clone());
    let harmonic_free_prior_variance = Cochain::new(1, branch.harmonic_free_prior_variance.clone());
    let harmonic_free_posterior_variance =
        Cochain::new(1, branch.harmonic_free_posterior_variance.clone());
    let harmonic_free_variance_reduction =
        Cochain::new(1, branch.harmonic_free_variance_reduction.clone());
    let edge_theta = Cochain::new(1, result.edge_theta.clone());
    let edge_phi = Cochain::new(1, result.edge_phi.clone());
    let toroidal_alignment_sq = Cochain::new(1, result.toroidal_alignment_sq.clone());

    write_1cochain_vtk_fields(
        branch_dir.join("fields.vtk"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("posterior_mean", &posterior_mean),
            ("absolute_mean_error", &absolute_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
            ("observed_mask", &observed_mask),
            ("nearest_observation_value", &nearest_observation_value),
            (
                "nearest_observation_distance",
                &nearest_observation_distance,
            ),
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
            ("edge_theta", &edge_theta),
            ("edge_phi", &edge_phi),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;

    write_1form_vector_proxy_vtk_fields(
        branch_dir.join("posterior_mean_vector.vtk"),
        &result.coords,
        &result.topology,
        "posterior_mean_vector",
        &posterior_mean,
        &[
            ("truth", &truth),
            ("absolute_mean_error", &absolute_mean_error),
            ("posterior_variance", &posterior_variance),
            ("observed_mask", &observed_mask),
        ],
    )?;
    write_branch_surface_vector_vtk(result, branch, &branch_dir)?;
    write_branch_surface_vector_precision_formula(branch, &branch_dir)?;

    write_branch_edge_csv(result, branch, &branch_dir)?;
    write_branch_observation_csv(branch, &branch_dir)?;
    write_branch_observation_variance_diagnostics(result, branch, &branch_dir)?;
    write_branch_variance_pattern_outputs(result, branch, &branch_dir)?;
    write_branch_summary(branch, result, &branch_dir)?;

    Ok(())
}

fn write_branch_surface_vector_vtk(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let posterior_mean = Cochain::new(1, branch.posterior_mean.clone());
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let posterior_mean_magnitude = vector_magnitudes(&posterior_mean_vectors);
    let surface = &branch.variance_pattern.surface_vector;
    let posterior_variance_vectors = ambient_variance_vectors(surface, false);
    let prior_variance_vectors = ambient_variance_vectors(surface, true);
    let posterior_marginal_std = surface.trace.posterior.map(|value| value.max(0.0).sqrt());

    write_top_cell_vtk_fields(
        branch_dir.join("posterior_mean_surface_vector.vtk"),
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

fn write_branch_surface_vector_precision_formula(
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("surface_vector_precision.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "Surface vector field model for branch '{}'",
        branch.name
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Let x denote the posterior 1-cochain random variable."
    )?;
    writeln!(
        writer,
        "Let A denote the barycenter reconstruction operator that maps edge cochains to"
    )?;
    writeln!(
        writer,
        "stacked cellwise Euclidean vectors, so v = A x with v in R^(3 N_cells)."
    )?;
    writeln!(writer)?;
    writeln!(writer, "The observation-conditioned cochain precision is")?;
    writeln!(writer, "  Q_post = Q_prior + (1 / sigma_obs^2) H^T H,")?;
    writeln!(
        writer,
        "where H is the edge-observation selector matrix and sigma_obs^2 is the"
    )?;
    writeln!(writer, "observation noise variance.")?;
    writeln!(writer)?;
    if branch.name == "full_unconstrained" {
        writeln!(writer, "For this unconstrained branch,")?;
        writeln!(writer, "  Sigma_x = Q_post^(-1),")?;
    } else {
        writeln!(
            writer,
            "For this harmonic-free constrained branch, with constraint matrix C,"
        )?;
        writeln!(
            writer,
            "  Sigma_x = Q_post^(-1) - Q_post^(-1) C^T (C Q_post^(-1) C^T)^(-1) C Q_post^(-1),"
        )?;
        writeln!(
            writer,
            "which is the covariance of the Gaussian conditioned on C x = 0."
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "The pushed-forward surface-vector covariance is")?;
    writeln!(writer, "  Sigma_v = A Sigma_x A^T.")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "The corresponding surface-vector precision is therefore"
    )?;
    writeln!(writer, "  Q_v = (A Sigma_x A^T)^+,")?;
    writeln!(
        writer,
        "where ^+ denotes the Moore-Penrose pseudoinverse, since A may introduce"
    )?;
    writeln!(
        writer,
        "linear dependencies between reconstructed cell vectors."
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "In the current implementation Q_v is not assembled explicitly."
    )?;
    writeln!(
        writer,
        "Instead, RBMC is applied in cochain space and pushed through A to estimate"
    )?;
    writeln!(
        writer,
        "the diagonal 3x3 cell blocks of Sigma_v needed for visualization."
    )?;
    writeln!(writer)?;
    writeln!(
        writer,
        "The VTK outputs store only diagonal summaries of Sigma_v:"
    )?;
    writeln!(
        writer,
        "  directional_variance_i = [Sigma_v(ii,xx), Sigma_v(ii,yy), Sigma_v(ii,zz)],"
    )?;
    writeln!(
        writer,
        "  marginal_variance_i = trace(Sigma_v,i) = Sigma_v(ii,xx) + Sigma_v(ii,yy) + Sigma_v(ii,zz),"
    )?;
    writeln!(
        writer,
        "where Sigma_v,i is the 3x3 covariance block for cell i."
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

fn write_selected_observations_csv(
    result: &Torus1FormConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("selected_observations.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "observation_index,edge_index,direction,target_theta,target_phi,edge_theta,edge_phi,toroidal_alignment_sq,selection_distance,used_fallback"
    )?;
    for selected in &result.selected_observations {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
            selected.observation_index,
            selected.edge_index,
            selected.direction.as_str(),
            selected.target_theta,
            selected.target_phi,
            selected.edge_theta,
            selected.edge_phi,
            selected.toroidal_alignment_sq,
            selected.selection_distance,
            selected.used_fallback
        )?;
    }
    Ok(())
}

fn write_overall_summary(
    result: &Torus1FormConditioningResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(out_dir.join("summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Torus 1-form Matérn conditioning")?;
    writeln!(writer, "major_radius={}", result.major_radius)?;
    writeln!(writer, "minor_radius={}", result.minor_radius)?;
    writeln!(writer, "num_rbmc_probes={}", result.num_rbmc_probes)?;
    writeln!(writer, "rbmc_batch_count={}", result.rbmc_batch_count)?;
    writeln!(writer, "rng_seed={}", result.rng_seed)?;
    writeln!(writer, "latent_edge_variances=exact_sparse_inverse_diag")?;
    writeln!(
        writer,
        "surface_vector_variance_mode={}",
        result.surface_vector_variance_mode.as_str()
    )?;
    writeln!(writer, "transformed_variances=rbmc")?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "neighbourhood_radius={}",
        result.neighbourhood_radius
    )?;
    writeln!(writer, "far_radius={}", result.far_radius)?;
    writeln!(
        writer,
        "observation_count={}",
        result.selected_observations.len()
    )?;
    Ok(())
}

fn write_branch_edge_csv(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("edge_fields.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "edge_index,theta,phi,toroidal_alignment_sq,truth,posterior_mean,absolute_mean_error,prior_variance,posterior_variance,variance_reduction,observed_mask,nearest_observation_value,nearest_observation_distance,harmonic_free_truth,harmonic_free_posterior_mean,harmonic_free_absolute_mean_error,harmonic_free_prior_variance,harmonic_free_posterior_variance,harmonic_free_variance_reduction"
    )?;

    for edge_index in 0..branch.truth.len() {
        writeln!(
            writer,
            "{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            edge_index,
            result.edge_theta[edge_index],
            result.edge_phi[edge_index],
            result.toroidal_alignment_sq[edge_index],
            branch.truth[edge_index],
            branch.posterior_mean[edge_index],
            branch.absolute_mean_error[edge_index],
            branch.prior_variance[edge_index],
            branch.posterior_variance[edge_index],
            branch.variance_reduction[edge_index],
            branch.observed_mask[edge_index],
            branch.nearest_observation_value[edge_index],
            branch.nearest_observation_distance[edge_index],
            branch.harmonic_free_truth[edge_index],
            branch.harmonic_free_posterior_mean[edge_index],
            branch.harmonic_free_absolute_mean_error[edge_index],
            branch.harmonic_free_prior_variance[edge_index],
            branch.harmonic_free_posterior_variance[edge_index],
            branch.harmonic_free_variance_reduction[edge_index],
        )?;
    }

    Ok(())
}

fn write_branch_observation_csv(
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("observations.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "observation_index,edge_index,direction,target_theta,target_phi,edge_theta,edge_phi,used_fallback,observation_value,posterior_mean_at_observation,abs_error_at_observation,prior_variance_at_observation,posterior_variance_at_observation,harmonic_free_truth_at_observation,harmonic_free_posterior_mean_at_observation,harmonic_free_abs_error_at_observation,harmonic_free_prior_variance_at_observation,harmonic_free_posterior_variance_at_observation"
    )?;
    for summary in &branch.observation_summaries {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            summary.observation_index,
            summary.edge_index,
            summary.direction.as_str(),
            summary.target_theta,
            summary.target_phi,
            summary.edge_theta,
            summary.edge_phi,
            summary.used_fallback,
            summary.observation_value,
            summary.posterior_mean_at_observation,
            summary.abs_error_at_observation,
            summary.prior_variance_at_observation,
            summary.posterior_variance_at_observation,
            summary.harmonic_free_truth_at_observation,
            summary.harmonic_free_posterior_mean_at_observation,
            summary.harmonic_free_abs_error_at_observation,
            summary.harmonic_free_prior_variance_at_observation,
            summary.harmonic_free_posterior_variance_at_observation,
        )?;
    }
    Ok(())
}

fn write_branch_observation_variance_diagnostics(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    write_branch_observation_relative_edge_variance_csv(result, branch, branch_dir)?;
    write_branch_observation_variance_profile_csv(result, branch, branch_dir)?;
    write_branch_observation_variance_diagnostics_txt(result, branch, branch_dir)?;
    Ok(())
}

fn write_branch_observation_relative_edge_variance_csv(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("observation_relative_edge_variance.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "observation_index,observed_edge_index,observation_direction,edge_index,is_observed_edge,intrinsic_distance,toroidal_alignment_sq,orientation_relation,prior_variance,posterior_variance,variance_ratio,harmonic_free_prior_variance,harmonic_free_posterior_variance,harmonic_free_variance_ratio"
    )?;

    for selected in &result.selected_observations {
        for edge_index in 0..branch.posterior_variance.len() {
            let intrinsic_distance = intrinsic_torus_distance(
                result.major_radius,
                result.minor_radius,
                result.edge_theta[edge_index],
                result.edge_phi[edge_index],
                selected.edge_theta,
                selected.edge_phi,
            );
            let orientation_relation = classify_orientation_relation(
                selected.direction,
                result.toroidal_alignment_sq[edge_index],
            );
            writeln!(
                writer,
                "{},{},{},{},{},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                selected.observation_index,
                selected.edge_index,
                selected.direction.as_str(),
                edge_index,
                edge_index == selected.edge_index,
                intrinsic_distance,
                result.toroidal_alignment_sq[edge_index],
                orientation_relation.as_str(),
                branch.prior_variance[edge_index],
                branch.posterior_variance[edge_index],
                safe_ratio(
                    branch.posterior_variance[edge_index],
                    branch.prior_variance[edge_index]
                ),
                branch.harmonic_free_prior_variance[edge_index],
                branch.harmonic_free_posterior_variance[edge_index],
                safe_ratio(
                    branch.harmonic_free_posterior_variance[edge_index],
                    branch.harmonic_free_prior_variance[edge_index],
                ),
            )?;
        }
    }

    Ok(())
}

fn write_branch_observation_variance_profile_csv(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("observation_variance_profile.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "observation_index,observed_edge_index,observation_direction,orientation_relation,distance_min_scale,distance_max_scale,distance_min,distance_max,count,mean_variance_ratio,min_variance_ratio,max_variance_ratio,mean_harmonic_free_variance_ratio,min_harmonic_free_variance_ratio,max_harmonic_free_variance_ratio"
    )?;

    let orientation_filters = [
        (None, "all"),
        (
            Some(ObservationOrientationRelation::Compatible),
            ObservationOrientationRelation::Compatible.as_str(),
        ),
        (
            Some(ObservationOrientationRelation::Oblique),
            ObservationOrientationRelation::Oblique.as_str(),
        ),
        (
            Some(ObservationOrientationRelation::Transverse),
            ObservationOrientationRelation::Transverse.as_str(),
        ),
    ];

    let mut previous_scale = 0.0;
    for &distance_max_scale in OBSERVATION_PROFILE_DISTANCE_SCALES {
        let distance_min_scale = previous_scale;
        previous_scale = distance_max_scale;

        for selected in &result.selected_observations {
            let distance_min = distance_min_scale * result.effective_range;
            let distance_max = distance_max_scale * result.effective_range;

            for (orientation_filter, orientation_label) in orientation_filters {
                let mut variance_ratios = Vec::new();
                let mut harmonic_free_variance_ratios = Vec::new();

                for edge_index in 0..branch.posterior_variance.len() {
                    if edge_index == selected.edge_index {
                        continue;
                    }

                    let intrinsic_distance = intrinsic_torus_distance(
                        result.major_radius,
                        result.minor_radius,
                        result.edge_theta[edge_index],
                        result.edge_phi[edge_index],
                        selected.edge_theta,
                        selected.edge_phi,
                    );
                    if intrinsic_distance <= distance_min || intrinsic_distance > distance_max {
                        continue;
                    }

                    let relation = classify_orientation_relation(
                        selected.direction,
                        result.toroidal_alignment_sq[edge_index],
                    );
                    if !orientation_relation_filter_matches(orientation_filter, relation) {
                        continue;
                    }

                    variance_ratios.push(safe_ratio(
                        branch.posterior_variance[edge_index],
                        branch.prior_variance[edge_index],
                    ));
                    harmonic_free_variance_ratios.push(safe_ratio(
                        branch.harmonic_free_posterior_variance[edge_index],
                        branch.harmonic_free_prior_variance[edge_index],
                    ));
                }

                let (count, mean_variance_ratio, min_variance_ratio, max_variance_ratio) =
                    summarize_values(&variance_ratios);
                let (
                    _harmonic_count,
                    mean_harmonic_free_variance_ratio,
                    min_harmonic_free_variance_ratio,
                    max_harmonic_free_variance_ratio,
                ) = summarize_values(&harmonic_free_variance_ratios);

                writeln!(
                    writer,
                    "{},{},{},{},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
                    selected.observation_index,
                    selected.edge_index,
                    selected.direction.as_str(),
                    orientation_label,
                    distance_min_scale,
                    distance_max_scale,
                    distance_min,
                    distance_max,
                    count,
                    mean_variance_ratio,
                    min_variance_ratio,
                    max_variance_ratio,
                    mean_harmonic_free_variance_ratio,
                    min_harmonic_free_variance_ratio,
                    max_harmonic_free_variance_ratio,
                )?;
            }
        }
    }

    Ok(())
}

fn write_branch_observation_variance_diagnostics_txt(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("observation_variance_diagnostics.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Observation-Oriented Variance Diagnostics")?;
    writeln!(writer, "branch={}", branch.name)?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "profile_distance_scales={:?}",
        OBSERVATION_PROFILE_DISTANCE_SCALES
    )?;
    writeln!(
        writer,
        "note=profile shells exclude the observed edge itself and split nearby edges into compatible, oblique, and transverse orientation classes"
    )?;

    for selected in &result.selected_observations {
        writeln!(
            writer,
            "observation={} edge={} direction={} toroidal_alignment_sq={:.6}",
            selected.observation_index,
            selected.edge_index,
            selected.direction.as_str(),
            selected.toroidal_alignment_sq,
        )?;
        for &(distance_max_scale, label) in
            &[(0.20, "very_local"), (0.50, "local"), (1.00, "range_scale")]
        {
            let distance_max = distance_max_scale * result.effective_range;
            let orientation_filters = [
                (
                    Some(ObservationOrientationRelation::Compatible),
                    "compatible",
                ),
                (Some(ObservationOrientationRelation::Oblique), "oblique"),
                (
                    Some(ObservationOrientationRelation::Transverse),
                    "transverse",
                ),
                (None, "all"),
            ];

            write!(
                writer,
                "  {}_radius_scale={:.2} intrinsic_radius={:.6}",
                label, distance_max_scale, distance_max
            )?;

            for (orientation_filter, orientation_label) in orientation_filters {
                let mut variance_ratios = Vec::new();
                for edge_index in 0..branch.posterior_variance.len() {
                    if edge_index == selected.edge_index {
                        continue;
                    }
                    let intrinsic_distance = intrinsic_torus_distance(
                        result.major_radius,
                        result.minor_radius,
                        result.edge_theta[edge_index],
                        result.edge_phi[edge_index],
                        selected.edge_theta,
                        selected.edge_phi,
                    );
                    if intrinsic_distance <= 0.0 || intrinsic_distance > distance_max {
                        continue;
                    }

                    let relation = classify_orientation_relation(
                        selected.direction,
                        result.toroidal_alignment_sq[edge_index],
                    );
                    if !orientation_relation_filter_matches(orientation_filter, relation) {
                        continue;
                    }
                    variance_ratios.push(safe_ratio(
                        branch.posterior_variance[edge_index],
                        branch.prior_variance[edge_index],
                    ));
                }

                let (count, mean_ratio, min_ratio, max_ratio) = summarize_values(&variance_ratios);
                write!(
                    writer,
                    " {}_count={} {}_mean_ratio={:.6} {}_min_ratio={:.6} {}_max_ratio={:.6}",
                    orientation_label,
                    count,
                    orientation_label,
                    mean_ratio,
                    orientation_label,
                    min_ratio,
                    orientation_label,
                    max_ratio,
                )?;
            }
            writeln!(writer)?;
        }
    }

    Ok(())
}

fn write_branch_variance_pattern_outputs(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    write_branch_variance_pattern_summary_csv(branch, branch_dir)?;
    write_branch_variance_pattern_shell_profiles_csv(branch, branch_dir)?;
    write_branch_variance_pattern_summary_txt(branch, branch_dir)?;
    write_branch_variance_pattern_cell_vtks(result, branch, branch_dir)?;
    write_branch_variance_pattern_observation_edge_vtks(result, branch, branch_dir)?;
    Ok(())
}

fn write_branch_variance_pattern_summary_csv(
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("variance_pattern_summary.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "object,observation_count,very_local_ratio,local_ratio,range_ratio,far_ratio,localization_auc,monotonicity_score,very_local_orientation_contrast,local_orientation_contrast"
    )?;
    for row in &branch.variance_pattern.summary_rows {
        writeln!(
            writer,
            "{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.object,
            row.observation_count,
            row.very_local_ratio,
            row.local_ratio,
            row.range_ratio,
            row.far_ratio,
            row.localization_auc,
            row.monotonicity_score,
            row.very_local_orientation_contrast,
            row.local_orientation_contrast,
        )?;
    }
    Ok(())
}

fn write_branch_variance_pattern_shell_profiles_csv(
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("variance_pattern_shell_profiles.csv"))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "object,observation_index,observation_direction,distance_min_scale,distance_max_scale,distance_min,distance_max,shell_mid_scale,count,mean_ratio"
    )?;
    for row in &branch.variance_pattern.shell_profile_rows {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12}",
            row.object,
            row.observation_index,
            row.observation_direction.as_str(),
            row.distance_min_scale,
            row.distance_max_scale,
            row.distance_min,
            row.distance_max,
            row.shell_mid_scale,
            row.count,
            row.mean_ratio,
        )?;
    }
    Ok(())
}

fn write_branch_variance_pattern_summary_txt(
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("variance_pattern_summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "Variance Pattern Summary")?;
    writeln!(writer, "branch={}", branch.name)?;
    writeln!(
        writer,
        "smoothing_bandwidth={}",
        branch.variance_pattern.smoothing_bandwidth
    )?;
    writeln!(
        writer,
        "smoothing_cutoff={}",
        branch.variance_pattern.smoothing_cutoff
    )?;
    writeln!(
        writer,
        "shell_distance_scales={:?}",
        branch.variance_pattern.shell_distance_scales
    )?;
    writeln!(
        writer,
        "note=raw edge metrics diagnose anisotropy; smoothed matched-component and circulation are the primary radial-decay diagnostics"
    )?;
    for row in &branch.variance_pattern.summary_rows {
        writeln!(
            writer,
            "object={} very_local_ratio={} local_ratio={} range_ratio={} far_ratio={} localization_auc={} monotonicity_score={} very_local_orientation_contrast={} local_orientation_contrast={}",
            row.object,
            row.very_local_ratio,
            row.local_ratio,
            row.range_ratio,
            row.far_ratio,
            row.localization_auc,
            row.monotonicity_score,
            row.very_local_orientation_contrast,
            row.local_orientation_contrast,
        )?;
    }
    Ok(())
}

fn write_branch_variance_pattern_cell_vtks(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    write_top_cell_scalar_vtk_fields(
        branch_dir.join("reconstructed_component_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_var_toroidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .toroidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_toroidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_toroidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .toroidal
                    .ratio
                    .as_slice(),
            ),
            (
                "prior_var_poloidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .poloidal
                    .prior
                    .as_slice(),
            ),
            (
                "post_var_poloidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "ratio_poloidal",
                branch
                    .variance_pattern
                    .reconstructed
                    .poloidal
                    .ratio
                    .as_slice(),
            ),
            (
                "trace_prior",
                branch.variance_pattern.reconstructed.trace.prior.as_slice(),
            ),
            (
                "trace_post",
                branch
                    .variance_pattern
                    .reconstructed
                    .trace
                    .posterior
                    .as_slice(),
            ),
            (
                "trace_ratio",
                branch.variance_pattern.reconstructed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    write_top_cell_scalar_vtk_fields(
        branch_dir.join("smoothed_component_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "smoothed_prior_toroidal",
                branch.variance_pattern.smoothed.toroidal.prior.as_slice(),
            ),
            (
                "smoothed_post_toroidal",
                branch
                    .variance_pattern
                    .smoothed
                    .toroidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_toroidal",
                branch.variance_pattern.smoothed.toroidal.ratio.as_slice(),
            ),
            (
                "smoothed_prior_poloidal",
                branch.variance_pattern.smoothed.poloidal.prior.as_slice(),
            ),
            (
                "smoothed_post_poloidal",
                branch
                    .variance_pattern
                    .smoothed
                    .poloidal
                    .posterior
                    .as_slice(),
            ),
            (
                "smoothed_ratio_poloidal",
                branch.variance_pattern.smoothed.poloidal.ratio.as_slice(),
            ),
            (
                "smoothed_trace_prior",
                branch.variance_pattern.smoothed.trace.prior.as_slice(),
            ),
            (
                "smoothed_trace_post",
                branch.variance_pattern.smoothed.trace.posterior.as_slice(),
            ),
            (
                "smoothed_trace_ratio",
                branch.variance_pattern.smoothed.trace.ratio.as_slice(),
            ),
        ],
    )?;
    write_top_cell_scalar_vtk_fields(
        branch_dir.join("circulation_variance.vtk"),
        &result.coords,
        &result.topology,
        &[
            (
                "prior_circulation",
                branch.variance_pattern.circulation.prior.as_slice(),
            ),
            (
                "post_circulation",
                branch.variance_pattern.circulation.posterior.as_slice(),
            ),
            (
                "ratio_circulation",
                branch.variance_pattern.circulation.ratio.as_slice(),
            ),
        ],
    )?;
    Ok(())
}

fn write_branch_variance_pattern_observation_edge_vtks(
    result: &Torus1FormConditioningResult,
    branch: &Torus1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let observation_dir = branch_dir.join("observation_edge_vtks");
    fs::create_dir_all(&observation_dir)?;
    let variance_ratio = Cochain::new(
        1,
        ratio_vector(&branch.posterior_variance, &branch.prior_variance),
    );
    for selected in &result.selected_observations {
        let distance = Cochain::new(
            1,
            FeecVector::from_iterator(
                branch.posterior_variance.len(),
                (0..branch.posterior_variance.len()).map(|edge_index| {
                    intrinsic_torus_distance(
                        result.major_radius,
                        result.minor_radius,
                        result.edge_theta[edge_index],
                        result.edge_phi[edge_index],
                        selected.edge_theta,
                        selected.edge_phi,
                    )
                }),
            ),
        );
        let compatible_mask = Cochain::new(
            1,
            FeecVector::from_iterator(
                branch.posterior_variance.len(),
                (0..branch.posterior_variance.len()).map(|edge_index| {
                    f64::from(
                        classify_orientation_relation(
                            selected.direction,
                            result.toroidal_alignment_sq[edge_index],
                        ) == ObservationOrientationRelation::Compatible,
                    )
                }),
            ),
        );
        let oblique_mask = Cochain::new(
            1,
            FeecVector::from_iterator(
                branch.posterior_variance.len(),
                (0..branch.posterior_variance.len()).map(|edge_index| {
                    f64::from(
                        classify_orientation_relation(
                            selected.direction,
                            result.toroidal_alignment_sq[edge_index],
                        ) == ObservationOrientationRelation::Oblique,
                    )
                }),
            ),
        );
        let transverse_mask = Cochain::new(
            1,
            FeecVector::from_iterator(
                branch.posterior_variance.len(),
                (0..branch.posterior_variance.len()).map(|edge_index| {
                    f64::from(
                        classify_orientation_relation(
                            selected.direction,
                            result.toroidal_alignment_sq[edge_index],
                        ) == ObservationOrientationRelation::Transverse,
                    )
                }),
            ),
        );
        write_1cochain_vtk_fields(
            observation_dir.join(format!("observation_{:02}.vtk", selected.observation_index)),
            &result.coords,
            &result.topology,
            &[
                ("distance_to_observation", &distance),
                ("compatible_mask", &compatible_mask),
                ("oblique_mask", &oblique_mask),
                ("transverse_mask", &transverse_mask),
                ("variance_ratio", &variance_ratio),
            ],
        )?;
    }
    Ok(())
}

fn write_branch_summary(
    branch: &Torus1FormBranchResult,
    result: &Torus1FormConditioningResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(branch_dir.join("summary.txt"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "branch={}", branch.name)?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "neighbourhood_radius={}",
        result.neighbourhood_radius
    )?;
    writeln!(writer, "far_radius={}", result.far_radius)?;
    writeln!(
        writer,
        "harmonic_coefficients_truth={},{}",
        branch.harmonic_coefficients_truth[0], branch.harmonic_coefficients_truth[1]
    )?;
    writeln!(
        writer,
        "harmonic_coefficients_posterior_mean={},{}",
        branch.harmonic_coefficients_posterior_mean[0],
        branch.harmonic_coefficients_posterior_mean[1]
    )?;
    write_observed_summary(&mut writer, &branch.summary.observed)?;
    write_region_summary(&mut writer, "near", &branch.summary.near)?;
    write_region_summary(&mut writer, "far", &branch.summary.far)?;
    Ok(())
}

fn write_observed_summary(writer: &mut impl Write, observed: &ObservedSummary) -> io::Result<()> {
    writeln!(writer, "observed_count={}", observed.count)?;
    writeln!(writer, "observed_max_abs_error={}", observed.max_abs_error)?;
    writeln!(
        writer,
        "observed_mean_abs_error={}",
        observed.mean_abs_error
    )?;
    writeln!(
        writer,
        "observed_harmonic_free_mean_abs_error={}",
        observed.harmonic_free_mean_abs_error
    )?;
    writeln!(
        writer,
        "observed_variance_ratio_mean={}",
        observed.variance_ratio_mean
    )?;
    writeln!(
        writer,
        "observed_harmonic_free_variance_ratio_mean={}",
        observed.harmonic_free_variance_ratio_mean
    )?;
    Ok(())
}

fn write_region_summary(
    writer: &mut impl Write,
    label: &str,
    region: &RegionSummary,
) -> io::Result<()> {
    writeln!(writer, "{}_count={}", label, region.count)?;
    writeln!(writer, "{}_mean_abs_error={}", label, region.mean_abs_error)?;
    writeln!(
        writer,
        "{}_harmonic_free_mean_abs_error={}",
        label, region.harmonic_free_mean_abs_error
    )?;
    writeln!(
        writer,
        "{}_variance_ratio_mean={}",
        label, region.variance_ratio_mean
    )?;
    writeln!(
        writer,
        "{}_harmonic_free_variance_ratio_mean={}",
        label, region.harmonic_free_variance_ratio_mean
    )?;
    writeln!(
        writer,
        "{}_variance_reduction_mean={}",
        label, region.variance_reduction_mean
    )?;
    writeln!(
        writer,
        "{}_harmonic_free_variance_reduction_mean={}",
        label, region.harmonic_free_variance_reduction_mean
    )?;
    Ok(())
}

fn gmrf_vec_to_feec(vec: &GmrfVector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::types::CooMatrix as GmrfCooMatrix;

    #[test]
    fn wrap_angle_difference_uses_shortest_arc() {
        let delta = wrap_angle_difference(PI - 0.1, -PI + 0.1);
        assert!((delta + 0.2).abs() < 1e-12);

        let delta = wrap_angle_difference(-PI + 0.1, PI - 0.1);
        assert!((delta - 0.2).abs() < 1e-12);
    }

    #[test]
    fn intrinsic_torus_distance_wraps_phi_across_branch_cut() {
        let distance = intrinsic_torus_distance(1.0, 0.3, 0.2, PI - 0.1, 0.2, -PI + 0.1);
        let expected = (1.0 + 0.3 * 0.2_f64.cos()) * 0.2;
        assert!((distance - expected).abs() < 1e-12);
    }

    #[test]
    fn safe_ratio_returns_zero_for_zero_denominator() {
        assert_eq!(safe_ratio(1.0, 0.0), 0.0);
    }

    #[test]
    fn classify_orientation_relation_distinguishes_compatible_oblique_and_transverse() {
        assert_eq!(
            classify_orientation_relation(ObservationDirection::Toroidal, 0.95),
            ObservationOrientationRelation::Compatible
        );
        assert_eq!(
            classify_orientation_relation(ObservationDirection::Toroidal, 0.50),
            ObservationOrientationRelation::Oblique
        );
        assert_eq!(
            classify_orientation_relation(ObservationDirection::Toroidal, 0.05),
            ObservationOrientationRelation::Transverse
        );

        assert_eq!(
            classify_orientation_relation(ObservationDirection::Poloidal, 0.05),
            ObservationOrientationRelation::Compatible
        );
        assert_eq!(
            classify_orientation_relation(ObservationDirection::Poloidal, 0.50),
            ObservationOrientationRelation::Oblique
        );
        assert_eq!(
            classify_orientation_relation(ObservationDirection::Poloidal, 0.95),
            ObservationOrientationRelation::Transverse
        );
    }

    #[test]
    fn transformed_rbmc_identity_matches_latent_estimator() {
        let mut coo = GmrfCooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.5);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 2.5);
        let precision = GmrfSparseMatrix::from(&coo);
        let constraints = GmrfDenseMatrix::zeros(0, 3);

        let mut latent_workspace = build_rbmc_workspace(&precision, &constraints).unwrap();
        let latent = estimate_latent_rbmc_variances(&mut latent_workspace, 3, 24, 4, 17).unwrap();

        let mut transformed_workspace = build_rbmc_workspace(&precision, &constraints).unwrap();
        let identity = SparseRowLinearOperator::identity(3);
        let transformed =
            estimate_transformed_rbmc_variances(&mut transformed_workspace, &identity, 24, 4, 17)
                .unwrap();

        assert_eq!(latent.unconstrained, transformed.unconstrained);
        assert_eq!(latent.harmonic_free, transformed.harmonic_free);
    }

    #[test]
    fn exact_latent_variances_match_gmrf_decomposition() {
        let mut coo = GmrfCooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.5);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 2.5);
        let precision = GmrfSparseMatrix::from(&coo);
        let constraints = GmrfDenseMatrix::from_fn(1, 3, |_, j| if j == 0 { 1.0 } else { 0.0 });

        let mut workspace = build_rbmc_workspace(&precision, &constraints).unwrap();
        let exact = exact_latent_variances(&mut workspace, &constraints).unwrap();

        let mut gmrf =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(3), precision.clone()).unwrap();
        let decomposition = gmrf
            .exact_constrained_variance_decomposition(&constraints)
            .unwrap();

        assert_eq!(exact.unconstrained, decomposition.unconstrained_diag);
        assert_eq!(exact.harmonic_free, decomposition.constrained_diag);
    }

    #[test]
    fn exact_transformed_variances_match_latent_for_identity_operator() {
        let mut coo = GmrfCooMatrix::new(3, 3);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 3.5);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 2.5);
        let precision = GmrfSparseMatrix::from(&coo);
        let constraints = GmrfDenseMatrix::zeros(0, 3);

        let mut latent_workspace = build_rbmc_workspace(&precision, &constraints).unwrap();
        let latent = exact_latent_variances(&mut latent_workspace, &constraints).unwrap();

        let mut transformed_workspace = build_rbmc_workspace(&precision, &constraints).unwrap();
        let identity = SparseRowLinearOperator::identity(3);
        let transformed =
            exact_transformed_variances(&mut transformed_workspace, &identity).unwrap();

        assert_eq!(latent.unconstrained, transformed.unconstrained);
        assert_eq!(latent.harmonic_free, transformed.harmonic_free);
    }

    #[test]
    fn clip_rbmc_estimates_caps_posterior_at_prior() {
        let prior = RbmcVarianceEstimates {
            unconstrained: GmrfVector::from_vec(vec![1.0, 2.0]),
            harmonic_free: GmrfVector::from_vec(vec![0.5, 0.25]),
        };
        let posterior = RbmcVarianceEstimates {
            unconstrained: GmrfVector::from_vec(vec![1.5, 1.25]),
            harmonic_free: GmrfVector::from_vec(vec![0.75, 0.1]),
        };

        let clipped = clip_rbmc_estimates_to_prior(&prior, &posterior);
        assert_eq!(clipped.unconstrained, GmrfVector::from_vec(vec![1.0, 1.25]));
        assert_eq!(clipped.harmonic_free, GmrfVector::from_vec(vec![0.5, 0.1]));
    }

    #[test]
    fn smoothing_operator_rows_are_normalized_and_cut_off() {
        let geometry = TorusCellGeometry {
            major_radius: 2.0,
            minor_radius: 0.7,
            theta: vec![0.0, 0.0, 0.0],
            phi: vec![0.0, 0.05, 1.0],
        };
        let smoothing = build_gaussian_smoothing_operator(&geometry, 0.1, 0.2).unwrap();

        for row in &smoothing.rows {
            let sum = row.iter().map(|(_, value)| *value).sum::<f64>();
            assert!((sum - 1.0).abs() <= 1e-12);
        }
        assert!(
            smoothing.rows[0].iter().all(|(col, _)| *col != 2),
            "cutoff should exclude distant cells from the smoothing stencil"
        );
    }

    #[test]
    fn circulation_operator_matches_face_boundary_incidence() {
        let topology = Complex::standard(2);
        let operator = build_local_circulation_operator(&topology, topology.edges().len()).unwrap();
        let boundary = topology.boundary_operator(2);

        assert_eq!(operator.nrows(), topology.cells().len());
        assert_eq!(operator.ncols, topology.edges().len());
        assert_eq!(operator.rows[0].len(), 3);

        let mut expected = BTreeMap::new();
        for (edge_index, cell_index, value) in boundary.triplet_iter() {
            assert_eq!(cell_index, 0);
            expected.insert(edge_index, *value);
        }
        let actual = operator.rows[0]
            .iter()
            .copied()
            .collect::<BTreeMap<usize, f64>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn reconstructed_component_operators_have_top_cell_rows_and_local_support() {
        let config = Torus1FormConditioningConfig::default();
        let mesh_bytes = fs::read(&config.mesh_path).unwrap();
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let edge_geometry = build_torus_edge_geometry(&topology, &coords).unwrap();
        let cell_geometry = build_torus_cell_geometry(
            &topology,
            &coords,
            edge_geometry.major_radius,
            edge_geometry.minor_radius,
        )
        .unwrap();

        let toroidal =
            build_reconstructed_component_operator(&topology, &coords, &cell_geometry, true)
                .unwrap();
        let poloidal =
            build_reconstructed_component_operator(&topology, &coords, &cell_geometry, false)
                .unwrap();

        assert_eq!(toroidal.nrows(), topology.cells().len());
        assert_eq!(poloidal.nrows(), topology.cells().len());
        assert_eq!(toroidal.ncols, topology.edges().len());
        assert!(toroidal.rows.iter().all(|row| row.len() <= 3));
        assert!(poloidal.rows.iter().all(|row| row.len() <= 3));
        assert!(toroidal.rows.iter().any(|row| !row.is_empty()));
        assert!(poloidal.rows.iter().any(|row| !row.is_empty()));
    }

    #[test]
    fn embedded_component_operators_have_top_cell_rows_and_local_support() {
        let config = Torus1FormConditioningConfig::default();
        let mesh_bytes = fs::read(&config.mesh_path).unwrap();
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);

        let x = build_embedded_component_operator(&topology, &coords, 0).unwrap();
        let y = build_embedded_component_operator(&topology, &coords, 1).unwrap();
        let z = build_embedded_component_operator(&topology, &coords, 2).unwrap();

        assert_eq!(x.nrows(), topology.cells().len());
        assert_eq!(y.nrows(), topology.cells().len());
        assert_eq!(z.nrows(), topology.cells().len());
        assert_eq!(x.ncols, topology.edges().len());
        assert!(x.rows.iter().all(|row| row.len() <= 3));
        assert!(y.rows.iter().all(|row| row.len() <= 3));
        assert!(z.rows.iter().all(|row| row.len() <= 3));
        assert!(x.rows.iter().any(|row| !row.is_empty()));
        assert!(y.rows.iter().any(|row| !row.is_empty()));
        assert!(z.rows.iter().any(|row| !row.is_empty()));
    }

    #[test]
    fn ambient_field_set_trace_sums_xyz_components() {
        let prior = Torus1FormAmbientVarianceEstimates {
            x: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![1.0, 2.0]),
                harmonic_free: GmrfVector::from_vec(vec![0.5, 1.5]),
            },
            y: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![0.25, 0.75]),
                harmonic_free: GmrfVector::from_vec(vec![0.2, 0.3]),
            },
            z: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![0.1, 0.4]),
                harmonic_free: GmrfVector::from_vec(vec![0.05, 0.15]),
            },
        };
        let posterior = Torus1FormAmbientVarianceEstimates {
            x: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![0.5, 1.0]),
                harmonic_free: GmrfVector::from_vec(vec![0.25, 0.5]),
            },
            y: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![0.125, 0.25]),
                harmonic_free: GmrfVector::from_vec(vec![0.1, 0.2]),
            },
            z: RbmcVarianceEstimates {
                unconstrained: GmrfVector::from_vec(vec![0.05, 0.2]),
                harmonic_free: GmrfVector::from_vec(vec![0.025, 0.1]),
            },
        };

        let fields = build_ambient_field_set(&prior, &posterior, false);
        assert!((fields.trace.prior[0] - 1.35).abs() < 1e-12);
        assert!((fields.trace.prior[1] - 3.15).abs() < 1e-12);
        assert!((fields.trace.posterior[0] - 0.675).abs() < 1e-12);
        assert!((fields.trace.posterior[1] - 1.45).abs() < 1e-12);
    }
}
