use crate::diagnostics::infer_torus_radii;
#[cfg(test)]
use crate::hodge_1form_conditioning::lock_feec_harmonic_tests;
use crate::hodge_1form_conditioning::{
    compute_harmonic_basis_1form, harmonic_coefficients_1form,
    mass_orthonormalize_harmonic_basis_1form, run_hodge_1form_conditioning, Hodge1FormBranchResult,
    Hodge1FormConditioningConfig,
};
use crate::matern_1form::build_hodge_laplacian_1form;
use crate::torus_1form_conditioning::{ObservationDirection, Torus1FormObservationTarget};
use crate::torus_1form_pde_conditioning::default_torus_shell_resolution_1_mesh_path;
use crate::vtk::write_1cochain_vtk_fields;
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use exterior::field::EmbeddedDiffFormClosure;
use formoniq::io::{sample_1form_cell_vectors, write_top_cell_vtk_fields};
use manifold::{
    geometry::coord::mesh::MeshCoords, io::gmsh::gmsh2coord_complex, topology::complex::Complex,
};
use std::collections::HashSet;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

const EPS: f64 = 1e-12;
const HARMONIC_TOROIDAL_SCALE: f64 = 0.75;
const HARMONIC_POLOIDAL_SCALE: f64 = -0.50;
const TOROIDAL_ALIGNMENT_MIN: f64 = 0.8;
const POLOIDAL_ALIGNMENT_MAX: f64 = 0.2;

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

#[derive(Debug, Clone)]
pub struct Torus1FormHodgeConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub harmonic_dim: usize,
    pub observation_targets: Vec<Torus1FormObservationTarget>,
}

impl Default for Torus1FormHodgeConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_resolution_1_mesh_path(),
            kappa: 4.0,
            tau: 1.0,
            noise_variance: 1e-8,
            harmonic_dim: 2,
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
pub struct Torus1FormHodgeConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub truth: FeecVector,
    pub observations: FeecVector,
    pub observation_indices: Vec<usize>,
    pub selected_observations: Vec<Torus1FormSelectedObservation>,
    pub edge_theta: FeecVector,
    pub edge_phi: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
    pub harmonic_basis: common::linalg::nalgebra::Matrix,
    pub exact: Hodge1FormBranchResult,
    pub coexact: Hodge1FormBranchResult,
    pub harmonic: Hodge1FormBranchResult,
}

struct TorusEdgeGeometry {
    major_radius: f64,
    minor_radius: f64,
    theta: Vec<f64>,
    phi: Vec<f64>,
    toroidal_alignment_sq: Vec<f64>,
}

pub fn run_torus_1form_hodge_conditioning(
    config: &Torus1FormHodgeConditioningConfig,
) -> Result<Torus1FormHodgeConditioningResult, Box<dyn Error>> {
    validate_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let edge_geometry = build_torus_edge_geometry(&topology, &coords)?;
    let selected_observations =
        select_observation_edges(&edge_geometry, &config.observation_targets)?;
    let observation_indices = selected_observations
        .iter()
        .map(|observation| observation.edge_index)
        .collect::<Vec<_>>();
    let observation_matrix = selector_to_feec_csr(topology.skeleton(1).len(), &observation_indices);

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis_raw =
        compute_harmonic_basis_1form(&topology, &metric, config.harmonic_dim, None)?;
    let harmonic_basis =
        mass_orthonormalize_harmonic_basis_1form(&harmonic_basis_raw, &hodge.mass_u)?;
    let truth = build_truth(&topology, &coords, &hodge.mass_u, &harmonic_basis)?;

    let generic = run_hodge_1form_conditioning(&Hodge1FormConditioningConfig {
        topology: &topology,
        coords: &coords,
        metric: &metric,
        truth: &truth,
        observation_matrix: &observation_matrix,
        kappa: config.kappa,
        tau: config.tau,
        noise_variance: config.noise_variance,
        harmonic_dim: config.harmonic_dim,
        harmonic_basis_override: Some(&harmonic_basis_raw),
    })?;

    Ok(Torus1FormHodgeConditioningResult {
        topology,
        coords,
        truth,
        observations: generic.observations,
        observation_indices,
        selected_observations,
        edge_theta: FeecVector::from_vec(edge_geometry.theta),
        edge_phi: FeecVector::from_vec(edge_geometry.phi),
        toroidal_alignment_sq: FeecVector::from_vec(edge_geometry.toroidal_alignment_sq),
        harmonic_basis: generic.harmonic_basis,
        exact: generic.exact,
        coexact: generic.coexact,
        harmonic: generic.harmonic,
    })
}

pub fn write_torus_1form_hodge_conditioning_outputs(
    result: &Torus1FormHodgeConditioningResult,
    out_dir: impl AsRef<Path>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    write_selected_observations_csv(result, out_dir.join("selected_observations.csv"))?;
    write_branch_outputs(result, &result.exact, out_dir)?;
    write_branch_outputs(result, &result.coexact, out_dir)?;
    write_branch_outputs(result, &result.harmonic, out_dir)?;
    Ok(())
}

fn validate_config(config: &Torus1FormHodgeConditioningConfig) -> Result<(), Box<dyn Error>> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive").into());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive").into());
    }
    if config.harmonic_dim == 0 {
        return Err(invalid_input("harmonic_dim must be at least 1 for the torus example").into());
    }
    if config.observation_targets.is_empty() {
        return Err(invalid_input("at least one observation target is required").into());
    }
    Ok(())
}

fn build_truth(
    topology: &Complex,
    coords: &MeshCoords,
    mass_u: &FeecCsr,
    harmonic_basis: &common::linalg::nalgebra::Matrix,
) -> Result<FeecVector, io::Error> {
    let (major_radius, minor_radius) = infer_torus_radii(coords).map_err(invalid_data)?;
    let seed = build_local_seed_cochain(topology, coords, major_radius, minor_radius);
    let mut truth = remove_harmonic_content(&seed.coeffs, harmonic_basis, mass_u);
    if harmonic_basis.ncols() >= 1 {
        truth += harmonic_basis
            .column(0)
            .into_owned()
            .scale(HARMONIC_TOROIDAL_SCALE);
    }
    if harmonic_basis.ncols() >= 2 {
        truth += harmonic_basis
            .column(1)
            .into_owned()
            .scale(HARMONIC_POLOIDAL_SCALE);
    }
    Ok(truth)
}

fn remove_harmonic_content(
    field: &FeecVector,
    harmonic_basis: &common::linalg::nalgebra::Matrix,
    mass_u: &FeecCsr,
) -> FeecVector {
    let coefficients = harmonic_coefficients_1form(field, harmonic_basis, mass_u);
    let mut harmonic_free = field.clone();
    for j in 0..harmonic_basis.ncols() {
        harmonic_free -= harmonic_basis.column(j).into_owned().scale(coefficients[j]);
    }
    harmonic_free
}

fn selector_to_feec_csr(dimension: usize, indices: &[usize]) -> FeecCsr {
    let selector = gmrf_core::observation::observation_selector(dimension, indices);
    let mut coo = FeecCoo::new(selector.nrows(), selector.ncols());
    for (row, col, value) in selector.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
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
            if matches_alignment(target.direction, geometry.toroidal_alignment_sq[edge_index]) {
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

fn matches_alignment(direction: ObservationDirection, toroidal_alignment_sq: f64) -> bool {
    match direction {
        ObservationDirection::Toroidal => toroidal_alignment_sq >= TOROIDAL_ALIGNMENT_MIN,
        ObservationDirection::Poloidal => toroidal_alignment_sq <= POLOIDAL_ALIGNMENT_MAX,
    }
}

fn update_best_candidate(best: &mut Option<(usize, f64)>, edge_index: usize, distance: f64) {
    match best {
        Some((_, best_distance)) if distance >= *best_distance => {}
        _ => *best = Some((edge_index, distance)),
    }
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

fn write_selected_observations_csv(
    result: &Torus1FormHodgeConditioningResult,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "observation_index,edge_index,direction,target_theta,target_phi,edge_theta,edge_phi,toroidal_alignment_sq,selection_distance,used_fallback"
    )?;
    for observation in &result.selected_observations {
        writeln!(
            writer,
            "{},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{}",
            observation.observation_index,
            observation.edge_index,
            observation_direction_str(observation.direction),
            observation.target_theta,
            observation.target_phi,
            observation.edge_theta,
            observation.edge_phi,
            observation.toroidal_alignment_sq,
            observation.selection_distance,
            observation.used_fallback
        )?;
    }
    Ok(())
}

fn write_branch_outputs(
    result: &Torus1FormHodgeConditioningResult,
    branch: &Hodge1FormBranchResult,
    out_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let branch_dir = out_dir.join(branch.kind.as_str());
    fs::create_dir_all(&branch_dir)?;

    let truth = Cochain::new(1, result.truth.clone());
    let posterior_mean = Cochain::new(1, branch.posterior_mean.clone());
    let absolute_mean_error = Cochain::new(1, branch.absolute_mean_error.clone());
    let prior_variance = Cochain::new(1, branch.prior_variance.clone());
    let posterior_variance = Cochain::new(1, branch.posterior_variance.clone());
    let variance_reduction = Cochain::new(1, branch.variance_reduction.clone());

    write_1cochain_vtk_fields(
        branch_dir.join("edge_fields.vtk"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("posterior_mean", &posterior_mean),
            ("absolute_mean_error", &absolute_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
        ],
    )?;
    write_branch_surface_vector_vtk(result, branch, &branch_dir)?;

    let mut summary = BufWriter::new(File::create(branch_dir.join("summary.txt"))?);
    writeln!(summary, "branch={}", branch.kind.as_str())?;
    writeln!(summary, "latent_dimension={}", branch.latent_dimension)?;
    writeln!(summary, "observation_count={}", branch.observation_count)?;
    writeln!(
        summary,
        "max_abs_observation_error={}",
        branch.max_abs_observation_error
    )?;
    writeln!(
        summary,
        "mean_abs_observation_error={}",
        branch.mean_abs_observation_error
    )?;
    writeln!(
        summary,
        "harmonic_residual_norm_truth={}",
        branch.harmonic_residual_norm_truth
    )?;
    writeln!(
        summary,
        "harmonic_residual_norm_posterior_mean={}",
        branch.harmonic_residual_norm_posterior_mean
    )?;
    writeln!(
        summary,
        "curl_residual_norm_posterior_mean={}",
        branch.posterior_bias_diagnostics.curl_residual_norm
    )?;
    writeln!(
        summary,
        "curl_residual_relative_posterior_mean={}",
        branch.posterior_bias_diagnostics.curl_residual_relative
    )?;
    writeln!(
        summary,
        "coclosed_residual_norm_posterior_mean={}",
        branch.posterior_bias_diagnostics.coclosed_residual_norm
    )?;
    writeln!(
        summary,
        "coclosed_residual_relative_posterior_mean={}",
        branch.posterior_bias_diagnostics.coclosed_residual_relative
    )?;
    write_vector_csv(
        branch_dir.join("observations.csv"),
        "observation_index,edge_index,observed_value,posterior_mean,residual",
        result
            .selected_observations
            .iter()
            .enumerate()
            .map(|(row, observation)| {
                format!(
                    "{},{},{:.12},{:.12},{:.12}",
                    row,
                    observation.edge_index,
                    branch.observation_values[row],
                    branch.posterior_observation_mean[row],
                    branch.observation_residual[row]
                )
            }),
    )?;

    Ok(())
}

fn write_branch_surface_vector_vtk(
    result: &Torus1FormHodgeConditioningResult,
    branch: &Hodge1FormBranchResult,
    branch_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let truth = Cochain::new(1, result.truth.clone());
    let posterior_mean = Cochain::new(1, branch.posterior_mean.clone());
    let absolute_mean_error = Cochain::new(1, branch.absolute_mean_error.clone());

    let truth_vectors = sample_1form_cell_vectors(&result.coords, &result.topology, &truth)?;
    let posterior_mean_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &posterior_mean)?;
    let absolute_mean_error_vectors =
        sample_1form_cell_vectors(&result.coords, &result.topology, &absolute_mean_error)?;

    let truth_magnitude = vector_magnitudes(&truth_vectors);
    let posterior_mean_magnitude = vector_magnitudes(&posterior_mean_vectors);
    let absolute_mean_error_magnitude = vector_magnitudes(&absolute_mean_error_vectors);
    let surface = &branch.reconstructed_barycenter_variance;
    let posterior_variance_vectors = surface.posterior_vtk_vectors();
    let prior_variance_vectors = surface.prior_vtk_vectors();
    let posterior_marginal_std = surface.posterior_trace.map(|value| value.max(0.0).sqrt());

    write_top_cell_vtk_fields(
        branch_dir.join("posterior_mean_surface_vector.vtk"),
        &result.coords,
        &result.topology,
        &[
            ("truth_surface_vector", truth_vectors.as_slice()),
            (
                "posterior_mean_surface_vector",
                posterior_mean_vectors.as_slice(),
            ),
            (
                "absolute_mean_error_surface_vector",
                absolute_mean_error_vectors.as_slice(),
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
            ("truth_magnitude", truth_magnitude.as_slice()),
            ("magnitude", posterior_mean_magnitude.as_slice()),
            (
                "absolute_mean_error_magnitude",
                absolute_mean_error_magnitude.as_slice(),
            ),
            ("marginal_variance", surface.posterior_trace.as_slice()),
            ("marginal_std", posterior_marginal_std.as_slice()),
            ("prior_marginal_variance", surface.prior_trace.as_slice()),
            ("marginal_variance_ratio", surface.trace_ratio.as_slice()),
        ],
    )?;

    Ok(())
}

fn write_vector_csv(
    path: impl AsRef<Path>,
    header: &str,
    rows: impl IntoIterator<Item = String>,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{header}")?;
    for row in rows {
        writeln!(writer, "{row}")?;
    }
    Ok(())
}

fn observation_direction_str(direction: ObservationDirection) -> &'static str {
    match direction {
        ObservationDirection::Toroidal => "toroidal",
        ObservationDirection::Poloidal => "poloidal",
    }
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|vector| {
            (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn torus_hodge_conditioning_runs_with_feec_harmonics_by_default() {
        let _lock = lock_feec_harmonic_tests();
        let result =
            run_torus_1form_hodge_conditioning(&Torus1FormHodgeConditioningConfig::default())
                .expect("torus Hodge conditioning should run");

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let out_dir =
            std::env::temp_dir().join(format!("feg_infer_torus_hodge_conditioning_{stamp}"));
        write_torus_1form_hodge_conditioning_outputs(&result, &out_dir)
            .expect("torus Hodge outputs should write");

        assert_eq!(result.harmonic_basis.ncols(), 2);
        assert_eq!(
            result.exact.observation_count,
            result.selected_observations.len()
        );
        assert_eq!(
            result.coexact.observation_count,
            result.selected_observations.len()
        );
        assert_eq!(
            result.harmonic.observation_count,
            result.selected_observations.len()
        );
        assert!(result
            .exact
            .posterior_mean
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .coexact
            .posterior_mean
            .iter()
            .all(|value| value.is_finite()));
        assert!(result
            .harmonic
            .posterior_mean
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
        assert!(
            result
                .coexact
                .posterior_bias_diagnostics
                .coclosed_residual_relative
                <= 3e-2,
            "coexact posterior mean coclosed defect regressed: {:?}",
            result.coexact.posterior_bias_diagnostics
        );

        for branch in ["exact", "coexact", "harmonic"] {
            let surface_vtk = out_dir
                .join(branch)
                .join("posterior_mean_surface_vector.vtk");
            assert!(
                surface_vtk.is_file(),
                "missing reconstructed surface vector VTK for branch {branch}: {surface_vtk:?}"
            );
            let content = fs::read_to_string(&surface_vtk)
                .expect("surface vector VTK should be readable as text");
            assert!(content.contains("VECTORS posterior_mean_surface_vector double"));
            assert!(content.contains("VECTORS truth_surface_vector double"));
            assert!(content.contains("VECTORS absolute_mean_error_surface_vector double"));
            assert!(content.contains("VECTORS posterior_directional_variance double"));
            assert!(content.contains("VECTORS prior_directional_variance double"));
            assert!(content.contains("SCALARS marginal_variance double"));
            assert!(content.contains("SCALARS marginal_std double"));
            assert!(content.contains("SCALARS prior_marginal_variance double"));
            assert!(content.contains("SCALARS marginal_variance_ratio double"));
        }

        fs::remove_dir_all(&out_dir).expect("temporary output directory should clean up");
    }
}
