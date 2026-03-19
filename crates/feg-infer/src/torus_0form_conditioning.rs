use crate::diagnostics::infer_torus_radii;
use crate::matern_0form::{
    build_laplace_beltrami_0form, build_matern_precision_0form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use crate::util::convert_whittle_params_to_matern;
use common::linalg::nalgebra::Vector as FeecVector;
use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector};
use gmrf_core::Gmrf;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::io;
use std::path::PathBuf;

const DEFAULT_OBSERVATION_TARGETS: [(f64, f64); 5] = [
    (-2.35, -2.40),
    (-0.75, 0.85),
    (0.65, 2.30),
    (1.95, -0.55),
    (2.55, 1.60),
];

#[derive(Debug, Clone)]
pub struct Torus0FormConditioningConfig {
    pub mesh_path: PathBuf,
    pub kappa: f64,
    pub tau: f64,
    pub noise_variance: f64,
    pub neighbourhood_radius_scale: f64,
    pub observation_targets: Vec<(f64, f64)>,
}

impl Default for Torus0FormConditioningConfig {
    fn default() -> Self {
        Self {
            mesh_path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../meshes/torus_shell.msh"),
            kappa: 5.0,
            tau: 1.0,
            noise_variance: 1e-6,
            neighbourhood_radius_scale: 0.75,
            observation_targets: DEFAULT_OBSERVATION_TARGETS.to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ObservationConditioningSummary {
    pub observation_index: usize,
    pub vertex_index: usize,
    pub theta: f64,
    pub phi: f64,
    pub observation_value: f64,
    pub posterior_mean_at_observation: f64,
    pub abs_error_at_observation: f64,
    pub prior_variance_at_observation: f64,
    pub posterior_variance_at_observation: f64,
    pub neighbourhood_count: usize,
    pub neighbourhood_mean: f64,
    pub neighbourhood_mean_abs_deviation_from_observation: f64,
    pub global_mean_abs_deviation_from_observation: f64,
    pub neighbourhood_prior_variance_mean: f64,
    pub neighbourhood_posterior_variance_mean: f64,
    pub neighbourhood_variance_reduction_mean: f64,
}

pub struct Torus0FormConditioningResult {
    pub topology: Complex,
    pub coords: MeshCoords,
    pub truth: FeecVector,
    pub posterior_mean: FeecVector,
    pub absolute_mean_error: FeecVector,
    pub prior_variance: FeecVector,
    pub posterior_variance: FeecVector,
    pub variance_reduction: FeecVector,
    pub observed_mask: FeecVector,
    pub nearest_observation_value: FeecVector,
    pub observation_indices: Vec<usize>,
    pub observation_values: Vec<f64>,
    pub observation_summaries: Vec<ObservationConditioningSummary>,
    pub theta: FeecVector,
    pub phi: FeecVector,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub effective_range: f64,
    pub neighbourhood_radius: f64,
}

struct TorusVertexGeometry {
    major_radius: f64,
    minor_radius: f64,
    theta: Vec<f64>,
    phi: Vec<f64>,
}

pub fn run_torus_0form_conditioning(
    config: &Torus0FormConditioningConfig,
) -> Result<Torus0FormConditioningResult, Box<dyn Error>> {
    validate_config(config)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let geometry = build_torus_vertex_geometry(&coords)?;
    let truth = build_artificial_truth(&geometry);
    let observation_indices =
        select_observation_vertices(&coords, &geometry, &config.observation_targets)?;

    let laplace = build_laplace_beltrami_0form(&topology, &metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let truth_gmrf = feec_vec_to_gmrf(&truth);
    let observation_matrix = observation_selector(truth.len(), &observation_indices);
    let observations = &observation_matrix * &truth_gmrf;

    let empty_constraints = GmrfDenseMatrix::zeros(0, truth.len());
    let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(truth.len()), q_prior.clone())?;
    let prior_variance = prior
        .exact_constrained_variance_decomposition(&empty_constraints)?
        .unconstrained_diag;

    let (posterior_precision, information) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        config.noise_variance,
    );
    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)?;
    let posterior_mean = posterior.mean().clone();
    let posterior_variance = posterior
        .exact_constrained_variance_decomposition(&empty_constraints)?
        .unconstrained_diag;

    let (_nu, _variance, effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);
    let neighbourhood_radius = config.neighbourhood_radius_scale * effective_range;

    let posterior_mean = gmrf_vec_to_feec(&posterior_mean);
    let prior_variance = gmrf_vec_to_feec(&prior_variance);
    let posterior_variance = gmrf_vec_to_feec(&posterior_variance);
    let absolute_mean_error = FeecVector::from_iterator(
        truth.len(),
        (0..truth.len()).map(|i| (posterior_mean[i] - truth[i]).abs()),
    );
    let variance_reduction = FeecVector::from_iterator(
        truth.len(),
        (0..truth.len()).map(|i| prior_variance[i] - posterior_variance[i]),
    );
    let observation_values = observation_indices
        .iter()
        .map(|&idx| truth[idx])
        .collect::<Vec<_>>();
    let observation_summaries = build_observation_summaries(
        &coords,
        &geometry,
        &observation_indices,
        &observation_values,
        &posterior_mean,
        &prior_variance,
        &posterior_variance,
        neighbourhood_radius,
    );
    let observed_mask = build_observed_mask(truth.len(), &observation_indices);
    let nearest_observation_value =
        build_nearest_observation_value_field(&coords, &observation_indices, &observation_values);

    Ok(Torus0FormConditioningResult {
        topology,
        coords,
        truth,
        posterior_mean,
        absolute_mean_error,
        prior_variance,
        posterior_variance,
        variance_reduction,
        observed_mask,
        nearest_observation_value,
        observation_indices,
        observation_values,
        observation_summaries,
        theta: FeecVector::from_vec(geometry.theta),
        phi: FeecVector::from_vec(geometry.phi),
        major_radius: geometry.major_radius,
        minor_radius: geometry.minor_radius,
        effective_range,
        neighbourhood_radius,
    })
}

fn validate_config(config: &Torus0FormConditioningConfig) -> Result<(), Box<dyn Error>> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(invalid_input("kappa must be finite and positive").into());
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(invalid_input("tau must be finite and positive").into());
    }
    if !config.noise_variance.is_finite() || config.noise_variance <= 0.0 {
        return Err(invalid_input("noise_variance must be finite and positive").into());
    }
    if !config.neighbourhood_radius_scale.is_finite() || config.neighbourhood_radius_scale <= 0.0 {
        return Err(invalid_input("neighbourhood_radius_scale must be finite and positive").into());
    }
    if config.observation_targets.is_empty() {
        return Err(invalid_input("at least one observation target is required").into());
    }
    Ok(())
}

fn build_torus_vertex_geometry(coords: &MeshCoords) -> Result<TorusVertexGeometry, io::Error> {
    let (major_radius, minor_radius) = infer_torus_radii(coords).map_err(invalid_data)?;
    let mut theta = Vec::with_capacity(coords.nvertices());
    let mut phi = Vec::with_capacity(coords.nvertices());

    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = coord[1];
        let z = coord[2];
        let rho = (x * x + y * y).sqrt();
        phi.push(y.atan2(x));
        theta.push(z.atan2(rho - major_radius));
    }

    Ok(TorusVertexGeometry {
        major_radius,
        minor_radius,
        theta,
        phi,
    })
}

fn build_artificial_truth(geometry: &TorusVertexGeometry) -> FeecVector {
    FeecVector::from_iterator(
        geometry.theta.len(),
        geometry
            .theta
            .iter()
            .zip(&geometry.phi)
            .map(|(&theta, &phi)| {
                0.35 + 0.60 * (phi - 0.4).cos()
                    + 0.30 * (theta + 0.7).sin()
                    + 0.10 * (phi - theta).cos()
            }),
    )
}

fn select_observation_vertices(
    coords: &MeshCoords,
    geometry: &TorusVertexGeometry,
    targets: &[(f64, f64)],
) -> Result<Vec<usize>, io::Error> {
    let mut selected = Vec::with_capacity(targets.len());
    let mut used = HashSet::with_capacity(targets.len());

    for &(theta, phi) in targets {
        let target = torus_point(geometry.major_radius, geometry.minor_radius, theta, phi);
        let best = (0..coords.nvertices())
            .filter(|idx| !used.contains(idx))
            .min_by(|lhs, rhs| {
                let lhs_dist = squared_distance_to_target(&target, coords.coord(*lhs));
                let rhs_dist = squared_distance_to_target(&target, coords.coord(*rhs));
                lhs_dist
                    .partial_cmp(&rhs_dist)
                    .expect("torus vertex distances should be finite")
            })
            .ok_or_else(|| invalid_data("failed to find a unique observation vertex"))?;
        used.insert(best);
        selected.push(best);
    }

    Ok(selected)
}

fn build_observation_summaries(
    coords: &MeshCoords,
    geometry: &TorusVertexGeometry,
    observation_indices: &[usize],
    observation_values: &[f64],
    posterior_mean: &FeecVector,
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
    neighbourhood_radius: f64,
) -> Vec<ObservationConditioningSummary> {
    observation_indices
        .iter()
        .zip(observation_values.iter().copied())
        .enumerate()
        .map(|(observation_index, (&vertex_index, observation_value))| {
            let center = coord_to_point(coords.coord(vertex_index));
            let mut neighbourhood_count = 0usize;
            let mut neighbourhood_mean_sum = 0.0;
            let mut neighbourhood_abs_dev_sum = 0.0;
            let mut neighbourhood_prior_var_sum = 0.0;
            let mut neighbourhood_posterior_var_sum = 0.0;
            let mut global_abs_dev_sum = 0.0;

            for (idx, coord) in coords.coord_iter().enumerate() {
                let abs_dev = (posterior_mean[idx] - observation_value).abs();
                global_abs_dev_sum += abs_dev;
                if euclidean_distance_to_point(coord, &center) <= neighbourhood_radius {
                    neighbourhood_count += 1;
                    neighbourhood_mean_sum += posterior_mean[idx];
                    neighbourhood_abs_dev_sum += abs_dev;
                    neighbourhood_prior_var_sum += prior_variance[idx];
                    neighbourhood_posterior_var_sum += posterior_variance[idx];
                }
            }

            let neighbourhood_count_f64 = neighbourhood_count as f64;
            let neighbourhood_mean = neighbourhood_mean_sum / neighbourhood_count_f64;
            let neighbourhood_mean_abs_deviation_from_observation =
                neighbourhood_abs_dev_sum / neighbourhood_count_f64;
            let neighbourhood_prior_variance_mean =
                neighbourhood_prior_var_sum / neighbourhood_count_f64;
            let neighbourhood_posterior_variance_mean =
                neighbourhood_posterior_var_sum / neighbourhood_count_f64;

            ObservationConditioningSummary {
                observation_index,
                vertex_index,
                theta: geometry.theta[vertex_index],
                phi: geometry.phi[vertex_index],
                observation_value,
                posterior_mean_at_observation: posterior_mean[vertex_index],
                abs_error_at_observation: (posterior_mean[vertex_index] - observation_value).abs(),
                prior_variance_at_observation: prior_variance[vertex_index],
                posterior_variance_at_observation: posterior_variance[vertex_index],
                neighbourhood_count,
                neighbourhood_mean,
                neighbourhood_mean_abs_deviation_from_observation,
                global_mean_abs_deviation_from_observation: global_abs_dev_sum
                    / coords.nvertices() as f64,
                neighbourhood_prior_variance_mean,
                neighbourhood_posterior_variance_mean,
                neighbourhood_variance_reduction_mean: neighbourhood_prior_variance_mean
                    - neighbourhood_posterior_variance_mean,
            }
        })
        .collect()
}

fn build_observed_mask(dimension: usize, observation_indices: &[usize]) -> FeecVector {
    let mut mask = FeecVector::zeros(dimension);
    for &idx in observation_indices {
        mask[idx] = 1.0;
    }
    mask
}

fn build_nearest_observation_value_field(
    coords: &MeshCoords,
    observation_indices: &[usize],
    observation_values: &[f64],
) -> FeecVector {
    FeecVector::from_iterator(
        coords.nvertices(),
        coords.coord_iter().map(|coord| {
            let point = coord_to_point(coord);
            observation_indices
                .iter()
                .zip(observation_values.iter().copied())
                .min_by(|(lhs_idx, _), (rhs_idx, _)| {
                    let lhs_dist = squared_distance_to_target(&point, coords.coord(**lhs_idx));
                    let rhs_dist = squared_distance_to_target(&point, coords.coord(**rhs_idx));
                    lhs_dist
                        .partial_cmp(&rhs_dist)
                        .expect("distances should be finite")
                })
                .map(|(_, value)| value)
                .expect("at least one observation is required")
        }),
    )
}

fn torus_point(major_radius: f64, minor_radius: f64, theta: f64, phi: f64) -> [f64; 3] {
    let rho = major_radius + minor_radius * theta.cos();
    [rho * phi.cos(), rho * phi.sin(), minor_radius * theta.sin()]
}

fn coord_to_point(coord: manifold::geometry::coord::CoordRef<'_>) -> [f64; 3] {
    [coord[0], coord[1], coord[2]]
}

fn squared_distance_to_target(
    target: &[f64; 3],
    coord: manifold::geometry::coord::CoordRef<'_>,
) -> f64 {
    let dx = coord[0] - target[0];
    let dy = coord[1] - target[1];
    let dz = coord[2] - target[2];
    dx * dx + dy * dy + dz * dz
}

fn euclidean_distance_to_point(
    coord: manifold::geometry::coord::CoordRef<'_>,
    point: &[f64; 3],
) -> f64 {
    squared_distance_to_target(point, coord).sqrt()
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
    use std::collections::HashSet;

    fn default_mesh_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh")
    }

    #[test]
    fn torus_point_matches_requested_radii() {
        let point = torus_point(1.0, 0.3, std::f64::consts::FRAC_PI_2, 0.0);
        let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
        assert!((rho - 1.0).abs() < 1e-12);
        assert!((point[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn artificial_truth_is_finite_and_nontrivial_on_torus_mesh() {
        let mesh_bytes = std::fs::read(default_mesh_path()).expect("torus mesh should load");
        let (_topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let geometry = build_torus_vertex_geometry(&coords).expect("geometry should build");
        let truth = build_artificial_truth(&geometry);

        let min = truth.iter().copied().fold(f64::INFINITY, f64::min);
        let max = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(truth.iter().all(|value| value.is_finite()));
        assert!(max - min > 0.5, "truth field should vary across the torus");
    }

    #[test]
    fn observation_vertex_selection_is_unique() {
        let mesh_bytes = std::fs::read(default_mesh_path()).expect("torus mesh should load");
        let (_topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let geometry = build_torus_vertex_geometry(&coords).expect("geometry should build");
        let selected =
            select_observation_vertices(&coords, &geometry, &DEFAULT_OBSERVATION_TARGETS)
                .expect("observation vertices should be selected");

        let unique = selected.iter().copied().collect::<HashSet<_>>();
        assert_eq!(selected.len(), DEFAULT_OBSERVATION_TARGETS.len());
        assert_eq!(unique.len(), selected.len());
        assert!(selected.iter().all(|idx| *idx < coords.nvertices()));
    }
}
