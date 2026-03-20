use crate::matern_0form::{
    build_matern_precision_0form, build_matern_system_matrix_0form, LaplaceBeltrami0Form,
    MaternConfig as Matern0FormConfig, MaternMassInverse as Matern0FormMassInverse,
};
use crate::matern_1form::{
    build_matern_mass_inverse_1form, build_matern_precision_1form,
    build_matern_system_matrix_1form, HodgeLaplacian1Form, MaternConfig, MaternMassInverse,
};
use crate::sparse::feec_csr_to_gmrf;
use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use exterior::field::EmbeddedDiffFormClosure;
use formoniq::{assemble::assemble_galvec, operators::SourceElVec};
use gmrf_core::{
    types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector},
    Gmrf, Solver,
};
use manifold::{
    geometry::{coord::mesh::MeshCoords, metric::mesh::MeshLengths},
    topology::complex::Complex,
};
use rand::SeedableRng;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SummaryStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
}

impl SummaryStats {
    pub fn ratio(&self) -> f64 {
        if self.min.abs() <= EPS {
            f64::INFINITY
        } else {
            self.max / self.min
        }
    }
}

#[derive(Debug, Clone)]
pub struct EdgeVarianceDiagnostics {
    pub edge_lengths: FeecVector,
    pub mass_diag: FeecVector,
    pub mass_lumped_diag: FeecVector,
    pub forcing_scale_diag: FeecVector,
    pub variances: FeecVector,
    pub std_devs: FeecVector,
    pub variance_per_length: FeecVector,
    pub variance_per_length2: FeecVector,
    pub variance_per_mass_diag: FeecVector,
    pub variance_per_mass_lumped_diag: FeecVector,
    pub variance_per_forcing_scale_diag: FeecVector,
    pub std_per_length: FeecVector,
    pub std_per_sqrt_mass_diag: FeecVector,
    pub std_per_sqrt_mass_lumped_diag: FeecVector,
    pub std_per_sqrt_forcing_scale_diag: FeecVector,
}

#[derive(Debug, Clone)]
pub struct NodeVarianceDiagnostics {
    pub rho: FeecVector,
    pub mass_diag: FeecVector,
    pub mass_lumped_diag: FeecVector,
    pub forcing_scale_diag: FeecVector,
    pub variances: FeecVector,
    pub std_devs: FeecVector,
    pub variance_per_mass_diag: FeecVector,
    pub variance_per_mass_lumped_diag: FeecVector,
    pub variance_per_forcing_scale_diag: FeecVector,
    pub std_per_sqrt_mass_diag: FeecVector,
    pub std_per_sqrt_mass_lumped_diag: FeecVector,
    pub std_per_sqrt_forcing_scale_diag: FeecVector,
}

#[derive(Debug, Clone)]
pub struct StandardizedForcingDiagnostics {
    pub mean: FeecVector,
    pub variances: FeecVector,
    pub std_devs: FeecVector,
}

#[derive(Debug, Clone)]
pub struct HarmonicAttributionDiagnostics {
    pub removed_edge_diagnostics: EdgeVarianceDiagnostics,
    pub removed_fraction: FeecVector,
}

#[derive(Debug, Clone, Copy)]
pub struct LinearEqualityConstraints<'a> {
    pub matrix: &'a GmrfDenseMatrix,
    pub rhs: &'a GmrfVector,
}

#[derive(Debug, Clone)]
pub struct StrategyMetricSummary {
    pub variance_per_forcing_scale_diag: Option<SummaryStats>,
    pub standardized_forcing_mean: Option<SummaryStats>,
    pub abs_standardized_forcing_mean: Option<SummaryStats>,
    pub standardized_forcing_variance: Option<SummaryStats>,
    pub corr_variance_per_forcing_scale_diag_midpoint_rho: Option<f64>,
    pub corr_standardized_forcing_variance_midpoint_rho: Option<f64>,
    pub corr_standardized_forcing_variance_edge_length2: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct MaternStrategyDiagnosticsReport {
    pub mass_inverse: MaternMassInverse,
    pub prior_precision: FeecCsr,
    pub forcing_scale_diag: FeecVector,
    pub edge_diagnostics: EdgeVarianceDiagnostics,
    pub constrained_edge_diagnostics: Option<EdgeVarianceDiagnostics>,
    pub harmonic_attribution: Option<HarmonicAttributionDiagnostics>,
    pub standardized_forcing: StandardizedForcingDiagnostics,
    pub metric_summary: StrategyMetricSummary,
}

#[derive(Debug, Clone)]
pub struct Matern0FormDiagnosticsReport {
    pub mass_inverse: Matern0FormMassInverse,
    pub prior_precision: FeecCsr,
    pub forcing_scale_diag: FeecVector,
    pub node_diagnostics: NodeVarianceDiagnostics,
    pub standardized_forcing: StandardizedForcingDiagnostics,
}

#[derive(Debug, Clone)]
pub struct TorusEdgeGeometry {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub midpoint_rho: FeecVector,
    pub midpoint_theta: FeecVector,
    pub gaussian_curvature: FeecVector,
    pub toroidal_alignment_sq: FeecVector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimateWithError {
    pub mean: f64,
    pub standard_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContributionEstimate {
    pub absolute: EstimateWithError,
    pub fraction_of_total: EstimateWithError,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhomogeneityContributions {
    pub total: f64,
    pub harmonic: f64,
    pub edge_length: f64,
    pub minor_angle: f64,
    pub direction: f64,
    pub interaction: f64,
    pub residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhomogeneityContributionSummary {
    pub total: EstimateWithError,
    pub harmonic: ContributionEstimate,
    pub edge_length: ContributionEstimate,
    pub minor_angle: ContributionEstimate,
    pub direction: ContributionEstimate,
    pub interaction: ContributionEstimate,
    pub residual: ContributionEstimate,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualStudyContributions {
    pub total_post_length: f64,
    pub position_even_fourier: f64,
    pub direction_legendre: f64,
    pub interaction_even: f64,
    pub discrete_surrogates: f64,
    pub unexplained: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualStudyContributionSummary {
    pub total_post_length: EstimateWithError,
    pub position_even_fourier: ContributionEstimate,
    pub direction_legendre: ContributionEstimate,
    pub interaction_even: ContributionEstimate,
    pub discrete_surrogates: ContributionEstimate,
    pub unexplained: ContributionEstimate,
}

#[derive(Debug, Clone)]
pub struct TorusVarianceAttributionDiagnostics {
    pub harmonic_free_edge_diagnostics: EdgeVarianceDiagnostics,
    pub harmonic_removed_edge_diagnostics: EdgeVarianceDiagnostics,
    pub harmonic_removed_fraction: FeecVector,
    pub geometry: TorusEdgeGeometry,
    pub field_decomposition: InhomogeneityFieldDecomposition,
    pub contribution_summary: InhomogeneityContributionSummary,
    pub batch_contributions: Vec<InhomogeneityContributions>,
    pub rbmc_batch_sizes: Vec<usize>,
    pub rbmc_floor_hits: usize,
    pub harmonic_free_floor_hits: usize,
}

#[derive(Debug, Clone)]
pub struct InhomogeneityFieldDecomposition {
    pub log_unconstrained_variance: FeecVector,
    pub log_harmonic_free_variance: FeecVector,
    pub log_harmonic_free_variance_per_length2: FeecVector,
    pub minor_angle_component: FeecVector,
    pub direction_component: FeecVector,
    pub interaction_component: FeecVector,
    pub residual_component: FeecVector,
}

#[derive(Debug, Clone)]
pub struct ResidualStudyFieldDecomposition {
    pub log_harmonic_free_variance_per_length2: FeecVector,
    pub position_even_fourier_component: FeecVector,
    pub direction_legendre_component: FeecVector,
    pub interaction_even_component: FeecVector,
    pub discrete_surrogate_component: FeecVector,
    pub unexplained_residual: FeecVector,
}

#[derive(Debug, Clone)]
pub struct Matern1FormTorusAttributionReport {
    pub mass_inverse: MaternMassInverse,
    pub prior_precision: FeecCsr,
    pub forcing_scale_diag: FeecVector,
    pub edge_diagnostics: EdgeVarianceDiagnostics,
    pub torus_attribution: TorusVarianceAttributionDiagnostics,
    pub standardized_forcing: StandardizedForcingDiagnostics,
}

#[derive(Debug, Clone)]
pub struct Matern1FormTorusResidualStudyReport {
    pub mass_inverse: MaternMassInverse,
    pub prior_precision: FeecCsr,
    pub forcing_scale_diag: FeecVector,
    pub edge_diagnostics: EdgeVarianceDiagnostics,
    pub harmonic_free_edge_diagnostics: EdgeVarianceDiagnostics,
    pub harmonic_removed_edge_diagnostics: EdgeVarianceDiagnostics,
    pub harmonic_removed_fraction: FeecVector,
    pub geometry: TorusEdgeGeometry,
    pub field_decomposition: ResidualStudyFieldDecomposition,
    pub contribution_summary: ResidualStudyContributionSummary,
    pub batch_contributions: Vec<ResidualStudyContributions>,
    pub rbmc_batch_sizes: Vec<usize>,
    pub rbmc_floor_hits: usize,
    pub harmonic_free_floor_hits: usize,
}

struct TorusVarianceBaseEstimates {
    prior: Gmrf,
    prior_precision: FeecCsr,
    forcing_scale_diag: FeecVector,
    geometry: TorusEdgeGeometry,
    edge_diagnostics: EdgeVarianceDiagnostics,
    harmonic_free_edge_diagnostics: EdgeVarianceDiagnostics,
    harmonic_removed_edge_diagnostics: EdgeVarianceDiagnostics,
    harmonic_removed_fraction: FeecVector,
    prior_variances: GmrfVector,
    harmonic_free_variances: GmrfVector,
    batch_variances: Vec<GmrfVector>,
    batch_harmonic_free_variances: Vec<GmrfVector>,
    rbmc_batch_sizes: Vec<usize>,
    rbmc_floor_hits: usize,
    harmonic_free_floor_hits: usize,
}

pub fn summarize_vector(vec: &FeecVector) -> Option<SummaryStats> {
    if vec.is_empty() {
        return None;
    }

    let n = vec.len() as f64;
    let mean = vec.iter().sum::<f64>() / n;
    let var = vec.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let min = vec
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).expect("finite values"))
        .expect("nonempty vector");
    let max = vec
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).expect("finite values"))
        .expect("nonempty vector");

    Some(SummaryStats {
        min,
        max,
        mean,
        std: var.sqrt(),
    })
}

pub fn pearson_correlation(lhs: &FeecVector, rhs: &FeecVector) -> Option<f64> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }

    let n = lhs.len() as f64;
    let mean_l = lhs.iter().sum::<f64>() / n;
    let mean_r = rhs.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_l = 0.0;
    let mut var_r = 0.0;

    for i in 0..lhs.len() {
        let dl = lhs[i] - mean_l;
        let dr = rhs[i] - mean_r;
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }

    if var_l <= EPS || var_r <= EPS {
        return None;
    }

    Some(cov / (var_l.sqrt() * var_r.sqrt()))
}

pub fn edge_lengths(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let edge_skeleton = topology.skeleton(1);
    FeecVector::from_iterator(
        edge_skeleton.len(),
        edge_skeleton.handle_iter().map(|edge| {
            let v0 = coords.coord(edge.vertices[0]);
            let v1 = coords.coord(edge.vertices[1]);
            (v1 - v0).norm()
        }),
    )
}

pub fn edge_midpoint_rho(topology: &Complex, coords: &MeshCoords) -> FeecVector {
    let edge_skeleton = topology.skeleton(1);
    FeecVector::from_iterator(
        edge_skeleton.len(),
        edge_skeleton.handle_iter().map(|edge| {
            let v0 = coords.coord(edge.vertices[0]);
            let v1 = coords.coord(edge.vertices[1]);
            let midpoint = (v0 + v1) / 2.0;
            (midpoint[0].powi(2) + midpoint[1].powi(2)).sqrt()
        }),
    )
}

pub fn vertex_rho(coords: &MeshCoords) -> FeecVector {
    FeecVector::from_iterator(
        coords.nvertices(),
        coords.coord_iter().map(|coord| {
            let x = coord[0];
            let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
            (x * x + y * y).sqrt()
        }),
    )
}

pub fn infer_torus_radii(coords: &MeshCoords) -> Result<(f64, f64), String> {
    if coords.dim() < 3 {
        return Err(format!(
            "expected 3D embedded torus coordinates, got dimension {}",
            coords.dim()
        ));
    }

    let mut rho_min = f64::INFINITY;
    let mut rho_max = 0.0_f64;
    for coord in coords.coord_iter() {
        let rho = (coord[0].powi(2) + coord[1].powi(2)).sqrt();
        rho_min = rho_min.min(rho);
        rho_max = rho_max.max(rho);
    }

    let major_radius = 0.5 * (rho_max + rho_min);
    let minor_radius = 0.5 * (rho_max - rho_min);
    if !major_radius.is_finite() || !minor_radius.is_finite() || minor_radius <= EPS {
        return Err("failed to infer positive torus radii from mesh coordinates".to_string());
    }
    Ok((major_radius, minor_radius))
}

pub fn build_analytic_torus_harmonic_basis(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
) -> Result<FeecMatrix, String> {
    let (major_radius, minor_radius) = infer_torus_radii(coords)?;
    let toroidal = EmbeddedDiffFormClosure::ambient_one_form(
        |p| {
            let x = p[0];
            let y = p[1];
            let rho = (x * x + y * y).sqrt().max(EPS);
            FeecVector::from_column_slice(&[-y / (rho * rho), x / (rho * rho), 0.0])
        },
        coords.dim(),
        topology.dim(),
    );
    let poloidal = EmbeddedDiffFormClosure::ambient_one_form(
        move |p| {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            let rho = (x * x + y * y).sqrt().max(EPS);
            FeecVector::from_column_slice(&[
                -z * x / (minor_radius * rho * rho),
                -z * y / (minor_radius * rho * rho),
                (rho - major_radius) / (minor_radius * rho),
            ])
        },
        coords.dim(),
        topology.dim(),
    );

    let toroidal_coeffs =
        assemble_galvec(topology, metric, SourceElVec::new(&toroidal, coords, None));
    let poloidal_coeffs =
        assemble_galvec(topology, metric, SourceElVec::new(&poloidal, coords, None));

    Ok(FeecMatrix::from_columns(&[
        toroidal_coeffs,
        poloidal_coeffs,
    ]))
}

pub fn build_torus_edge_geometry(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<TorusEdgeGeometry, String> {
    let (major_radius, minor_radius) = infer_torus_radii(coords)?;
    let edge_skeleton = topology.skeleton(1);

    let mut midpoint_rho = Vec::with_capacity(edge_skeleton.len());
    let mut midpoint_theta = Vec::with_capacity(edge_skeleton.len());
    let mut gaussian_curvature = Vec::with_capacity(edge_skeleton.len());
    let mut toroidal_alignment_sq = Vec::with_capacity(edge_skeleton.len());

    for edge in edge_skeleton.handle_iter() {
        let v0 = coords.coord(edge.vertices[0]);
        let v1 = coords.coord(edge.vertices[1]);
        let midpoint = (v0 + v1) / 2.0;
        let rho = (midpoint[0].powi(2) + midpoint[1].powi(2)).sqrt();
        let theta = midpoint[2].atan2(rho - major_radius);
        let phi = midpoint[1].atan2(midpoint[0]);
        let tangent = v1 - v0;
        let tangent_norm = tangent.norm();

        let alignment_sq = if tangent_norm <= EPS || rho <= EPS {
            0.0
        } else {
            let unit_tangent = tangent / tangent_norm;
            let e_phi = FeecVector::from_column_slice(&[-phi.sin(), phi.cos(), 0.0]);
            unit_tangent.dot(&e_phi).powi(2).clamp(0.0, 1.0)
        };

        midpoint_rho.push(rho);
        midpoint_theta.push(theta);
        gaussian_curvature.push(torus_gaussian_curvature(major_radius, minor_radius, rho));
        toroidal_alignment_sq.push(alignment_sq);
    }

    Ok(TorusEdgeGeometry {
        major_radius,
        minor_radius,
        midpoint_rho: FeecVector::from_vec(midpoint_rho),
        midpoint_theta: FeecVector::from_vec(midpoint_theta),
        gaussian_curvature: FeecVector::from_vec(gaussian_curvature),
        toroidal_alignment_sq: FeecVector::from_vec(toroidal_alignment_sq),
    })
}

fn torus_gaussian_curvature(major_radius: f64, minor_radius: f64, rho: f64) -> f64 {
    let denom = minor_radius * minor_radius * rho;
    if denom.abs() <= EPS {
        0.0
    } else {
        (rho - major_radius) / denom
    }
}

pub fn compute_edge_variance_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    mass_matrix: &FeecCsr,
    variances: &GmrfVector,
    forcing_scale_diag: &FeecVector,
) -> Result<EdgeVarianceDiagnostics, String> {
    let edge_skeleton = topology.skeleton(1);
    if mass_matrix.nrows() != edge_skeleton.len() || mass_matrix.ncols() != edge_skeleton.len() {
        return Err(format!(
            "mass matrix shape {}x{} does not match edge count {}",
            mass_matrix.nrows(),
            mass_matrix.ncols(),
            edge_skeleton.len()
        ));
    }
    if variances.len() != edge_skeleton.len() {
        return Err(format!(
            "variance length {} does not match edge count {}",
            variances.len(),
            edge_skeleton.len()
        ));
    }
    if forcing_scale_diag.len() != edge_skeleton.len() {
        return Err(format!(
            "forcing scale length {} does not match edge count {}",
            forcing_scale_diag.len(),
            edge_skeleton.len()
        ));
    }

    let edge_lengths = edge_lengths(topology, coords);
    let mass_diag = matrix_diag(mass_matrix);
    let mass_lumped_diag = lumped_diag(mass_matrix);
    let variances = FeecVector::from_vec(variances.iter().copied().collect());
    let std_devs = variances.map(|v| v.max(0.0).sqrt());
    let length_squared = edge_lengths.component_mul(&edge_lengths);
    let sqrt_mass_diag = mass_diag.map(safe_sqrt);
    let sqrt_mass_lumped_diag = mass_lumped_diag.map(safe_sqrt);
    let sqrt_forcing_scale_diag = forcing_scale_diag.map(safe_sqrt);

    let variance_per_length = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(variances[i], edge_lengths[i])),
    );
    let variance_per_length2 = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(variances[i], length_squared[i])),
    );
    let std_per_length = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(std_devs[i], edge_lengths[i])),
    );
    let variance_per_mass_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(variances[i], mass_diag[i])),
    );
    let variance_per_mass_lumped_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(variances[i], mass_lumped_diag[i])),
    );
    let variance_per_forcing_scale_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(variances[i], forcing_scale_diag[i])),
    );
    let std_per_sqrt_mass_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(std_devs[i], sqrt_mass_diag[i])),
    );
    let std_per_sqrt_mass_lumped_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(std_devs[i], sqrt_mass_lumped_diag[i])),
    );
    let std_per_sqrt_forcing_scale_diag = FeecVector::from_iterator(
        edge_skeleton.len(),
        (0..edge_skeleton.len()).map(|i| safe_div(std_devs[i], sqrt_forcing_scale_diag[i])),
    );

    Ok(EdgeVarianceDiagnostics {
        edge_lengths,
        mass_diag,
        mass_lumped_diag,
        forcing_scale_diag: forcing_scale_diag.clone(),
        variances,
        std_devs,
        variance_per_length,
        variance_per_length2,
        variance_per_mass_diag,
        variance_per_mass_lumped_diag,
        variance_per_forcing_scale_diag,
        std_per_length,
        std_per_sqrt_mass_diag,
        std_per_sqrt_mass_lumped_diag,
        std_per_sqrt_forcing_scale_diag,
    })
}

pub fn compute_node_variance_diagnostics(
    coords: &MeshCoords,
    mass_matrix: &FeecCsr,
    variances: &GmrfVector,
    forcing_scale_diag: &FeecVector,
) -> Result<NodeVarianceDiagnostics, String> {
    if mass_matrix.nrows() != coords.nvertices() || mass_matrix.ncols() != coords.nvertices() {
        return Err(format!(
            "mass matrix shape {}x{} does not match vertex count {}",
            mass_matrix.nrows(),
            mass_matrix.ncols(),
            coords.nvertices()
        ));
    }
    if variances.len() != coords.nvertices() {
        return Err(format!(
            "variance length {} does not match vertex count {}",
            variances.len(),
            coords.nvertices()
        ));
    }
    if forcing_scale_diag.len() != coords.nvertices() {
        return Err(format!(
            "forcing scale length {} does not match vertex count {}",
            forcing_scale_diag.len(),
            coords.nvertices()
        ));
    }

    let rho = vertex_rho(coords);
    let mass_diag = matrix_diag(mass_matrix);
    let mass_lumped_diag = lumped_diag(mass_matrix);
    let variances = FeecVector::from_vec(variances.iter().copied().collect());
    let std_devs = variances.map(|v| v.max(0.0).sqrt());
    let sqrt_mass_diag = mass_diag.map(safe_sqrt);
    let sqrt_mass_lumped_diag = mass_lumped_diag.map(safe_sqrt);
    let sqrt_forcing_scale_diag = forcing_scale_diag.map(safe_sqrt);

    let variance_per_mass_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(variances[i], mass_diag[i])),
    );
    let variance_per_mass_lumped_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(variances[i], mass_lumped_diag[i])),
    );
    let variance_per_forcing_scale_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(variances[i], forcing_scale_diag[i])),
    );
    let std_per_sqrt_mass_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(std_devs[i], sqrt_mass_diag[i])),
    );
    let std_per_sqrt_mass_lumped_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(std_devs[i], sqrt_mass_lumped_diag[i])),
    );
    let std_per_sqrt_forcing_scale_diag = FeecVector::from_iterator(
        coords.nvertices(),
        (0..coords.nvertices()).map(|i| safe_div(std_devs[i], sqrt_forcing_scale_diag[i])),
    );

    Ok(NodeVarianceDiagnostics {
        rho,
        mass_diag,
        mass_lumped_diag,
        forcing_scale_diag: forcing_scale_diag.clone(),
        variances,
        std_devs,
        variance_per_mass_diag,
        variance_per_mass_lumped_diag,
        variance_per_forcing_scale_diag,
        std_per_sqrt_mass_diag,
        std_per_sqrt_mass_lumped_diag,
        std_per_sqrt_forcing_scale_diag,
    })
}

pub fn forcing_scale_diag_from_strategy(
    topology: &Complex,
    metric: &MeshLengths,
    mass_matrix: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecVector, String> {
    let mass_inverse = build_matern_mass_inverse_1form(topology, metric, mass_matrix, strategy);
    forcing_scale_diag_from_mass_inverse(mass_matrix, &mass_inverse, strategy)
}

pub fn forcing_scale_diag_from_mass_inverse(
    mass_matrix: &FeecCsr,
    mass_inverse: &FeecCsr,
    strategy: MaternMassInverse,
) -> Result<FeecVector, String> {
    match strategy {
        MaternMassInverse::RowSumLumped => Ok(lumped_diag(mass_matrix)),
        MaternMassInverse::Nc1ProjectedSparseInverse => {
            let gmrf = feec_csr_to_gmrf(mass_inverse);
            let mut solver = Solver::default();
            let diag = solver.selected_inverse_diag(&gmrf).map_err(|err| {
                format!("failed to compute projected inverse scaling diagonal: {err}")
            })?;
            Ok(FeecVector::from_vec(diag.iter().copied().collect()))
        }
    }
}

pub fn build_harmonic_orthogonality_constraints(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> Result<GmrfDenseMatrix, String> {
    if mass_u.nrows() != mass_u.ncols() {
        return Err("mass matrix must be square".to_string());
    }
    if harmonic_basis.nrows() != mass_u.ncols() {
        return Err(format!(
            "harmonic basis row count {} does not match mass dimension {}",
            harmonic_basis.nrows(),
            mass_u.ncols()
        ));
    }

    let weighted_basis = mass_u * harmonic_basis;
    Ok(GmrfDenseMatrix::from_fn(
        harmonic_basis.ncols(),
        harmonic_basis.nrows(),
        |i, j| weighted_basis[(j, i)],
    ))
}

pub fn estimate_constrained_mc_variances(
    prior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
    num_samples: usize,
    rng_seed: u64,
) -> Result<GmrfVector, String> {
    if num_samples == 0 {
        return Err("num_samples must be >= 1".to_string());
    }
    if constraint_matrix.ncols() != prior.dimension() {
        return Err(format!(
            "constraint matrix columns {} must match latent dimension {}",
            constraint_matrix.ncols(),
            prior.dimension()
        ));
    }
    if constraint_matrix.nrows() != constraint_rhs.len() {
        return Err(format!(
            "constraint rhs length {} must match constraint rows {}",
            constraint_rhs.len(),
            constraint_matrix.nrows()
        ));
    }

    let dim = prior.dimension();
    let mut mean = vec![0.0; dim];
    let mut m2 = vec![0.0; dim];
    let mut count = 0.0;

    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    for _ in 0..num_samples {
        let sample = if constraint_matrix.nrows() == 0 {
            prior
                .sample_one_solve(&mut rng)
                .map_err(|err| format!("failed to sample unconstrained draw: {err}"))?
        } else {
            prior
                .sample_constrained(constraint_matrix, constraint_rhs, &mut rng)
                .map_err(|err| format!("failed to sample constrained draw: {err}"))?
        };
        count += 1.0;

        for i in 0..dim {
            let delta = sample[i] - mean[i];
            mean[i] += delta / count;
            let delta2 = sample[i] - mean[i];
            m2[i] += delta * delta2;
        }
    }

    Ok(GmrfVector::from_iterator(
        dim,
        (0..dim).map(|i| m2[i] / count),
    ))
}

pub fn compute_constrained_edge_variance_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    mass_matrix: &FeecCsr,
    prior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
    forcing_scale_diag: &FeecVector,
    num_samples: usize,
    rng_seed: u64,
) -> Result<EdgeVarianceDiagnostics, String> {
    let constrained_variances = estimate_constrained_mc_variances(
        prior,
        constraint_matrix,
        constraint_rhs,
        num_samples,
        rng_seed,
    )?;
    compute_edge_variance_diagnostics(
        topology,
        coords,
        mass_matrix,
        &constrained_variances,
        forcing_scale_diag,
    )
}

pub fn compute_exact_constrained_edge_variance_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    mass_matrix: &FeecCsr,
    prior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    forcing_scale_diag: &FeecVector,
) -> Result<(EdgeVarianceDiagnostics, HarmonicAttributionDiagnostics), String> {
    let decomposition = prior
        .exact_constrained_variance_decomposition(constraint_matrix)
        .map_err(|err| {
            format!("failed to compute exact constrained variance decomposition: {err}")
        })?;

    let constrained_edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        mass_matrix,
        &decomposition.constrained_diag,
        forcing_scale_diag,
    )?;
    let removed_edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        mass_matrix,
        &decomposition.removed_diag,
        forcing_scale_diag,
    )?;
    let removed_fraction = FeecVector::from_iterator(
        decomposition.unconstrained_diag.len(),
        (0..decomposition.unconstrained_diag.len()).map(|i| {
            safe_div(
                decomposition.removed_diag[i],
                decomposition.unconstrained_diag[i],
            )
        }),
    );

    Ok((
        constrained_edge_diagnostics,
        HarmonicAttributionDiagnostics {
            removed_edge_diagnostics,
            removed_fraction,
        },
    ))
}

pub fn standardized_forcing_from_sample(
    system_matrix: &FeecCsr,
    forcing_scale_diag: &FeecVector,
    sample: &GmrfVector,
) -> Result<FeecVector, String> {
    if system_matrix.nrows() != system_matrix.ncols() {
        return Err("system matrix must be square".to_string());
    }
    if forcing_scale_diag.len() != system_matrix.nrows() {
        return Err(format!(
            "forcing scale length {} does not match matrix size {}",
            forcing_scale_diag.len(),
            system_matrix.nrows()
        ));
    }
    if sample.len() != system_matrix.ncols() {
        return Err(format!(
            "sample length {} does not match matrix size {}",
            sample.len(),
            system_matrix.ncols()
        ));
    }

    let sample_feec = FeecVector::from_vec(sample.iter().copied().collect());
    let rhs = system_matrix * sample_feec;
    Ok(FeecVector::from_iterator(
        rhs.len(),
        (0..rhs.len()).map(|i| safe_div(rhs[i], safe_sqrt(forcing_scale_diag[i]))),
    ))
}

pub fn estimate_standardized_forcing_diagnostics(
    prior: &Gmrf,
    system_matrix: &FeecCsr,
    forcing_scale_diag: &FeecVector,
    num_samples: usize,
    rng_seed: u64,
) -> Result<StandardizedForcingDiagnostics, String> {
    if num_samples == 0 {
        return Err("num_samples must be >= 1".to_string());
    }

    let dim = system_matrix.nrows();
    let mut mean = vec![0.0; dim];
    let mut m2 = vec![0.0; dim];
    let mut count = 0.0;

    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    for _ in 0..num_samples {
        let sample = prior
            .sample_one_solve(&mut rng)
            .map_err(|err| format!("failed to sample from prior: {err}"))?;
        let standardized =
            standardized_forcing_from_sample(system_matrix, forcing_scale_diag, &sample)?;
        count += 1.0;

        for i in 0..dim {
            let delta = standardized[i] - mean[i];
            mean[i] += delta / count;
            let delta2 = standardized[i] - mean[i];
            m2[i] += delta * delta2;
        }
    }

    let variances = FeecVector::from_iterator(dim, (0..dim).map(|i| m2[i] / count));
    let std_devs = variances.map(|v| v.max(0.0).sqrt());
    Ok(StandardizedForcingDiagnostics {
        mean: FeecVector::from_vec(mean),
        variances,
        std_devs,
    })
}

pub fn summarize_strategy_metrics(
    diagnostics: &EdgeVarianceDiagnostics,
    standardized: &StandardizedForcingDiagnostics,
    midpoint_rho: &FeecVector,
) -> StrategyMetricSummary {
    let edge_length_sq = diagnostics
        .edge_lengths
        .component_mul(&diagnostics.edge_lengths);
    StrategyMetricSummary {
        variance_per_forcing_scale_diag: summarize_vector(
            &diagnostics.variance_per_forcing_scale_diag,
        ),
        standardized_forcing_mean: summarize_vector(&standardized.mean),
        abs_standardized_forcing_mean: summarize_vector(&standardized.mean.map(|v| v.abs())),
        standardized_forcing_variance: summarize_vector(&standardized.variances),
        corr_variance_per_forcing_scale_diag_midpoint_rho: pearson_correlation(
            &diagnostics.variance_per_forcing_scale_diag,
            midpoint_rho,
        ),
        corr_standardized_forcing_variance_midpoint_rho: pearson_correlation(
            &standardized.variances,
            midpoint_rho,
        ),
        corr_standardized_forcing_variance_edge_length2: pearson_correlation(
            &standardized.variances,
            &edge_length_sq,
        ),
    }
}

pub fn compute_matern_1form_strategy_diagnostics(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
    num_samples: usize,
    rng_seed: u64,
) -> Result<MaternStrategyDiagnosticsReport, String> {
    compute_matern_1form_strategy_diagnostics_with_constraints(
        topology,
        coords,
        metric,
        hodge,
        config,
        num_samples,
        rng_seed,
        None,
    )
}

pub fn compute_matern_1form_strategy_diagnostics_with_constraints(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
    num_samples: usize,
    rng_seed: u64,
    constraints: Option<LinearEqualityConstraints<'_>>,
) -> Result<MaternStrategyDiagnosticsReport, String> {
    let prior_precision = build_matern_precision_1form(topology, metric, hodge, config);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);
    let q_prior_factor = q_prior_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let mut prior =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(hodge.mass_u.nrows()), q_prior_gmrf)
            .map_err(|err| format!("failed to build prior: {err}"))?
            .with_precision_sqrt(q_prior_factor);

    let mut variance_rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let prior_variances = prior
        .mc_variances(num_samples, &mut variance_rng)
        .map_err(|err| format!("failed to estimate prior variances: {err}"))?;

    let mass_inverse =
        build_matern_mass_inverse_1form(topology, metric, &hodge.mass_u, config.mass_inverse);
    let forcing_scale_diag =
        forcing_scale_diag_from_mass_inverse(&hodge.mass_u, &mass_inverse, config.mass_inverse)?;

    let edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        &hodge.mass_u,
        &prior_variances,
        &forcing_scale_diag,
    )?;
    let (constrained_edge_diagnostics, harmonic_attribution) = match constraints {
        Some(constraints) => {
            let (constrained_edge_diagnostics, harmonic_attribution) =
                compute_exact_constrained_edge_variance_diagnostics(
                    topology,
                    coords,
                    &hodge.mass_u,
                    &mut prior,
                    constraints.matrix,
                    &forcing_scale_diag,
                )?;
            (
                Some(constrained_edge_diagnostics),
                Some(harmonic_attribution),
            )
        }
        None => (None, None),
    };

    let system_matrix = build_matern_system_matrix_1form(hodge, config.kappa);
    let standardized_forcing = estimate_standardized_forcing_diagnostics(
        &prior,
        &system_matrix,
        &forcing_scale_diag,
        num_samples,
        rng_seed.wrapping_add(1_000_003),
    )?;

    let midpoint_rho = edge_midpoint_rho(topology, coords);
    let metric_summary =
        summarize_strategy_metrics(&edge_diagnostics, &standardized_forcing, &midpoint_rho);

    Ok(MaternStrategyDiagnosticsReport {
        mass_inverse: config.mass_inverse,
        prior_precision,
        forcing_scale_diag,
        edge_diagnostics,
        constrained_edge_diagnostics,
        harmonic_attribution,
        standardized_forcing,
        metric_summary,
    })
}

fn compute_torus_variance_base_estimates(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
    num_rbmc_probes: usize,
    rng_seed: u64,
    rbmc_batch_count: usize,
    constraints: LinearEqualityConstraints<'_>,
) -> Result<TorusVarianceBaseEstimates, String> {
    if num_rbmc_probes == 0 {
        return Err("num_rbmc_probes must be >= 1".to_string());
    }
    if rbmc_batch_count == 0 {
        return Err("rbmc_batch_count must be >= 1".to_string());
    }

    let prior_precision = build_matern_precision_1form(topology, metric, hodge, config);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);
    let q_prior_factor = q_prior_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let mut prior =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(hodge.mass_u.nrows()), q_prior_gmrf)
            .map_err(|err| format!("failed to build prior: {err}"))?
            .with_precision_sqrt(q_prior_factor);

    let mass_inverse =
        build_matern_mass_inverse_1form(topology, metric, &hodge.mass_u, config.mass_inverse);
    let forcing_scale_diag =
        forcing_scale_diag_from_mass_inverse(&hodge.mass_u, &mass_inverse, config.mass_inverse)?;
    let geometry = build_torus_edge_geometry(topology, coords)?;

    let rbmc_batch_sizes = rbmc_batch_sizes(num_rbmc_probes, rbmc_batch_count);
    let mut rbmc_floor_hits = 0_usize;
    let mut batch_variances = Vec::with_capacity(rbmc_batch_sizes.len());
    for (batch_idx, batch_size) in rbmc_batch_sizes.iter().copied().enumerate() {
        let batch_seed = rng_seed.wrapping_add(
            0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((batch_idx as u64).wrapping_add(1)),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(batch_seed);
        let batch_raw = prior
            .rbmc_variances(batch_size, &mut rng)
            .map_err(|err| format!("failed to estimate RBMC variances: {err}"))?;
        let (batch_stabilized, hits) = stabilize_positive_variances(&batch_raw);
        rbmc_floor_hits += hits;
        batch_variances.push(batch_stabilized);
    }

    let prior_variances = weighted_average_vectors(&batch_variances, &rbmc_batch_sizes)?;
    let edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        &hodge.mass_u,
        &prior_variances,
        &forcing_scale_diag,
    )?;

    let removed_diag = prior
        .constrained_variance_correction_diag(constraints.matrix)
        .map_err(|err| format!("failed to compute harmonic variance correction: {err}"))?;
    let harmonic_removed_edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        &hodge.mass_u,
        &removed_diag,
        &forcing_scale_diag,
    )?;
    let (harmonic_free_variances, harmonic_free_floor_hits) =
        stabilize_harmonic_free_variances(&prior_variances, &removed_diag);
    let harmonic_free_edge_diagnostics = compute_edge_variance_diagnostics(
        topology,
        coords,
        &hodge.mass_u,
        &harmonic_free_variances,
        &forcing_scale_diag,
    )?;
    let harmonic_removed_fraction = FeecVector::from_iterator(
        prior_variances.len(),
        (0..prior_variances.len()).map(|i| {
            safe_div(
                removed_diag[i],
                removed_diag[i] + harmonic_free_variances[i].max(0.0),
            )
        }),
    );

    let mut batch_harmonic_free_floor_hits = 0_usize;
    let mut batch_harmonic_free_variances = Vec::with_capacity(batch_variances.len());
    for batch_variance in &batch_variances {
        let (batch_harmonic_free, hits) =
            stabilize_harmonic_free_variances(batch_variance, &removed_diag);
        batch_harmonic_free_floor_hits += hits;
        batch_harmonic_free_variances.push(batch_harmonic_free);
    }

    Ok(TorusVarianceBaseEstimates {
        prior,
        prior_precision,
        forcing_scale_diag,
        geometry,
        edge_diagnostics,
        harmonic_free_edge_diagnostics,
        harmonic_removed_edge_diagnostics,
        harmonic_removed_fraction,
        prior_variances,
        harmonic_free_variances,
        batch_variances,
        batch_harmonic_free_variances,
        rbmc_batch_sizes,
        rbmc_floor_hits,
        harmonic_free_floor_hits: harmonic_free_floor_hits + batch_harmonic_free_floor_hits,
    })
}

pub fn compute_matern_1form_torus_inhomogeneity_report(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
    num_rbmc_probes: usize,
    rng_seed: u64,
    rbmc_batch_count: usize,
    constraints: LinearEqualityConstraints<'_>,
) -> Result<Matern1FormTorusAttributionReport, String> {
    let base = compute_torus_variance_base_estimates(
        topology,
        coords,
        metric,
        hodge,
        config,
        num_rbmc_probes,
        rng_seed,
        rbmc_batch_count,
        constraints,
    )?;
    let field_decomposition = fit_inhomogeneity_field_decomposition(
        &base.prior_variances,
        &base.harmonic_free_variances,
        &base.edge_diagnostics.edge_lengths,
        &base.geometry,
    )?;
    let batch_contributions = base
        .batch_variances
        .iter()
        .zip(base.batch_harmonic_free_variances.iter())
        .map(|(batch_variance, batch_harmonic_free)| {
            estimate_inhomogeneity_contributions(
                batch_variance,
                batch_harmonic_free,
                &base.edge_diagnostics.edge_lengths,
                &base.geometry,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contribution_summary = summarize_inhomogeneity_contributions(&batch_contributions)?;

    let system_matrix = build_matern_system_matrix_1form(hodge, config.kappa);
    let standardized_forcing = estimate_standardized_forcing_diagnostics(
        &base.prior,
        &system_matrix,
        &base.forcing_scale_diag,
        num_rbmc_probes,
        rng_seed.wrapping_add(1_000_003),
    )?;

    Ok(Matern1FormTorusAttributionReport {
        mass_inverse: config.mass_inverse,
        prior_precision: base.prior_precision,
        forcing_scale_diag: base.forcing_scale_diag,
        edge_diagnostics: base.edge_diagnostics,
        torus_attribution: TorusVarianceAttributionDiagnostics {
            harmonic_free_edge_diagnostics: base.harmonic_free_edge_diagnostics,
            harmonic_removed_edge_diagnostics: base.harmonic_removed_edge_diagnostics,
            harmonic_removed_fraction: base.harmonic_removed_fraction,
            geometry: base.geometry,
            field_decomposition,
            contribution_summary,
            batch_contributions,
            rbmc_batch_sizes: base.rbmc_batch_sizes,
            rbmc_floor_hits: base.rbmc_floor_hits,
            harmonic_free_floor_hits: base.harmonic_free_floor_hits,
        },
        standardized_forcing,
    })
}

pub fn compute_matern_1form_torus_residual_study_report(
    topology: &Complex,
    coords: &MeshCoords,
    metric: &MeshLengths,
    hodge: &HodgeLaplacian1Form,
    config: MaternConfig,
    num_rbmc_probes: usize,
    rng_seed: u64,
    rbmc_batch_count: usize,
    constraints: LinearEqualityConstraints<'_>,
) -> Result<Matern1FormTorusResidualStudyReport, String> {
    let base = compute_torus_variance_base_estimates(
        topology,
        coords,
        metric,
        hodge,
        config,
        num_rbmc_probes,
        rng_seed,
        rbmc_batch_count,
        constraints,
    )?;
    let field_decomposition = fit_residual_study_field_decomposition(
        &base.harmonic_free_variances,
        &base.edge_diagnostics,
        &base.geometry,
    )?;
    let batch_contributions = base
        .batch_harmonic_free_variances
        .iter()
        .map(|batch_harmonic_free| {
            estimate_residual_study_contributions(
                batch_harmonic_free,
                &base.edge_diagnostics,
                &base.geometry,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let contribution_summary = summarize_residual_study_contributions(&batch_contributions)?;

    Ok(Matern1FormTorusResidualStudyReport {
        mass_inverse: config.mass_inverse,
        prior_precision: base.prior_precision,
        forcing_scale_diag: base.forcing_scale_diag,
        edge_diagnostics: base.edge_diagnostics,
        harmonic_free_edge_diagnostics: base.harmonic_free_edge_diagnostics,
        harmonic_removed_edge_diagnostics: base.harmonic_removed_edge_diagnostics,
        harmonic_removed_fraction: base.harmonic_removed_fraction,
        geometry: base.geometry,
        field_decomposition,
        contribution_summary,
        batch_contributions,
        rbmc_batch_sizes: base.rbmc_batch_sizes,
        rbmc_floor_hits: base.rbmc_floor_hits,
        harmonic_free_floor_hits: base.harmonic_free_floor_hits,
    })
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

fn estimate_inhomogeneity_contributions(
    unconstrained_variances: &GmrfVector,
    harmonic_free_variances: &GmrfVector,
    edge_lengths: &FeecVector,
    geometry: &TorusEdgeGeometry,
) -> Result<InhomogeneityContributions, String> {
    let unconstrained = FeecVector::from_vec(unconstrained_variances.iter().copied().collect());
    let harmonic_free = FeecVector::from_vec(harmonic_free_variances.iter().copied().collect());
    let total = log_field_variance(&unconstrained)?;
    let harmonic_free_total = log_field_variance(&harmonic_free)?;
    let post_length = log_field_variance(
        &harmonic_free.component_div(&edge_lengths.component_mul(edge_lengths)),
    )?;

    let log_u = log_vector(&FeecVector::from_iterator(
        harmonic_free.len(),
        (0..harmonic_free.len())
            .map(|i| safe_div(harmonic_free[i], edge_lengths[i] * edge_lengths[i])),
    ))?;

    let block_columns = build_inhomogeneity_blocks(geometry);
    let shapley = shapley_block_contributions(&log_u, &block_columns)?;
    let explained_full =
        explained_variance_for_mask(&log_u, &block_columns, (1_usize << block_columns.len()) - 1)?;
    let residual = (post_length - explained_full).max(0.0);

    Ok(InhomogeneityContributions {
        total,
        harmonic: total - harmonic_free_total,
        edge_length: harmonic_free_total - post_length,
        minor_angle: shapley[0],
        direction: shapley[1],
        interaction: shapley[2],
        residual,
    })
}

fn fit_inhomogeneity_field_decomposition(
    unconstrained_variances: &GmrfVector,
    harmonic_free_variances: &GmrfVector,
    edge_lengths: &FeecVector,
    geometry: &TorusEdgeGeometry,
) -> Result<InhomogeneityFieldDecomposition, String> {
    let unconstrained = FeecVector::from_vec(unconstrained_variances.iter().copied().collect());
    let harmonic_free = FeecVector::from_vec(harmonic_free_variances.iter().copied().collect());
    let log_unconstrained_variance = log_vector(&unconstrained)?;
    let log_harmonic_free_variance = log_vector(&harmonic_free)?;
    let log_harmonic_free_variance_per_length2 = log_vector(&FeecVector::from_iterator(
        harmonic_free.len(),
        (0..harmonic_free.len())
            .map(|i| safe_div(harmonic_free[i], edge_lengths[i] * edge_lengths[i])),
    ))?;
    let block_columns = build_inhomogeneity_blocks(geometry);
    let fitted = fit_block_model(&log_harmonic_free_variance_per_length2, &block_columns)?;

    Ok(InhomogeneityFieldDecomposition {
        log_unconstrained_variance,
        log_harmonic_free_variance,
        log_harmonic_free_variance_per_length2,
        minor_angle_component: fitted.block_components[0].clone(),
        direction_component: fitted.block_components[1].clone(),
        interaction_component: fitted.block_components[2].clone(),
        residual_component: fitted.residual,
    })
}

fn estimate_residual_study_contributions(
    harmonic_free_variances: &GmrfVector,
    edge_diagnostics: &EdgeVarianceDiagnostics,
    geometry: &TorusEdgeGeometry,
) -> Result<ResidualStudyContributions, String> {
    let harmonic_free = FeecVector::from_vec(harmonic_free_variances.iter().copied().collect());
    let response = log_vector(
        &harmonic_free.component_div(
            &edge_diagnostics
                .edge_lengths
                .component_mul(&edge_diagnostics.edge_lengths),
        ),
    )?;
    let block_columns = build_residual_study_blocks(edge_diagnostics, geometry)?;
    let shapley = shapley_block_contributions(&response, &block_columns)?;
    let explained_full = explained_variance_for_mask(
        &response,
        &block_columns,
        (1_usize << block_columns.len()) - 1,
    )?;
    let total_post_length = log_field_variance(
        &harmonic_free.component_div(
            &edge_diagnostics
                .edge_lengths
                .component_mul(&edge_diagnostics.edge_lengths),
        ),
    )?;

    Ok(ResidualStudyContributions {
        total_post_length,
        position_even_fourier: shapley[0],
        direction_legendre: shapley[1],
        interaction_even: shapley[2],
        discrete_surrogates: shapley[3],
        unexplained: (total_post_length - explained_full).max(0.0),
    })
}

fn fit_residual_study_field_decomposition(
    harmonic_free_variances: &GmrfVector,
    edge_diagnostics: &EdgeVarianceDiagnostics,
    geometry: &TorusEdgeGeometry,
) -> Result<ResidualStudyFieldDecomposition, String> {
    let harmonic_free = FeecVector::from_vec(harmonic_free_variances.iter().copied().collect());
    let response = log_vector(
        &harmonic_free.component_div(
            &edge_diagnostics
                .edge_lengths
                .component_mul(&edge_diagnostics.edge_lengths),
        ),
    )?;
    let block_columns = build_residual_study_blocks(edge_diagnostics, geometry)?;
    let fitted = fit_block_model(&response, &block_columns)?;

    Ok(ResidualStudyFieldDecomposition {
        log_harmonic_free_variance_per_length2: response,
        position_even_fourier_component: fitted.block_components[0].clone(),
        direction_legendre_component: fitted.block_components[1].clone(),
        interaction_even_component: fitted.block_components[2].clone(),
        discrete_surrogate_component: fitted.block_components[3].clone(),
        unexplained_residual: fitted.residual,
    })
}

fn build_inhomogeneity_blocks(geometry: &TorusEdgeGeometry) -> Vec<Vec<FeecVector>> {
    let cos_theta = geometry.midpoint_theta.map(|theta| theta.cos());
    let cos_two_theta = geometry.midpoint_theta.map(|theta| (2.0 * theta).cos());
    let direction = geometry.toroidal_alignment_sq.clone();
    let interaction_cos_theta = direction.component_mul(&cos_theta);
    let interaction_cos_two_theta = direction.component_mul(&cos_two_theta);
    vec![
        vec![cos_theta, cos_two_theta],
        vec![direction],
        vec![interaction_cos_theta, interaction_cos_two_theta],
    ]
}

fn build_residual_study_blocks(
    edge_diagnostics: &EdgeVarianceDiagnostics,
    geometry: &TorusEdgeGeometry,
) -> Result<Vec<Vec<FeecVector>>, String> {
    let theta = &geometry.midpoint_theta;
    let toroidal_alignment_sq = &geometry.toroidal_alignment_sq;
    let s = toroidal_alignment_sq.map(|value| 2.0 * value - 1.0);
    let p1 = s.clone();
    let p2 = s.map(|value| 0.5 * (3.0 * value * value - 1.0));
    let p3 = s.map(|value| 0.5 * (5.0 * value.powi(3) - 3.0 * value));

    let position_even_fourier = (1..=6)
        .map(|k| theta.map(|angle| (k as f64 * angle).cos()))
        .collect::<Vec<_>>();
    let direction_legendre = vec![p1.clone(), p2.clone(), p3];
    let interaction_even = (1..=3)
        .flat_map(|k| {
            let cos_k_theta = theta.map(|angle| (k as f64 * angle).cos());
            [
                p1.component_mul(&cos_k_theta),
                p2.component_mul(&cos_k_theta),
            ]
        })
        .collect::<Vec<_>>();
    let length_sq = edge_diagnostics
        .edge_lengths
        .component_mul(&edge_diagnostics.edge_lengths);
    let discrete_surrogates = vec![
        log_vector(&edge_diagnostics.mass_diag.component_div(&length_sq))?,
        log_vector(&edge_diagnostics.mass_lumped_diag.component_div(&length_sq))?,
        log_vector(
            &edge_diagnostics
                .forcing_scale_diag
                .component_div(&length_sq),
        )?,
    ];

    Ok(vec![
        position_even_fourier,
        direction_legendre,
        interaction_even,
        discrete_surrogates,
    ])
}

fn log_field_variance(values: &FeecVector) -> Result<f64, String> {
    let logged = log_vector(values)?;
    let stats = summarize_vector(&logged).ok_or_else(|| "empty field".to_string())?;
    Ok(stats.std * stats.std)
}

fn log_vector(values: &FeecVector) -> Result<FeecVector, String> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("log-variance metrics require finite positive values".to_string());
    }
    Ok(values.map(|value| value.ln()))
}

fn explained_variance_for_mask(
    y: &FeecVector,
    block_columns: &[Vec<FeecVector>],
    mask: usize,
) -> Result<f64, String> {
    if mask == 0 {
        return Ok(0.0);
    }
    let mut columns: Vec<&FeecVector> = Vec::new();
    for (block_idx, block) in block_columns.iter().enumerate() {
        if (mask & (1 << block_idx)) != 0 {
            for column in block {
                columns.push(column);
            }
        }
    }
    fit_explained_variance(y, &columns)
}

fn fit_explained_variance(y: &FeecVector, columns: &[&FeecVector]) -> Result<f64, String> {
    let fit = fit_linear_model(y, columns)?;
    let total = summarize_vector(y)
        .ok_or_else(|| "empty response".to_string())?
        .std
        .powi(2);
    let residual_var = summarize_vector(&fit.residual)
        .ok_or_else(|| "empty residual".to_string())?
        .std
        .powi(2);
    Ok((total - residual_var).max(0.0))
}

struct LinearModelFit {
    residual: FeecVector,
    coefficients: FeecVector,
}

fn fit_linear_model(y: &FeecVector, columns: &[&FeecVector]) -> Result<LinearModelFit, String> {
    if y.is_empty() {
        return Err("cannot fit a linear model for an empty response".to_string());
    }
    if columns.iter().any(|column| column.len() != y.len()) {
        return Err("all regression columns must match response length".to_string());
    }

    let n = y.len();
    let p = columns.len() + 1;
    let mut design = FeecMatrix::zeros(n, p);
    for i in 0..n {
        design[(i, 0)] = 1.0;
    }
    for (j, column) in columns.iter().enumerate() {
        for i in 0..n {
            design[(i, j + 1)] = column[i];
        }
    }

    let pinv = design
        .clone()
        .pseudo_inverse(1e-12)
        .map_err(|_| "failed to compute pseudo-inverse for attribution fit".to_string())?;
    let beta = &pinv * y;
    let fitted = &design * &beta;
    Ok(LinearModelFit {
        residual: y.clone() - fitted,
        coefficients: beta,
    })
}

struct BlockModelFit {
    block_components: Vec<FeecVector>,
    residual: FeecVector,
}

fn fit_block_model(
    y: &FeecVector,
    block_columns: &[Vec<FeecVector>],
) -> Result<BlockModelFit, String> {
    let mut column_refs: Vec<&FeecVector> = Vec::new();
    for block in block_columns {
        for column in block {
            column_refs.push(column);
        }
    }
    let fit = fit_linear_model(y, &column_refs)?;
    let mut coeff_idx = 1_usize;
    let mut block_components = Vec::with_capacity(block_columns.len());
    for block in block_columns {
        let mut component = FeecVector::zeros(y.len());
        for column in block {
            let coeff = fit.coefficients[coeff_idx];
            coeff_idx += 1;
            component += coeff * column;
        }
        block_components.push(component);
    }

    Ok(BlockModelFit {
        block_components,
        residual: fit.residual,
    })
}

fn shapley_block_contributions(
    y: &FeecVector,
    block_columns: &[Vec<FeecVector>],
) -> Result<Vec<f64>, String> {
    let nblocks = block_columns.len();
    if nblocks == 0 {
        return Ok(Vec::new());
    }
    if nblocks >= usize::BITS as usize {
        return Err("too many attribution blocks".to_string());
    }

    let mut values = vec![0.0_f64; 1 << nblocks];
    for mask in 1..values.len() {
        values[mask] = explained_variance_for_mask(y, block_columns, mask)?;
    }

    let full_factorial = factorial(nblocks);
    let mut out = vec![0.0_f64; nblocks];
    for block_idx in 0..nblocks {
        for mask in 0..values.len() {
            if (mask & (1 << block_idx)) != 0 {
                continue;
            }
            let subset_size = mask.count_ones() as usize;
            let weight =
                factorial(subset_size) * factorial(nblocks - subset_size - 1) / full_factorial;
            out[block_idx] += weight * (values[mask | (1 << block_idx)] - values[mask]);
        }
    }
    Ok(out)
}

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0_f64, |acc, value| acc * value as f64)
}

fn summarize_inhomogeneity_contributions(
    batches: &[InhomogeneityContributions],
) -> Result<InhomogeneityContributionSummary, String> {
    if batches.is_empty() {
        return Err("at least one RBMC batch is required".to_string());
    }

    let totals = batches.iter().map(|batch| batch.total).collect::<Vec<_>>();
    let harmonics = batches
        .iter()
        .map(|batch| batch.harmonic)
        .collect::<Vec<_>>();
    let lengths = batches
        .iter()
        .map(|batch| batch.edge_length)
        .collect::<Vec<_>>();
    let minors = batches
        .iter()
        .map(|batch| batch.minor_angle)
        .collect::<Vec<_>>();
    let directions = batches
        .iter()
        .map(|batch| batch.direction)
        .collect::<Vec<_>>();
    let interactions = batches
        .iter()
        .map(|batch| batch.interaction)
        .collect::<Vec<_>>();
    let residuals = batches
        .iter()
        .map(|batch| batch.residual)
        .collect::<Vec<_>>();

    Ok(InhomogeneityContributionSummary {
        total: summarize_scalar_estimate(&totals),
        harmonic: summarize_contribution_estimate(&harmonics, &totals),
        edge_length: summarize_contribution_estimate(&lengths, &totals),
        minor_angle: summarize_contribution_estimate(&minors, &totals),
        direction: summarize_contribution_estimate(&directions, &totals),
        interaction: summarize_contribution_estimate(&interactions, &totals),
        residual: summarize_contribution_estimate(&residuals, &totals),
    })
}

fn summarize_residual_study_contributions(
    batches: &[ResidualStudyContributions],
) -> Result<ResidualStudyContributionSummary, String> {
    if batches.is_empty() {
        return Err("at least one RBMC batch is required".to_string());
    }

    let totals = batches
        .iter()
        .map(|batch| batch.total_post_length)
        .collect::<Vec<_>>();
    let positions = batches
        .iter()
        .map(|batch| batch.position_even_fourier)
        .collect::<Vec<_>>();
    let directions = batches
        .iter()
        .map(|batch| batch.direction_legendre)
        .collect::<Vec<_>>();
    let interactions = batches
        .iter()
        .map(|batch| batch.interaction_even)
        .collect::<Vec<_>>();
    let surrogates = batches
        .iter()
        .map(|batch| batch.discrete_surrogates)
        .collect::<Vec<_>>();
    let unexplained = batches
        .iter()
        .map(|batch| batch.unexplained)
        .collect::<Vec<_>>();

    Ok(ResidualStudyContributionSummary {
        total_post_length: summarize_scalar_estimate(&totals),
        position_even_fourier: summarize_contribution_estimate(&positions, &totals),
        direction_legendre: summarize_contribution_estimate(&directions, &totals),
        interaction_even: summarize_contribution_estimate(&interactions, &totals),
        discrete_surrogates: summarize_contribution_estimate(&surrogates, &totals),
        unexplained: summarize_contribution_estimate(&unexplained, &totals),
    })
}

fn summarize_scalar_estimate(values: &[f64]) -> EstimateWithError {
    if values.is_empty() {
        return EstimateWithError {
            mean: 0.0,
            standard_error: 0.0,
        };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() == 1 {
        return EstimateWithError {
            mean,
            standard_error: 0.0,
        };
    }
    let sample_var = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    EstimateWithError {
        mean,
        standard_error: (sample_var / values.len() as f64).sqrt(),
    }
}

fn summarize_contribution_estimate(values: &[f64], totals: &[f64]) -> ContributionEstimate {
    let fractions = values
        .iter()
        .zip(totals.iter())
        .map(|(value, total)| safe_div(*value, *total))
        .collect::<Vec<_>>();
    ContributionEstimate {
        absolute: summarize_scalar_estimate(values),
        fraction_of_total: summarize_scalar_estimate(&fractions),
    }
}

pub fn compute_matern_0form_diagnostics(
    coords: &MeshCoords,
    laplace: &LaplaceBeltrami0Form,
    config: Matern0FormConfig,
    num_samples: usize,
    rng_seed: u64,
) -> Result<Matern0FormDiagnosticsReport, String> {
    let prior_precision = build_matern_precision_0form(laplace, config);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);
    let q_prior_factor = q_prior_gmrf
        .cholesky_sqrt_lower()
        .map_err(|err| format!("failed to factor prior precision: {err}"))?;
    let mut prior =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(laplace.mass.nrows()), q_prior_gmrf)
            .map_err(|err| format!("failed to build prior: {err}"))?
            .with_precision_sqrt(q_prior_factor);

    let mut variance_rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let prior_variances = prior
        .mc_variances(num_samples, &mut variance_rng)
        .map_err(|err| format!("failed to estimate prior variances: {err}"))?;

    let forcing_scale_diag =
        forcing_scale_diag_from_0form_strategy(&laplace.mass, config.mass_inverse);

    let node_diagnostics = compute_node_variance_diagnostics(
        coords,
        &laplace.mass,
        &prior_variances,
        &forcing_scale_diag,
    )?;

    let system_matrix = build_matern_system_matrix_0form(laplace, config.kappa);
    let standardized_forcing = estimate_standardized_forcing_diagnostics(
        &prior,
        &system_matrix,
        &forcing_scale_diag,
        num_samples,
        rng_seed.wrapping_add(1_000_003),
    )?;

    Ok(Matern0FormDiagnosticsReport {
        mass_inverse: config.mass_inverse,
        prior_precision,
        forcing_scale_diag,
        node_diagnostics,
        standardized_forcing,
    })
}

pub fn forcing_scale_diag_from_0form_strategy(
    mass_matrix: &FeecCsr,
    strategy: Matern0FormMassInverse,
) -> FeecVector {
    match strategy {
        Matern0FormMassInverse::RowSumLumped => lumped_diag(mass_matrix),
    }
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den.abs() <= EPS {
        0.0
    } else {
        num / den
    }
}

fn safe_sqrt(value: f64) -> f64 {
    if value <= EPS {
        0.0
    } else {
        value.sqrt()
    }
}

pub fn matrix_diag(mat: &FeecCsr) -> FeecVector {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, col, value) in mat.triplet_iter() {
        if row == col {
            diag[row] += *value;
        }
    }
    FeecVector::from_vec(diag)
}

pub fn lumped_diag(mat: &FeecCsr) -> FeecVector {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    FeecVector::from_vec(diag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::{CooMatrix as FeecCoo, Matrix as FeecMatrix};
    use gmrf_core::types::{CooMatrix as GmrfCoo, SparseMatrix as GmrfSparse};
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::path::PathBuf;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn edge_lengths_are_positive() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let lengths = edge_lengths(&topology, &coords);
        assert!(lengths.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn compute_edge_variance_diagnostics_applies_expected_scalings() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();
        let variances =
            GmrfVector::from_iterator(edge_count, (0..edge_count).map(|i| i as f64 + 1.0));
        let mass = diagonal_matrix(edge_count, 2.5);
        let forcing_scale = FeecVector::from_vec(vec![5.0; edge_count]);

        let diagnostics = compute_edge_variance_diagnostics(
            &topology,
            &coords,
            &mass,
            &variances,
            &forcing_scale,
        )
        .expect("diagnostics should succeed");

        for i in 0..edge_count {
            let len = diagnostics.edge_lengths[i];
            assert!(approx_eq(diagnostics.variances[i], variances[i]));
            assert!(approx_eq(diagnostics.mass_diag[i], 2.5));
            assert!(approx_eq(diagnostics.mass_lumped_diag[i], 2.5));
            assert!(approx_eq(diagnostics.forcing_scale_diag[i], 5.0));
            assert!(approx_eq(
                diagnostics.variance_per_length[i],
                diagnostics.variances[i] / len
            ));
            assert!(approx_eq(
                diagnostics.variance_per_length2[i],
                diagnostics.variances[i] / (len * len)
            ));
            assert!(approx_eq(
                diagnostics.std_per_length[i],
                diagnostics.std_devs[i] / len
            ));
            assert!(approx_eq(
                diagnostics.variance_per_mass_diag[i],
                diagnostics.variances[i] / 2.5
            ));
            assert!(approx_eq(
                diagnostics.variance_per_mass_lumped_diag[i],
                diagnostics.variances[i] / 2.5
            ));
            assert!(approx_eq(
                diagnostics.variance_per_forcing_scale_diag[i],
                diagnostics.variances[i] / 5.0
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_mass_diag[i],
                diagnostics.std_devs[i] / 2.5_f64.sqrt()
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_mass_lumped_diag[i],
                diagnostics.std_devs[i] / 2.5_f64.sqrt()
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_forcing_scale_diag[i],
                diagnostics.std_devs[i] / 5.0_f64.sqrt()
            ));
        }
    }

    #[test]
    fn compute_edge_variance_diagnostics_checks_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();
        let good_mass = diagonal_matrix(edge_count, 1.0);
        let good_scale = FeecVector::from_vec(vec![1.0; edge_count]);

        let err = compute_edge_variance_diagnostics(
            &topology,
            &coords,
            &good_mass,
            &GmrfVector::zeros(1),
            &good_scale,
        )
        .expect_err("dimension mismatch should error");
        assert!(err.contains("variance length"));

        let bad_mass = diagonal_matrix(1, 1.0);
        let err = compute_edge_variance_diagnostics(
            &topology,
            &coords,
            &bad_mass,
            &GmrfVector::zeros(edge_count),
            &good_scale,
        )
        .expect_err("mass size mismatch should error");
        assert!(err.contains("mass matrix shape"));

        let err = compute_edge_variance_diagnostics(
            &topology,
            &coords,
            &good_mass,
            &GmrfVector::zeros(edge_count),
            &FeecVector::from_vec(vec![1.0]),
        )
        .expect_err("forcing scale mismatch should error");
        assert!(err.contains("forcing scale length"));
    }

    #[test]
    fn compute_node_variance_diagnostics_applies_expected_scalings() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (_topology, coords) = mesh.compute_coord_complex();
        let vertex_count = coords.nvertices();
        let variances =
            GmrfVector::from_iterator(vertex_count, (0..vertex_count).map(|i| i as f64 + 1.0));
        let mass = diagonal_matrix(vertex_count, 2.5);
        let forcing_scale = FeecVector::from_vec(vec![5.0; vertex_count]);

        let diagnostics =
            compute_node_variance_diagnostics(&coords, &mass, &variances, &forcing_scale)
                .expect("diagnostics should succeed");

        for i in 0..vertex_count {
            assert!(approx_eq(diagnostics.variances[i], variances[i]));
            assert!(approx_eq(diagnostics.mass_diag[i], 2.5));
            assert!(approx_eq(diagnostics.mass_lumped_diag[i], 2.5));
            assert!(approx_eq(diagnostics.forcing_scale_diag[i], 5.0));
            assert!(approx_eq(
                diagnostics.variance_per_mass_diag[i],
                diagnostics.variances[i] / 2.5
            ));
            assert!(approx_eq(
                diagnostics.variance_per_mass_lumped_diag[i],
                diagnostics.variances[i] / 2.5
            ));
            assert!(approx_eq(
                diagnostics.variance_per_forcing_scale_diag[i],
                diagnostics.variances[i] / 5.0
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_mass_diag[i],
                diagnostics.std_devs[i] / 2.5_f64.sqrt()
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_mass_lumped_diag[i],
                diagnostics.std_devs[i] / 2.5_f64.sqrt()
            ));
            assert!(approx_eq(
                diagnostics.std_per_sqrt_forcing_scale_diag[i],
                diagnostics.std_devs[i] / 5.0_f64.sqrt()
            ));
            assert!(diagnostics.rho[i].is_finite());
            assert!(diagnostics.rho[i] >= 0.0);
        }
    }

    #[test]
    fn compute_node_variance_diagnostics_checks_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (_topology, coords) = mesh.compute_coord_complex();
        let vertex_count = coords.nvertices();
        let good_mass = diagonal_matrix(vertex_count, 1.0);
        let good_scale = FeecVector::from_vec(vec![1.0; vertex_count]);

        let err = compute_node_variance_diagnostics(
            &coords,
            &good_mass,
            &GmrfVector::zeros(1),
            &good_scale,
        )
        .expect_err("dimension mismatch should error");
        assert!(err.contains("variance length"));

        let bad_mass = diagonal_matrix(1, 1.0);
        let err = compute_node_variance_diagnostics(
            &coords,
            &bad_mass,
            &GmrfVector::zeros(vertex_count),
            &good_scale,
        )
        .expect_err("mass size mismatch should error");
        assert!(err.contains("mass matrix shape"));

        let err = compute_node_variance_diagnostics(
            &coords,
            &good_mass,
            &GmrfVector::zeros(vertex_count),
            &FeecVector::from_vec(vec![1.0]),
        )
        .expect_err("forcing scale mismatch should error");
        assert!(err.contains("forcing scale length"));
    }

    #[test]
    fn compute_matern_0form_diagnostics_builds_finite_report() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let laplace = crate::matern_0form::build_laplace_beltrami_0form(&topology, &metric);

        let report = compute_matern_0form_diagnostics(
            &coords,
            &laplace,
            crate::matern_0form::MaternConfig {
                kappa: 2.0,
                tau: 1.0,
                mass_inverse: crate::matern_0form::MaternMassInverse::RowSumLumped,
            },
            8,
            19,
        )
        .expect("0-form diagnostics should build");

        assert_eq!(report.prior_precision.nrows(), coords.nvertices());
        assert_eq!(report.prior_precision.ncols(), coords.nvertices());
        assert_eq!(report.forcing_scale_diag.len(), coords.nvertices());
        assert_eq!(report.node_diagnostics.variances.len(), coords.nvertices());
        assert_eq!(report.standardized_forcing.mean.len(), coords.nvertices());
        assert!(report
            .node_diagnostics
            .variances
            .iter()
            .all(|v| v.is_finite()));
        assert!(report
            .standardized_forcing
            .variances
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn pearson_correlation_detects_linear_relation() {
        let lhs = FeecVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = FeecVector::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let corr = pearson_correlation(&lhs, &rhs).expect("correlation should exist");
        assert!(approx_eq(corr, 1.0));
    }

    #[test]
    fn lumped_and_matrix_diagonal_are_distinct_when_off_diagonal_exists() {
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 3.0);
        coo.push(0, 1, 0.5);
        coo.push(1, 0, 0.5);
        let mat = FeecCsr::from(&coo);

        let diag = matrix_diag(&mat);
        let lumped = lumped_diag(&mat);
        assert!(approx_eq(diag[0], 2.0));
        assert!(approx_eq(diag[1], 3.0));
        assert!(approx_eq(lumped[0], 2.5));
        assert!(approx_eq(lumped[1], 3.5));
    }

    #[test]
    fn standardized_forcing_from_sample_matches_hand_computation() {
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 2.0);
        coo.push(1, 1, 3.0);
        let a = FeecCsr::from(&coo);
        let d = FeecVector::from_vec(vec![4.0, 9.0]);
        let c = GmrfVector::from_vec(vec![1.0, 2.0]);

        let y = standardized_forcing_from_sample(&a, &d, &c).expect("standardization should work");
        assert!(approx_eq(y[0], 1.0));
        assert!(approx_eq(y[1], 2.0));
    }

    #[test]
    fn standardized_forcing_from_sample_checks_dimensions() {
        let mut coo = FeecCoo::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, 1.0);
        let a = FeecCsr::from(&coo);
        let err = standardized_forcing_from_sample(
            &a,
            &FeecVector::from_vec(vec![1.0]),
            &GmrfVector::zeros(2),
        )
        .expect_err("dimension mismatch should error");
        assert!(err.contains("forcing scale length"));
    }

    #[test]
    fn harmonic_orthogonality_constraints_use_mass_inner_product() {
        let mut mass_coo = FeecCoo::new(2, 2);
        mass_coo.push(0, 0, 2.0);
        mass_coo.push(1, 1, 3.0);
        let mass = FeecCsr::from(&mass_coo);
        let basis = FeecMatrix::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 2.0 });

        let constraints =
            build_harmonic_orthogonality_constraints(&basis, &mass).expect("constraints");

        assert_eq!(constraints.nrows(), 1);
        assert_eq!(constraints.ncols(), 2);
        assert!(approx_eq(constraints[(0, 0)], 2.0));
        assert!(approx_eq(constraints[(0, 1)], 6.0));
    }

    #[test]
    fn exact_constrained_edge_variance_diagnostics_match_coordinate_constraint() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();
        let mass = diagonal_matrix(edge_count, 2.5);
        let forcing_scale = FeecVector::from_vec(vec![5.0; edge_count]);

        let mut precision_coo = GmrfCoo::new(edge_count, edge_count);
        for i in 0..edge_count {
            precision_coo.push(i, i, 1.0);
        }
        let precision = GmrfSparse::from(&precision_coo);
        let q_factor = precision
            .cholesky_sqrt_lower()
            .expect("identity precision should factorize");
        let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(edge_count), precision)
            .expect("prior should build")
            .with_precision_sqrt(q_factor);
        let constraints =
            GmrfDenseMatrix::from_fn(1, edge_count, |_, j| if j == 0 { 1.0 } else { 0.0 });

        let (constrained, attribution) = compute_exact_constrained_edge_variance_diagnostics(
            &topology,
            &coords,
            &mass,
            &mut prior,
            &constraints,
            &forcing_scale,
        )
        .expect("exact constrained diagnostics should succeed");

        assert!(approx_eq(constrained.variances[0], 0.0));
        assert!(approx_eq(
            attribution.removed_edge_diagnostics.variances[0],
            1.0
        ));
        assert!(approx_eq(attribution.removed_fraction[0], 1.0));
        for i in 1..edge_count {
            assert!(approx_eq(constrained.variances[i], 1.0));
            assert!(approx_eq(
                attribution.removed_edge_diagnostics.variances[i],
                0.0
            ));
            assert!(approx_eq(attribution.removed_fraction[i], 0.0));
        }
    }

    #[test]
    fn torus_edge_geometry_is_finite_and_bounded() {
        let mesh_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);

        let geometry =
            build_torus_edge_geometry(&topology, &coords).expect("geometry should build");

        assert_eq!(geometry.midpoint_rho.len(), topology.skeleton(1).len());
        assert!(geometry
            .midpoint_rho
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert!(geometry
            .midpoint_theta
            .iter()
            .all(|value| value.is_finite()));
        assert!(geometry
            .gaussian_curvature
            .iter()
            .all(|value| value.is_finite()));
        assert!(geometry
            .toroidal_alignment_sq
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0 + 1e-12));
    }

    #[test]
    fn inhomogeneity_contribution_summary_preserves_component_sum() {
        let batches = vec![
            InhomogeneityContributions {
                total: 10.0,
                harmonic: 1.0,
                edge_length: 2.0,
                minor_angle: 3.0,
                direction: 1.5,
                interaction: 0.5,
                residual: 2.0,
            },
            InhomogeneityContributions {
                total: 12.0,
                harmonic: 1.5,
                edge_length: 2.5,
                minor_angle: 3.5,
                direction: 1.5,
                interaction: 0.5,
                residual: 2.5,
            },
        ];

        let summary =
            summarize_inhomogeneity_contributions(&batches).expect("summary should build");
        let component_sum = summary.harmonic.absolute.mean
            + summary.edge_length.absolute.mean
            + summary.minor_angle.absolute.mean
            + summary.direction.absolute.mean
            + summary.interaction.absolute.mean
            + summary.residual.absolute.mean;

        assert!(approx_eq(component_sum, summary.total.mean));
    }

    #[test]
    fn generic_three_block_shapley_preserves_explained_variance() {
        let x1 = FeecVector::from_vec(vec![-1.0, -0.2, 0.4, 1.1, 1.7]);
        let x2 = FeecVector::from_vec(vec![0.8, -0.5, 1.2, 0.3, -1.0]);
        let x3 = x1.component_mul(&x2);
        let y = FeecVector::from_iterator(
            x1.len(),
            (0..x1.len()).map(|i| 0.7 + 1.2 * x1[i] - 0.4 * x2[i] + 0.9 * x3[i]),
        );
        let blocks = vec![vec![x1], vec![x2], vec![x3]];

        let shapley = shapley_block_contributions(&y, &blocks).expect("shapley should work");
        let explained =
            explained_variance_for_mask(&y, &blocks, (1_usize << blocks.len()) - 1).unwrap();
        let fit = fit_block_model(&y, &blocks).expect("fit should work");
        let residual_var = summarize_vector(&fit.residual).unwrap().std.powi(2);

        assert!(approx_eq(shapley.iter().sum::<f64>(), explained));
        assert!(residual_var <= 1e-20);
    }

    #[test]
    fn residual_study_block_builder_is_finite_and_aligned() {
        let mesh_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh");
        let mesh_bytes = std::fs::read(mesh_path).expect("failed to read torus mesh");
        let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
        let edge_count = topology.skeleton(1).len();
        let geometry =
            build_torus_edge_geometry(&topology, &coords).expect("geometry should build");
        let diagnostics = synthetic_edge_diagnostics(
            edge_lengths(&topology, &coords),
            FeecVector::from_iterator(edge_count, (0..edge_count).map(|i| 1.0 + 0.001 * i as f64)),
            FeecVector::from_iterator(edge_count, (0..edge_count).map(|i| 1.2 + 0.001 * i as f64)),
            FeecVector::from_iterator(edge_count, (0..edge_count).map(|i| 0.9 + 0.001 * i as f64)),
            FeecVector::from_vec(vec![1.0; edge_count]),
        );

        let blocks =
            build_residual_study_blocks(&diagnostics, &geometry).expect("blocks should build");

        assert_eq!(blocks.len(), 4);
        assert!(blocks.iter().all(|block| !block.is_empty()));
        for block in blocks {
            for column in block {
                assert_eq!(column.len(), edge_count);
                assert!(column.iter().all(|value| value.is_finite()));
            }
        }
    }

    #[test]
    fn synthetic_residual_study_fit_recovers_known_structure() {
        let n = 24;
        let geometry = synthetic_torus_edge_geometry(n);
        let edge_lengths =
            FeecVector::from_iterator(n, (0..n).map(|i| 0.35 + 0.01 * (i % 5) as f64));
        let mass_diag = FeecVector::from_iterator(n, (0..n).map(|i| 1.2 + 0.03 * (i % 7) as f64));
        let mass_lumped_diag =
            FeecVector::from_iterator(n, (0..n).map(|i| 1.5 + 0.02 * (i % 5) as f64));
        let forcing_diag =
            FeecVector::from_iterator(n, (0..n).map(|i| 0.8 + 0.025 * (i % 6) as f64));
        let diagnostics = synthetic_edge_diagnostics(
            edge_lengths.clone(),
            mass_diag,
            mass_lumped_diag,
            forcing_diag,
            FeecVector::from_vec(vec![1.0; n]),
        );
        let blocks =
            build_residual_study_blocks(&diagnostics, &geometry).expect("blocks should build");

        let intercept = 0.3;
        let weights = [
            vec![0.5, -0.15, 0.08, -0.04, 0.03, -0.02],
            vec![0.35, -0.1, 0.07],
            vec![0.2, -0.06, 0.04, -0.03, 0.02, -0.01],
            vec![0.45, -0.18, 0.11],
        ];
        let response = FeecVector::from_iterator(
            n,
            (0..n).map(|i| {
                intercept
                    + blocks[0]
                        .iter()
                        .zip(weights[0].iter())
                        .map(|(column, weight)| weight * column[i])
                        .sum::<f64>()
                    + blocks[1]
                        .iter()
                        .zip(weights[1].iter())
                        .map(|(column, weight)| weight * column[i])
                        .sum::<f64>()
                    + blocks[2]
                        .iter()
                        .zip(weights[2].iter())
                        .map(|(column, weight)| weight * column[i])
                        .sum::<f64>()
                    + blocks[3]
                        .iter()
                        .zip(weights[3].iter())
                        .map(|(column, weight)| weight * column[i])
                        .sum::<f64>()
            }),
        );
        let harmonic_free_variances = GmrfVector::from_iterator(
            n,
            (0..n).map(|i| response[i].exp() * edge_lengths[i] * edge_lengths[i]),
        );

        let decomposition = fit_residual_study_field_decomposition(
            &harmonic_free_variances,
            &diagnostics,
            &geometry,
        )
        .expect("residual study decomposition should fit");
        let contributions = estimate_residual_study_contributions(
            &harmonic_free_variances,
            &diagnostics,
            &geometry,
        )
        .expect("residual study contributions should fit");
        let unexplained_var = summarize_vector(&decomposition.unexplained_residual)
            .unwrap()
            .std
            .powi(2);
        let component_sum = contributions.position_even_fourier
            + contributions.direction_legendre
            + contributions.interaction_even
            + contributions.discrete_surrogates
            + contributions.unexplained;

        assert!(unexplained_var <= 1e-18);
        assert!(approx_eq(component_sum, contributions.total_post_length));
    }

    #[test]
    fn residual_study_contribution_summary_preserves_component_sum() {
        let batches = vec![
            ResidualStudyContributions {
                total_post_length: 8.0,
                position_even_fourier: 1.5,
                direction_legendre: 1.0,
                interaction_even: 0.5,
                discrete_surrogates: 2.0,
                unexplained: 3.0,
            },
            ResidualStudyContributions {
                total_post_length: 10.0,
                position_even_fourier: 2.0,
                direction_legendre: 1.5,
                interaction_even: 0.7,
                discrete_surrogates: 2.3,
                unexplained: 3.5,
            },
        ];

        let summary =
            summarize_residual_study_contributions(&batches).expect("summary should build");
        let component_sum = summary.position_even_fourier.absolute.mean
            + summary.direction_legendre.absolute.mean
            + summary.interaction_even.absolute.mean
            + summary.discrete_surrogates.absolute.mean
            + summary.unexplained.absolute.mean;

        assert!(approx_eq(component_sum, summary.total_post_length.mean));
    }

    fn synthetic_torus_edge_geometry(n: usize) -> TorusEdgeGeometry {
        let major_radius = 1.0;
        let minor_radius = 0.3;
        let midpoint_theta = FeecVector::from_iterator(
            n,
            (0..n)
                .map(|i| -std::f64::consts::PI + 2.0 * std::f64::consts::PI * i as f64 / n as f64),
        );
        let midpoint_rho = midpoint_theta.map(|theta| major_radius + minor_radius * theta.cos());
        let gaussian_curvature = FeecVector::from_iterator(
            n,
            (0..n).map(|i| torus_gaussian_curvature(major_radius, minor_radius, midpoint_rho[i])),
        );
        let toroidal_alignment_sq = FeecVector::from_iterator(
            n,
            (0..n).map(|i| (i as f64 / (n.saturating_sub(1).max(1)) as f64).clamp(0.0, 1.0)),
        );
        TorusEdgeGeometry {
            major_radius,
            minor_radius,
            midpoint_rho,
            midpoint_theta,
            gaussian_curvature,
            toroidal_alignment_sq,
        }
    }

    fn synthetic_edge_diagnostics(
        edge_lengths: FeecVector,
        mass_diag: FeecVector,
        mass_lumped_diag: FeecVector,
        forcing_scale_diag: FeecVector,
        variances: FeecVector,
    ) -> EdgeVarianceDiagnostics {
        let std_devs = variances.map(|v| v.sqrt());
        let length_squared = edge_lengths.component_mul(&edge_lengths);
        let sqrt_mass_diag = mass_diag.map(safe_sqrt);
        let sqrt_mass_lumped_diag = mass_lumped_diag.map(safe_sqrt);
        let sqrt_forcing_scale_diag = forcing_scale_diag.map(safe_sqrt);
        let n = variances.len();
        EdgeVarianceDiagnostics {
            edge_lengths: edge_lengths.clone(),
            mass_diag: mass_diag.clone(),
            mass_lumped_diag: mass_lumped_diag.clone(),
            forcing_scale_diag: forcing_scale_diag.clone(),
            variances: variances.clone(),
            std_devs: std_devs.clone(),
            variance_per_length: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(variances[i], edge_lengths[i])),
            ),
            variance_per_length2: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(variances[i], length_squared[i])),
            ),
            variance_per_mass_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(variances[i], mass_diag[i])),
            ),
            variance_per_mass_lumped_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(variances[i], mass_lumped_diag[i])),
            ),
            variance_per_forcing_scale_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(variances[i], forcing_scale_diag[i])),
            ),
            std_per_length: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(std_devs[i], edge_lengths[i])),
            ),
            std_per_sqrt_mass_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(std_devs[i], sqrt_mass_diag[i])),
            ),
            std_per_sqrt_mass_lumped_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(std_devs[i], sqrt_mass_lumped_diag[i])),
            ),
            std_per_sqrt_forcing_scale_diag: FeecVector::from_iterator(
                n,
                (0..n).map(|i| safe_div(std_devs[i], sqrt_forcing_scale_diag[i])),
            ),
        }
    }

    fn diagonal_matrix(n: usize, value: f64) -> FeecCsr {
        let mut coo = FeecCoo::new(n, n);
        for i in 0..n {
            coo.push(i, i, value);
        }
        FeecCsr::from(&coo)
    }
}
