use feg_infer::torus_1form_conditioning::{
    run_torus_1form_conditioning, write_torus_1form_conditioning_outputs, ObservationDirection,
    SurfaceVectorVarianceMode, Torus1FormConditioningConfig,
};
use std::f64::consts::PI;
use std::fs;

fn test_config() -> Torus1FormConditioningConfig {
    let mut config = Torus1FormConditioningConfig::default();
    config.num_rbmc_probes = 128;
    config.rbmc_batch_count = 4;
    config.rng_seed = 13;
    config
}

fn mean_observation_relative_deviation(
    branch: &feg_infer::torus_1form_conditioning::Torus1FormBranchResult,
    max_distance: Option<f64>,
    min_distance: Option<f64>,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for edge_index in 0..branch.posterior_mean.len() {
        let distance = branch.nearest_observation_distance[edge_index];
        if let Some(max_distance) = max_distance {
            if distance > max_distance {
                continue;
            }
        }
        if let Some(min_distance) = min_distance {
            if distance <= min_distance {
                continue;
            }
        }
        sum += (branch.posterior_mean[edge_index] - branch.nearest_observation_value[edge_index])
            .abs();
        count += 1;
    }

    assert!(count > 0, "expected a non-empty distance bucket");
    sum / count as f64
}

fn mean_variance_ratio_in_band(
    branch: &feg_infer::torus_1form_conditioning::Torus1FormBranchResult,
    min_distance: f64,
    max_distance: f64,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for edge_index in 0..branch.posterior_variance.len() {
        let distance = branch.nearest_observation_distance[edge_index];
        if distance <= min_distance || distance > max_distance {
            continue;
        }
        sum += safe_ratio(
            branch.posterior_variance[edge_index],
            branch.prior_variance[edge_index],
        );
        count += 1;
    }

    assert!(count > 0, "expected a non-empty distance band");
    sum / count as f64
}

fn max_variance_ratio(
    prior_variance: &common::linalg::nalgebra::Vector<f64>,
    posterior_variance: &common::linalg::nalgebra::Vector<f64>,
) -> f64 {
    prior_variance
        .iter()
        .zip(posterior_variance.iter())
        .map(|(prior, posterior)| safe_ratio(*posterior, *prior))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= 1e-12 {
        0.0
    } else {
        numerator / denominator
    }
}

fn variance_pattern_row<'a>(
    branch: &'a feg_infer::torus_1form_conditioning::Torus1FormBranchResult,
    object: &str,
) -> &'a feg_infer::torus_1form_conditioning::Torus1FormVariancePatternSummaryRow {
    branch
        .variance_pattern
        .summary_rows
        .iter()
        .find(|row| row.object == object)
        .unwrap_or_else(|| panic!("missing variance-pattern summary row for {object}"))
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

fn orientation_sensitive_local_variance_means(
    result: &feg_infer::torus_1form_conditioning::Torus1FormConditioningResult,
    branch: &feg_infer::torus_1form_conditioning::Torus1FormBranchResult,
    distance_max_scale: f64,
) -> (f64, f64) {
    let mut compatible_sum = 0.0;
    let mut compatible_count = 0_usize;
    let mut transverse_sum = 0.0;
    let mut transverse_count = 0_usize;
    let distance_max = distance_max_scale * result.effective_range;

    for selected in &result.selected_observations {
        for edge_index in 0..branch.posterior_variance.len() {
            if edge_index == selected.edge_index {
                continue;
            }

            let distance = intrinsic_torus_distance(
                result.major_radius,
                result.minor_radius,
                result.edge_theta[edge_index],
                result.edge_phi[edge_index],
                selected.edge_theta,
                selected.edge_phi,
            );
            if distance <= 0.0 || distance > distance_max {
                continue;
            }

            let alignment = result.toroidal_alignment_sq[edge_index];
            let ratio = safe_ratio(
                branch.posterior_variance[edge_index],
                branch.prior_variance[edge_index],
            );
            match selected.direction {
                ObservationDirection::Toroidal => {
                    if alignment >= 0.8 {
                        compatible_sum += ratio;
                        compatible_count += 1;
                    } else if alignment <= 0.2 {
                        transverse_sum += ratio;
                        transverse_count += 1;
                    }
                }
                ObservationDirection::Poloidal => {
                    if alignment <= 0.2 {
                        compatible_sum += ratio;
                        compatible_count += 1;
                    } else if alignment >= 0.8 {
                        transverse_sum += ratio;
                        transverse_count += 1;
                    }
                }
            }
        }
    }

    assert!(compatible_count > 0, "expected compatible nearby edges");
    assert!(transverse_count > 0, "expected transverse nearby edges");
    (
        compatible_sum / compatible_count as f64,
        transverse_sum / transverse_count as f64,
    )
}

#[test]
fn torus_1form_conditioning_selects_edges_and_preserves_harmonic_design() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    assert_eq!(
        result.selected_observations.len(),
        result.observation_targets.len()
    );
    assert_eq!(
        result.selected_observations.len(),
        result.observation_indices.len()
    );

    let unique_edges = result
        .selected_observations
        .iter()
        .map(|selected| selected.edge_index)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique_edges.len(), result.selected_observations.len());

    for selected in &result.selected_observations {
        if selected.used_fallback {
            continue;
        }
        match selected.direction {
            ObservationDirection::Toroidal => {
                assert!(
                    selected.toroidal_alignment_sq >= 0.8,
                    "toroidal observation {} should land on a toroidal-aligned edge",
                    selected.observation_index
                );
            }
            ObservationDirection::Poloidal => {
                assert!(
                    selected.toroidal_alignment_sq <= 0.2,
                    "poloidal observation {} should land on a poloidal-aligned edge",
                    selected.observation_index
                );
            }
        }
    }

    let harmonic_free_truth = &result.harmonic_free_constrained.harmonic_coefficients_truth;
    assert!(
        harmonic_free_truth[0].abs() <= 1e-8 && harmonic_free_truth[1].abs() <= 1e-8,
        "harmonic-free truth should satisfy H^T M u = 0 up to tolerance; got {:?}",
        harmonic_free_truth
    );

    let full_truth = &result.full_unconstrained.harmonic_coefficients_truth;
    assert!(
        (full_truth[0] - 0.75).abs() <= 1e-8,
        "expected toroidal harmonic coefficient 0.75, got {}",
        full_truth[0]
    );
    assert!(
        (full_truth[1] + 0.50).abs() <= 1e-8,
        "expected poloidal harmonic coefficient -0.50, got {}",
        full_truth[1]
    );
}

#[test]
fn torus_1form_conditioning_branches_match_local_posterior_expectations() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    let constrained = &result.harmonic_free_constrained;
    assert!(
        constrained.summary.observed.max_abs_error <= 1e-3,
        "constrained branch should match observed edge integrals closely; got {}",
        constrained.summary.observed.max_abs_error
    );
    assert!(
        constrained.summary.observed.variance_ratio_mean < 0.25,
        "constrained branch should strongly collapse variance at observations; got {}",
        constrained.summary.observed.variance_ratio_mean
    );
    assert!(
        constrained.summary.observed.variance_ratio_mean
            < constrained.summary.near.variance_ratio_mean,
        "constrained branch should reduce variance more strongly at observations than nearby"
    );
    assert!(
        constrained.summary.near.variance_ratio_mean < 0.95,
        "constrained branch should reduce variance near observations; got {}",
        constrained.summary.near.variance_ratio_mean
    );
    assert!(
        constrained.summary.near.variance_ratio_mean < constrained.summary.far.variance_ratio_mean,
        "constrained branch variance reduction should weaken far from the observations"
    );
    assert!(
        constrained.summary.far.variance_ratio_mean > 0.90,
        "constrained branch should return close to prior variance far away; got {}",
        constrained.summary.far.variance_ratio_mean
    );
    let constrained_near_observation_deviation =
        mean_observation_relative_deviation(constrained, Some(result.neighbourhood_radius), None);
    let constrained_far_observation_deviation =
        mean_observation_relative_deviation(constrained, None, Some(result.far_radius));
    assert!(
        constrained_near_observation_deviation < constrained_far_observation_deviation,
        "constrained branch should stay closer to the observations nearby than far away"
    );

    let full = &result.full_unconstrained;
    assert!(
        full.summary.observed.max_abs_error <= 1e-3,
        "full branch should match observed edge integrals closely; got {}",
        full.summary.observed.max_abs_error
    );
    for i in 0..2 {
        let truth_coeff = full.harmonic_coefficients_truth[i];
        let posterior_coeff = full.harmonic_coefficients_posterior_mean[i];
        assert!(
            posterior_coeff.signum() == truth_coeff.signum(),
            "posterior harmonic coefficient {} should keep the truth sign; truth={} posterior={}",
            i,
            truth_coeff,
            posterior_coeff
        );
        assert!(
            posterior_coeff.abs() >= 0.15 * truth_coeff.abs(),
            "posterior harmonic coefficient {} should recover a meaningful fraction of the truth; truth={} posterior={}",
            i,
            truth_coeff,
            posterior_coeff
        );
    }
    let full_near_observation_deviation =
        mean_observation_relative_deviation(full, Some(result.neighbourhood_radius), None);
    let full_far_observation_deviation =
        mean_observation_relative_deviation(full, None, Some(result.far_radius));
    assert!(
        full_near_observation_deviation < full_far_observation_deviation,
        "full branch should stay closer to nearby observations than far away"
    );
    assert!(
        full.summary.observed.variance_ratio_mean < full.summary.near.variance_ratio_mean,
        "full branch should reduce variance more strongly at observations than nearby"
    );
    assert!(
        full.summary.near.harmonic_free_variance_ratio_mean
            < full.summary.far.harmonic_free_variance_ratio_mean,
        "full branch harmonic-free variance reduction should be stronger near observations"
    );
}

#[test]
fn torus_1form_conditioning_larger_kappa_localizes_fixed_distance_variance_reduction() {
    let mut low_kappa_config = test_config();
    low_kappa_config.kappa = 2.0;
    let low_kappa = run_torus_1form_conditioning(&low_kappa_config)
        .expect("low-kappa torus 1-form conditioning should succeed");

    let mut high_kappa_config = test_config();
    high_kappa_config.kappa = 8.0;
    let high_kappa = run_torus_1form_conditioning(&high_kappa_config)
        .expect("high-kappa torus 1-form conditioning should succeed");

    assert!(
        low_kappa.effective_range > high_kappa.effective_range,
        "effective range should shrink as kappa increases"
    );

    let band_min = 0.4;
    let band_max = 0.8;
    let low_ratio =
        mean_variance_ratio_in_band(&low_kappa.harmonic_free_constrained, band_min, band_max);
    let high_ratio =
        mean_variance_ratio_in_band(&high_kappa.harmonic_free_constrained, band_min, band_max);
    assert!(
        low_ratio < high_ratio,
        "at a fixed intrinsic distance band, smaller kappa should spread variance reduction further; low={} high={}",
        low_ratio,
        high_ratio
    );
}

#[test]
fn torus_1form_conditioning_surface_vector_trace_variances_do_not_increase() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    for (label, branch) in [
        (
            "harmonic_free_constrained",
            &result.harmonic_free_constrained,
        ),
        ("full_unconstrained", &result.full_unconstrained),
    ] {
        let surface = &branch.variance_pattern.surface_vector.trace;
        let max_ratio = max_variance_ratio(&surface.prior, &surface.posterior);
        assert!(
            max_ratio <= 1.0 + 1e-8,
            "{label} surface-vector trace marginal variances should not increase under conditioning; max ratio={max_ratio}",
        );
    }
}

#[test]
fn torus_1form_conditioning_surface_vector_trace_variances_do_not_increase_with_rbmc_clipping() {
    let mut config = test_config();
    config.surface_vector_variance_mode = SurfaceVectorVarianceMode::RbmcClipped;
    let result =
        run_torus_1form_conditioning(&config).expect("torus 1-form conditioning should succeed");

    for (label, branch) in [
        (
            "harmonic_free_constrained",
            &result.harmonic_free_constrained,
        ),
        ("full_unconstrained", &result.full_unconstrained),
    ] {
        let surface = &branch.variance_pattern.surface_vector.trace;
        let max_ratio = max_variance_ratio(&surface.prior, &surface.posterior);
        assert!(
            max_ratio <= 1.0 + 1e-8,
            "{label} clipped RBMC surface-vector trace marginal variances should not increase under conditioning; max ratio={max_ratio}",
        );
    }
}

#[test]
fn torus_1form_conditioning_edgewise_variances_do_not_increase() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    for (label, branch) in [
        (
            "harmonic_free_constrained",
            &result.harmonic_free_constrained,
        ),
        ("full_unconstrained", &result.full_unconstrained),
    ] {
        let max_latent = max_variance_ratio(&branch.prior_variance, &branch.posterior_variance);
        assert!(
            max_latent <= 1.0 + 1e-8,
            "{label} latent marginal variances should not increase under conditioning; max ratio={max_latent}",
        );

        let max_harmonic_free = max_variance_ratio(
            &branch.harmonic_free_prior_variance,
            &branch.harmonic_free_posterior_variance,
        );
        assert!(
            max_harmonic_free <= 1.0 + 1e-8,
            "{label} harmonic-free marginal variances should not increase under conditioning; max ratio={max_harmonic_free}",
        );
    }
}

#[test]
fn torus_1form_conditioning_local_variance_reduction_is_orientation_sensitive_and_reported() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    let (compatible_mean, transverse_mean) = orientation_sensitive_local_variance_means(
        &result,
        &result.harmonic_free_constrained,
        0.20,
    );
    assert!(
        compatible_mean < transverse_mean,
        "compatible nearby edges should retain less variance than transverse nearby edges; compatible={} transverse={}",
        compatible_mean,
        transverse_mean
    );

    let out_dir = std::env::temp_dir().join(format!(
        "torus_1form_conditioning_diag_{}_{}",
        std::process::id(),
        result.harmonic_free_constrained.posterior_mean.len()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    write_torus_1form_conditioning_outputs(&result, &out_dir)
        .expect("writing torus 1-form conditioning outputs should succeed");

    assert!(
        out_dir
            .join("harmonic_free_constrained/observation_relative_edge_variance.csv")
            .is_file(),
        "expected raw observation-relative variance diagnostics CSV"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/observation_variance_profile.csv")
            .is_file(),
        "expected shell-based observation variance profile CSV"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/observation_variance_diagnostics.txt")
            .is_file(),
        "expected human-readable observation variance diagnostics report"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/variance_pattern_summary.csv")
            .is_file(),
        "expected variance-pattern summary CSV"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/variance_pattern_shell_profiles.csv")
            .is_file(),
        "expected variance-pattern shell profile CSV"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/variance_pattern_summary.txt")
            .is_file(),
        "expected variance-pattern summary TXT"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/reconstructed_component_variance.vtk")
            .is_file(),
        "expected reconstructed-component variance VTK"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/smoothed_component_variance.vtk")
            .is_file(),
        "expected smoothed-component variance VTK"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/circulation_variance.vtk")
            .is_file(),
        "expected circulation variance VTK"
    );
    assert!(
        out_dir
            .join("harmonic_free_constrained/observation_edge_vtks/observation_00.vtk")
            .is_file(),
        "expected per-observation raw edge VTK"
    );

    let _ = fs::remove_dir_all(&out_dir);
}

#[test]
fn torus_1form_conditioning_variance_pattern_panel_matches_expected_locality() {
    let result = run_torus_1form_conditioning(&test_config())
        .expect("torus 1-form conditioning should succeed");

    let branch = &result.harmonic_free_constrained;
    let edge_all = variance_pattern_row(branch, "edge_all");
    let component_matched = variance_pattern_row(branch, "component_matched");
    let smoothed_matched = variance_pattern_row(branch, "smoothed_matched");
    let smoothed_trace = variance_pattern_row(branch, "smoothed_trace");
    let circulation = variance_pattern_row(branch, "circulation");

    assert!(
        component_matched.very_local_ratio < component_matched.far_ratio,
        "matched reconstructed component should retain less variance near observations than far away; near={} far={}",
        component_matched.very_local_ratio,
        component_matched.far_ratio
    );
    assert!(
        smoothed_matched.very_local_ratio < smoothed_matched.local_ratio
            && smoothed_matched.local_ratio < smoothed_matched.far_ratio,
        "smoothed matched component should decay outward from observations; very_local={} local={} far={}",
        smoothed_matched.very_local_ratio,
        smoothed_matched.local_ratio,
        smoothed_matched.far_ratio
    );
    assert!(
        smoothed_trace.very_local_ratio < smoothed_trace.far_ratio,
        "smoothed trace should be more reduced near observations than far away; near={} far={}",
        smoothed_trace.very_local_ratio,
        smoothed_trace.far_ratio
    );
    assert!(
        circulation.very_local_ratio < circulation.far_ratio,
        "circulation variance should reduce locally and return toward the prior far away; near={} far={}",
        circulation.very_local_ratio,
        circulation.far_ratio
    );
    assert!(
        edge_all.very_local_orientation_contrast > 0.0
            && edge_all.local_orientation_contrast.is_finite(),
        "raw edge very-local orientation contrast should be positive and the local shell should report a finite contrast; very_local={} local={}",
        edge_all.very_local_orientation_contrast,
        edge_all.local_orientation_contrast
    );
}

#[test]
fn torus_1form_conditioning_primary_localization_auc_decreases_with_kappa() {
    let mut low_kappa_config = test_config();
    low_kappa_config.kappa = 2.0;
    let low = run_torus_1form_conditioning(&low_kappa_config)
        .expect("low-kappa torus 1-form conditioning should succeed");

    let mut mid_kappa_config = test_config();
    mid_kappa_config.kappa = 4.0;
    let mid = run_torus_1form_conditioning(&mid_kappa_config)
        .expect("mid-kappa torus 1-form conditioning should succeed");

    let mut high_kappa_config = test_config();
    high_kappa_config.kappa = 8.0;
    let high = run_torus_1form_conditioning(&high_kappa_config)
        .expect("high-kappa torus 1-form conditioning should succeed");

    let low_smoothed = variance_pattern_row(&low.harmonic_free_constrained, "smoothed_matched");
    let mid_smoothed = variance_pattern_row(&mid.harmonic_free_constrained, "smoothed_matched");
    let high_smoothed = variance_pattern_row(&high.harmonic_free_constrained, "smoothed_matched");
    assert!(
        low_smoothed.localization_auc > mid_smoothed.localization_auc
            && mid_smoothed.localization_auc > high_smoothed.localization_auc,
        "smoothed matched-component localization AUC should decrease as kappa increases; low={} mid={} high={}",
        low_smoothed.localization_auc,
        mid_smoothed.localization_auc,
        high_smoothed.localization_auc
    );

    let low_circulation = variance_pattern_row(&low.harmonic_free_constrained, "circulation");
    let mid_circulation = variance_pattern_row(&mid.harmonic_free_constrained, "circulation");
    let high_circulation = variance_pattern_row(&high.harmonic_free_constrained, "circulation");
    assert!(
        low_circulation.localization_auc > mid_circulation.localization_auc
            && mid_circulation.localization_auc > high_circulation.localization_auc,
        "circulation localization AUC should decrease as kappa increases; low={} mid={} high={}",
        low_circulation.localization_auc,
        mid_circulation.localization_auc,
        high_circulation.localization_auc
    );
}
