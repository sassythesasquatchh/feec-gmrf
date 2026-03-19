use feg_infer::torus_0form_conditioning::{
    run_torus_0form_conditioning, Torus0FormConditioningConfig,
};

#[test]
fn scalar_torus_conditioning_tracks_observations_and_reduces_local_uncertainty() {
    let result = run_torus_0form_conditioning(&Torus0FormConditioningConfig::default())
        .expect("torus conditioning should succeed");

    assert_eq!(
        result.observation_summaries.len(),
        result.observation_indices.len()
    );
    assert!(!result.observation_summaries.is_empty());

    let max_observation_abs_error = result
        .observation_summaries
        .iter()
        .map(|summary| summary.abs_error_at_observation)
        .fold(0.0, f64::max);
    assert!(
        max_observation_abs_error < 5e-3,
        "posterior mean should match the observations closely at observed points; got max abs error {max_observation_abs_error:e}"
    );

    for summary in &result.observation_summaries {
        assert!(
            summary.neighbourhood_count > 1,
            "expected each kappa-scaled neighbourhood to include more than the observed vertex"
        );
        assert!(
            summary.neighbourhood_mean_abs_deviation_from_observation
                < summary.global_mean_abs_deviation_from_observation,
            "posterior mean should stay closer to the observation nearby than globally (obs {})",
            summary.observation_index
        );
        assert!(
            summary.posterior_variance_at_observation < 0.2 * summary.prior_variance_at_observation,
            "posterior variance should collapse strongly at the observed point (obs {})",
            summary.observation_index
        );
        assert!(
            summary.neighbourhood_posterior_variance_mean
                < summary.neighbourhood_prior_variance_mean,
            "posterior variance should be lower than the prior in the local neighbourhood (obs {})",
            summary.observation_index
        );
        assert!(
            summary.neighbourhood_variance_reduction_mean > 0.0,
            "expected a positive average variance reduction near observation {}",
            summary.observation_index
        );
    }

    let max_variance_increase = result
        .posterior_variance
        .iter()
        .zip(result.prior_variance.iter())
        .map(|(posterior, prior)| posterior - prior)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_variance_increase <= 1e-8,
        "posterior marginal variances should not exceed prior variances; max increase was {max_variance_increase:e}"
    );
}
