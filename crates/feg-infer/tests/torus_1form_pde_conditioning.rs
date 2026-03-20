use feg_infer::torus_1form_pde_conditioning::{
    run_torus_1form_pde_conditioning, write_torus_1form_pde_conditioning_outputs,
    Torus1FormPdeConditioningConfig,
};
use std::fs;

fn test_config() -> Torus1FormPdeConditioningConfig {
    let mut config = Torus1FormPdeConditioningConfig::default();
    config.num_rbmc_probes = 128;
    config.rbmc_batch_count = 4;
    config.rng_seed = 13;
    config
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

#[test]
fn torus_1form_pde_conditioning_builds_and_is_finite() {
    let result = run_torus_1form_pde_conditioning(&test_config())
        .expect("torus 1-form PDE conditioning should succeed");

    let ndofs = result.truth.len();
    assert_eq!(result.posterior_mean.len(), ndofs);
    assert_eq!(result.posterior_variance.len(), ndofs);
    assert_eq!(result.variance_ratio.len(), ndofs);
    assert_eq!(result.harmonic_free_posterior_mean.len(), ndofs);
    assert_eq!(result.pde_residual.len(), ndofs);
    assert!(result.truth.iter().all(|value| value.is_finite()));
    assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
    assert!(result.posterior_variance.iter().all(|value| value.is_finite()));
    assert!(
        result
            .variance_fields
            .surface_vector
            .trace
            .ratio
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(result.l2_error.is_finite());
    assert!(result.hd_error.is_finite());
}

#[test]
fn torus_1form_pde_conditioning_residuals_and_variances_behave() {
    let result = run_torus_1form_pde_conditioning(&test_config())
        .expect("torus 1-form PDE conditioning should succeed");

    assert!(
        result.truth_relative_residual_norm <= 1e-12,
        "truth projection should satisfy the discrete PDE rhs by construction; got {}",
        result.truth_relative_residual_norm
    );
    assert!(
        result.posterior_relative_residual_norm < 5e-2,
        "posterior mean should fit the PDE observations closely; got {}",
        result.posterior_relative_residual_norm
    );

    let max_latent = max_variance_ratio(&result.prior_variance, &result.posterior_variance);
    assert!(
        max_latent <= 1.0 + 1e-8,
        "latent marginal variances should not increase under conditioning; max ratio={max_latent}"
    );

    let max_harmonic_free = max_variance_ratio(
        &result.harmonic_free_prior_variance,
        &result.harmonic_free_posterior_variance,
    );
    assert!(
        max_harmonic_free <= 1.0 + 1e-8,
        "harmonic-free marginal variances should not increase under conditioning; max ratio={max_harmonic_free}"
    );

    let surface = &result.variance_fields.surface_vector.trace;
    let max_surface = max_variance_ratio(&surface.prior, &surface.posterior);
    assert!(
        max_surface <= 1.0 + 1e-8,
        "surface-vector trace variances should not increase under conditioning; max ratio={max_surface}"
    );
}

#[test]
fn torus_1form_pde_conditioning_writes_expected_outputs() {
    let result = run_torus_1form_pde_conditioning(&test_config())
        .expect("torus 1-form PDE conditioning should succeed");

    let out_dir = std::env::temp_dir().join(format!(
        "torus_1form_pde_conditioning_{}_{}",
        std::process::id(),
        result.truth.len()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    write_torus_1form_pde_conditioning_outputs(&result, &out_dir)
        .expect("writing torus 1-form PDE conditioning outputs should succeed");

    for relative in [
        "summary.txt",
        "fields.vtk",
        "edge_fields.csv",
        "posterior_mean_vector.vtk",
        "posterior_mean_surface_vector.vtk",
        "reconstructed_component_variance.vtk",
        "smoothed_component_variance.vtk",
        "circulation_variance.vtk",
    ] {
        assert!(
            out_dir.join(relative).is_file(),
            "expected output file {}",
            relative
        );
    }

    let _ = fs::remove_dir_all(&out_dir);
}
