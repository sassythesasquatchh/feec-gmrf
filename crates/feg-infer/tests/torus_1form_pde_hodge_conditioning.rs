use feg_infer::torus_1form_pde_hodge_conditioning::{
    run_torus_1form_pde_hodge_conditioning, write_torus_1form_pde_hodge_conditioning_outputs,
    Torus1FormPdeHodgeConditioningConfig,
};
use std::fs;

fn test_config() -> Torus1FormPdeHodgeConditioningConfig {
    let mut config = Torus1FormPdeHodgeConditioningConfig::default();
    config.num_rbmc_probes = 16;
    config.rbmc_batch_count = 4;
    config.rng_seed = 13;
    config
}

fn max_abs(values: &common::linalg::nalgebra::Vector<f64>) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn max_abs_diff(
    lhs: &common::linalg::nalgebra::Vector<f64>,
    rhs: &common::linalg::nalgebra::Vector<f64>,
) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

#[test]
fn torus_1form_pde_hodge_conditioning_builds_and_respects_branch_biases() {
    let result = run_torus_1form_pde_hodge_conditioning(&test_config())
        .expect("torus 1-form PDE Hodge-split conditioning should succeed");

    assert_eq!(result.harmonic.latent_dimension, 2);

    for branch in [&result.exact, &result.coexact, &result.harmonic] {
        assert_eq!(branch.observation_count, result.full.rhs.len());
        assert_eq!(
            branch.conditioning.posterior_mean.len(),
            result.full.truth.len()
        );
        assert_eq!(
            max_abs_diff(&branch.conditioning.rhs, &result.full.rhs),
            0.0,
            "branch {} should use the same rhs observations as the full model",
            branch.kind.as_str()
        );
        assert!(branch
            .conditioning
            .posterior_mean
            .iter()
            .all(|value| value.is_finite()));
        assert!(branch
            .conditioning
            .posterior_variance
            .iter()
            .all(|value| value.is_finite()));
        assert!(branch
            .conditioning
            .variance_fields
            .surface_vector
            .trace
            .ratio
            .iter()
            .all(|value| value.is_finite()));
        assert!(branch
            .conditioning
            .posterior_deterministic_l2_error
            .is_finite());
    }

    assert!(result.full.truth.iter().all(|value| value.is_finite()));
    assert!(result
        .full
        .posterior_mean
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .full
        .posterior_variance
        .iter()
        .all(|value| value.is_finite()));
    assert!(result.full.posterior_deterministic_l2_error.is_finite());

    for branch in [&result.exact, &result.coexact, &result.harmonic] {
        assert!(
            result.full.posterior_relative_residual_norm
                <= branch.conditioning.posterior_relative_residual_norm + 1e-10,
            "full model should fit the PDE observations at least as well as branch {}",
            branch.kind.as_str()
        );
    }

    assert!(
        result
            .exact
            .posterior_bias_diagnostics
            .curl_residual_relative
            <= 1e-10,
        "exact branch curl residual regressed: {:?}",
        result.exact.posterior_bias_diagnostics
    );
    assert!(
        result
            .coexact
            .posterior_bias_diagnostics
            .coclosed_residual_relative
            <= 5e-2,
        "coexact branch coclosed residual regressed: {:?}",
        result.coexact.posterior_bias_diagnostics
    );
    assert!(
        max_abs(&result.harmonic.conditioning.harmonic_free_posterior_mean) <= 1e-8,
        "harmonic branch harmonic-free posterior mean should be negligible"
    );
    assert!(
        max_abs(
            &result
                .harmonic
                .conditioning
                .harmonic_free_posterior_variance
        ) <= 1e-8,
        "harmonic branch harmonic-free posterior variance should be negligible"
    );
}

#[test]
fn torus_1form_pde_hodge_conditioning_writes_expected_outputs() {
    let result = run_torus_1form_pde_hodge_conditioning(&test_config())
        .expect("torus 1-form PDE Hodge-split conditioning should succeed");

    let out_dir = std::env::temp_dir().join(format!(
        "torus_1form_pde_hodge_conditioning_{}_{}",
        std::process::id(),
        result.full.truth.len()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    write_torus_1form_pde_hodge_conditioning_outputs(&result, &out_dir)
        .expect("writing torus 1-form PDE Hodge-split outputs should succeed");

    assert!(
        out_dir.join("comparison_summary.txt").is_file(),
        "expected root comparison summary"
    );
    let comparison_summary = fs::read_to_string(out_dir.join("comparison_summary.txt"))
        .expect("comparison summary should be readable as text");
    assert!(comparison_summary.contains("posterior_deterministic_l2_error="));

    for branch in ["full", "exact", "coexact", "harmonic"] {
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
                out_dir.join(branch).join(relative).is_file(),
                "expected output file for branch {branch}: {relative}"
            );
        }

        let summary = fs::read_to_string(out_dir.join(branch).join("summary.txt"))
            .expect("branch summary should be readable as text");
        assert!(summary.contains("posterior_deterministic_l2_error="));
    }

    let _ = fs::remove_dir_all(&out_dir);
}
