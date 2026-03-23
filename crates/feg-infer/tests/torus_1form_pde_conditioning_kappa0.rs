use feg_infer::torus_1form_pde_conditioning::{
    run_torus_1form_pde_conditioning_kappa0, write_torus_1form_pde_conditioning_kappa0_outputs,
    Torus1FormPdeConditioningKappa0Config,
};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use feg_infer::torus_1form_conditioning::SurfaceVectorVarianceMode;

fn test_config() -> Torus1FormPdeConditioningKappa0Config {
    let mut config = Torus1FormPdeConditioningKappa0Config::default();
    config.mesh_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_resolution_0.msh");
    config.surface_vector_variance_mode = SurfaceVectorVarianceMode::RbmcClipped;
    config.num_rbmc_probes = 16;
    config.rbmc_batch_count = 2;
    config.rng_seed = 13;
    config
}

fn test_result(
) -> &'static feg_infer::torus_1form_pde_conditioning::Torus1FormPdeConditioningKappa0Result {
    static RESULT: OnceLock<
        feg_infer::torus_1form_pde_conditioning::Torus1FormPdeConditioningKappa0Result,
    > = OnceLock::new();
    RESULT.get_or_init(|| {
        run_torus_1form_pde_conditioning_kappa0(&test_config())
            .expect("kappa0 torus 1-form PDE conditioning should succeed")
    })
}

#[test]
fn torus_1form_pde_conditioning_kappa0_is_harmonic_free_and_finite() {
    let result = test_result();

    assert_eq!(result.posterior_mean.len(), result.truth.len());
    assert_eq!(result.posterior_variance.len(), result.truth.len());
    assert!(result.truth.iter().all(|value| value.is_finite()));
    assert!(result.posterior_mean.iter().all(|value| value.is_finite()));
    assert!(result
        .posterior_variance
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .variance_fields
        .surface_vector
        .trace
        .ratio
        .iter()
        .all(|value| value.is_finite()));
    assert!(
        result
            .harmonic_coefficients_truth
            .iter()
            .all(|value| value.abs() <= 1e-6),
        "truth should be harmonic-free; got {:?}",
        result.harmonic_coefficients_truth
    );
    assert!(
        result
            .harmonic_coefficients_posterior_mean
            .iter()
            .all(|value| value.abs() <= 1e-6),
        "posterior mean should be harmonic-free; got {:?}",
        result.harmonic_coefficients_posterior_mean
    );
    assert!(
        result.posterior_relative_residual_norm < 5e-2,
        "posterior mean should fit the PDE observations closely; got {}",
        result.posterior_relative_residual_norm
    );
}

#[test]
fn torus_1form_pde_conditioning_kappa0_writes_expected_outputs() {
    let result = test_result();

    let out_dir = std::env::temp_dir().join(format!(
        "torus_1form_pde_conditioning_kappa0_{}_{}",
        std::process::id(),
        result.truth.len()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    write_torus_1form_pde_conditioning_kappa0_outputs(&result, &out_dir)
        .expect("writing kappa0 torus 1-form PDE conditioning outputs should succeed");

    for relative in [
        "summary.txt",
        "fields.vtk",
        "edge_fields.csv",
        "posterior_mean_vector.vtk",
        "posterior_mean_surface_vector.vtk",
        "surface_vector_stats.csv",
        "reconstructed_component_variance.vtk",
        "circulation_variance.vtk",
    ] {
        assert!(
            out_dir.join(relative).is_file(),
            "expected output file {}",
            relative
        );
    }

    let summary = fs::read_to_string(out_dir.join("summary.txt"))
        .expect("summary file should be readable as text");
    assert!(summary.contains("surface_vector_posterior_mean_magnitude_mean="));
    assert!(summary.contains("surface_vector_marginal_variance_ratio_mean="));

    let _ = fs::remove_dir_all(&out_dir);
}
