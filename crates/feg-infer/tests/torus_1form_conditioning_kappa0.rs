use feg_infer::torus_1form_conditioning::{
    run_torus_1form_conditioning_kappa0, write_torus_1form_conditioning_kappa0_outputs,
    SurfaceVectorVarianceMode, Torus1FormConditioningKappa0Config,
};
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

fn test_config() -> Torus1FormConditioningKappa0Config {
    let mut config = Torus1FormConditioningKappa0Config::default();
    config.mesh_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_resolution_0.msh");
    config.surface_vector_variance_mode = SurfaceVectorVarianceMode::RbmcClipped;
    config.num_rbmc_probes = 16;
    config.rbmc_batch_count = 2;
    config.rng_seed = 13;
    config
}

fn test_result() -> &'static feg_infer::torus_1form_conditioning::Torus1FormConditioningKappa0Result
{
    static RESULT: OnceLock<
        feg_infer::torus_1form_conditioning::Torus1FormConditioningKappa0Result,
    > = OnceLock::new();
    RESULT.get_or_init(|| {
        run_torus_1form_conditioning_kappa0(&test_config())
            .expect("kappa0 torus 1-form conditioning should succeed")
    })
}

#[test]
fn torus_1form_conditioning_kappa0_is_harmonic_free_and_tracks_observations() {
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
            .harmonic_coefficients_posterior_mean
            .iter()
            .all(|value| value.abs() <= 1e-6),
        "posterior mean should be harmonic-free; got {:?}",
        result.harmonic_coefficients_posterior_mean
    );
    assert!(
        result.observed_summary.max_abs_error < 1e-3,
        "posterior mean should match observations closely; got {}",
        result.observed_summary.max_abs_error
    );
}

#[test]
fn torus_1form_conditioning_kappa0_writes_expected_outputs() {
    let result = test_result();

    let out_dir = std::env::temp_dir().join(format!(
        "torus_1form_conditioning_kappa0_{}_{}",
        std::process::id(),
        result.truth.len()
    ));
    let _ = fs::remove_dir_all(&out_dir);
    write_torus_1form_conditioning_kappa0_outputs(&result, &out_dir)
        .expect("writing kappa0 torus 1-form conditioning outputs should succeed");

    for relative in [
        "summary.txt",
        "fields.vtk",
        "posterior_mean_vector.vtk",
        "posterior_mean_surface_vector.vtk",
        "surface_vector_stats.csv",
        "selected_observations.csv",
        "edge_fields.csv",
        "observations.csv",
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

    let surface_vtk = fs::read_to_string(out_dir.join("posterior_mean_surface_vector.vtk"))
        .expect("surface vector VTK should be readable as text");
    assert!(surface_vtk.contains("VECTORS truth_surface_vector double"));
    assert!(surface_vtk.contains("VECTORS posterior_mean_surface_vector double"));

    let _ = fs::remove_dir_all(&out_dir);
}
