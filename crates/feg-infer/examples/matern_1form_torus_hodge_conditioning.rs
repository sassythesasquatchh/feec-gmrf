use feg_infer::torus_1form_hodge_conditioning::{
    run_torus_1form_hodge_conditioning, write_torus_1form_hodge_conditioning_outputs,
    Torus1FormHodgeConditioningConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let config = Torus1FormHodgeConditioningConfig::default();
    let out_dir = PathBuf::from("out/matern_1form_torus_hodge_conditioning");

    let result = run_torus_1form_hodge_conditioning(&config)?;
    write_torus_1form_hodge_conditioning_outputs(&result, &out_dir)?;

    println!("Torus 1-form Hodge-sector conditioning");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "kappa={} tau={} noise_variance={} harmonic_dim={}",
        config.kappa, config.tau, config.noise_variance, config.harmonic_dim
    );
    println!("observations={}", result.selected_observations.len());
    print_branch_summary(&result.exact);
    print_branch_summary(&result.coexact);
    print_branch_summary(&result.harmonic);
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn print_branch_summary(branch: &feg_infer::hodge_1form_conditioning::Hodge1FormBranchResult) {
    println!("branch={}", branch.kind.as_str());
    println!(
        "  latent_dimension={} max_abs_observation_error={} mean_abs_observation_error={}",
        branch.latent_dimension,
        branch.max_abs_observation_error,
        branch.mean_abs_observation_error
    );
    println!(
        "  harmonic_residual_norm_truth={} harmonic_residual_norm_posterior_mean={}",
        branch.harmonic_residual_norm_truth, branch.harmonic_residual_norm_posterior_mean
    );
    println!(
        "  curl_residual_norm={} curl_residual_relative={} coclosed_residual_norm={} coclosed_residual_relative={}",
        branch.posterior_bias_diagnostics.curl_residual_norm,
        branch.posterior_bias_diagnostics.curl_residual_relative,
        branch.posterior_bias_diagnostics.coclosed_residual_norm,
        branch.posterior_bias_diagnostics.coclosed_residual_relative
    );
}
