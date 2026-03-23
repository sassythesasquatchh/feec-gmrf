use feg_infer::torus_1form_conditioning::{
    run_torus_1form_conditioning_kappa0, write_torus_1form_conditioning_kappa0_outputs,
    Torus1FormConditioningKappa0Config,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = Instant::now();
    let config = Torus1FormConditioningKappa0Config::default();
    let out_dir = PathBuf::from("out/matern_1form_torus_conditioning_kappa0");

    let result = run_torus_1form_conditioning_kappa0(&config)?;
    write_torus_1form_conditioning_kappa0_outputs(&result, &out_dir)?;

    println!("Torus 1-form Matérn conditioning (kappa=0, harmonic-free)");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "tau={} noise_variance={} surface_vector_variance_mode={} rbmc_probes={} rbmc_batches={} seed={}",
        config.tau,
        config.noise_variance,
        config.surface_vector_variance_mode.as_str(),
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed
    );
    println!("observations={}", result.selected_observations.len());
    println!(
        "observed max abs error={} variance ratio={}",
        result.observed_summary.max_abs_error, result.observed_summary.variance_ratio_mean
    );
    println!(
        "surface trace variance ratio mean={}",
        mean(&result.variance_fields.surface_vector.trace.ratio)
    );
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn mean(values: &common::linalg::nalgebra::Vector<f64>) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}
