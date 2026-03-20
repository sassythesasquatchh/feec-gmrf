use feg_infer::torus_1form_conditioning::SurfaceVectorVarianceMode;
use feg_infer::torus_1form_pde_conditioning::{
    run_torus_1form_pde_conditioning, write_torus_1form_pde_conditioning_outputs,
    Torus1FormPdeConditioningConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (config, out_dir) = parse_args()?;
    let total_start = Instant::now();

    let result = run_torus_1form_pde_conditioning(&config)?;
    write_torus_1form_pde_conditioning_outputs(&result, &out_dir)?;

    println!("Torus 1-form Matérn PDE conditioning");
    println!("mesh={}", config.mesh_path.display());
    println!(
        "kappa={} tau={} noise_variance={} surface_vector_variance_mode={} rbmc_probes={} rbmc_batches={} seed={}",
        config.kappa,
        config.tau,
        config.noise_variance,
        config.surface_vector_variance_mode.as_str(),
        config.num_rbmc_probes,
        config.rbmc_batch_count,
        config.rng_seed,
    );
    println!("effective_range={}", result.effective_range);
    println!(
        "l2_error={} hd_error={} posterior_relative_residual_norm={}",
        result.l2_error, result.hd_error, result.posterior_relative_residual_norm
    );
    println!(
        "edge_variance_ratio_mean={} surface_trace_variance_ratio_mean={}",
        mean(&result.variance_ratio),
        mean(&result.variance_fields.surface_vector.trace.ratio),
    );
    println!("wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn parse_args(
) -> Result<(Torus1FormPdeConditioningConfig, PathBuf), Box<dyn std::error::Error>> {
    let mut config = Torus1FormPdeConditioningConfig::default();
    let mut out_dir = PathBuf::from("out/matern_1form_torus_pde_conditioning");
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-path" => {
                config.mesh_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --mesh-path"))?,
                );
            }
            "--kappa" => {
                config.kappa = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --kappa"))?,
                    "--kappa",
                )?;
            }
            "--tau" => {
                config.tau = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --tau"))?,
                    "--tau",
                )?;
            }
            "--noise-variance" => {
                config.noise_variance = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --noise-variance"))?,
                    "--noise-variance",
                )?;
            }
            "--surface-variance-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --surface-variance-mode"))?;
                config.surface_vector_variance_mode = value
                    .parse::<SurfaceVectorVarianceMode>()
                    .map_err(invalid_input)?;
            }
            "--num-rbmc-probes" => {
                config.num_rbmc_probes = parse_usize_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --num-rbmc-probes"))?,
                    "--num-rbmc-probes",
                )?;
            }
            "--rbmc-batch-count" => {
                config.rbmc_batch_count = parse_usize_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --rbmc-batch-count"))?,
                    "--rbmc-batch-count",
                )?;
            }
            "--seed" => {
                config.rng_seed = parse_u64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --seed"))?,
                    "--seed",
                )?;
            }
            "--out-dir" => {
                out_dir = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --out-dir"))?,
                );
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(invalid_input(format!("unrecognized argument `{other}`")).into());
            }
        }
    }

    Ok((config, out_dir))
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-infer --example matern_1form_torus_pde_conditioning -- [options]"
    );
    println!("Options:");
    println!("  --mesh-path <path>          Input torus mesh path");
    println!("  --kappa <f64>               Matérn kappa parameter");
    println!("  --tau <f64>                 Matérn tau parameter");
    println!("  --noise-variance <f64>      Observation noise variance");
    println!("  --surface-variance-mode <mode>  exact | rbmc | rbmc-clipped");
    println!("  --num-rbmc-probes <usize>   RBMC probe count");
    println!("  --rbmc-batch-count <usize>  RBMC batch count");
    println!("  --seed <u64>                RNG seed");
    println!("  --out-dir <path>            Output directory");
}

fn parse_f64_arg(value: String, flag: &str) -> Result<f64, Box<dyn std::error::Error>> {
    value
        .parse::<f64>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn parse_usize_arg(value: String, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    value
        .parse::<usize>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn parse_u64_arg(value: String, flag: &str) -> Result<u64, Box<dyn std::error::Error>> {
    value
        .parse::<u64>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn mean(values: &common::linalg::nalgebra::Vector<f64>) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}
