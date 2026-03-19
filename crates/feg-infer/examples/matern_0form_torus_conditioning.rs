use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use feg_infer::torus_0form_conditioning::{
    run_torus_0form_conditioning, Torus0FormConditioningConfig, Torus0FormConditioningResult,
};
use feg_infer::vtk::write_0cochain_vtk_fields;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct ExampleConfig {
    conditioning: Torus0FormConditioningConfig,
    out_dir: PathBuf,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            conditioning: Torus0FormConditioningConfig::default(),
            out_dir: PathBuf::from("out/matern_0form_torus_conditioning"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let _ = fs::remove_dir_all(&config.out_dir);
    fs::create_dir_all(&config.out_dir)?;

    let result = run_torus_0form_conditioning(&config.conditioning)?;
    write_fields_vtk(&config.out_dir, &result)?;
    write_observation_summary_csv(&config.out_dir, &result)?;
    write_summary_txt(&config.out_dir, &config, &result)?;
    print_summary(&config, &result);

    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn std::error::Error>> {
    let mut config = ExampleConfig::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-path" => {
                config.conditioning.mesh_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --mesh-path"))?,
                );
            }
            "--kappa" => {
                config.conditioning.kappa = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --kappa"))?,
                    "--kappa",
                )?;
            }
            "--tau" => {
                config.conditioning.tau = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --tau"))?,
                    "--tau",
                )?;
            }
            "--noise-variance" => {
                config.conditioning.noise_variance = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --noise-variance"))?,
                    "--noise-variance",
                )?;
            }
            "--radius-scale" => {
                config.conditioning.neighbourhood_radius_scale = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --radius-scale"))?,
                    "--radius-scale",
                )?;
            }
            "--out-dir" => {
                config.out_dir = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --out-dir"))?,
                );
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(invalid_input(format!("unknown argument: {other}")).into());
            }
        }
    }

    Ok(config)
}

fn parse_f64_arg(value: String, flag: &str) -> Result<f64, io::Error> {
    value
        .parse::<f64>()
        .map_err(|_| invalid_input(format!("failed to parse {flag} as f64")))
}

fn write_fields_vtk(
    out_dir: &PathBuf,
    result: &Torus0FormConditioningResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let observation_samples = build_observation_samples(
        result.truth.len(),
        &result.observation_indices,
        &result.observation_values,
    );
    let truth = Cochain::new(0, result.truth.clone());
    let posterior_mean = Cochain::new(0, result.posterior_mean.clone());
    let abs_mean_error = Cochain::new(0, result.absolute_mean_error.clone());
    let prior_variance = Cochain::new(0, result.prior_variance.clone());
    let posterior_variance = Cochain::new(0, result.posterior_variance.clone());
    let variance_reduction = Cochain::new(0, result.variance_reduction.clone());
    let observed_mask = Cochain::new(0, result.observed_mask.clone());
    let observation_samples = Cochain::new(0, observation_samples);
    let nearest_observation_value = Cochain::new(0, result.nearest_observation_value.clone());
    let theta = Cochain::new(0, result.theta.clone());
    let phi = Cochain::new(0, result.phi.clone());

    write_0cochain_vtk_fields(
        out_dir.join("posterior_fields.vtk"),
        &result.coords,
        &result.topology,
        &[
            ("truth", &truth),
            ("posterior_mean", &posterior_mean),
            ("abs_mean_error", &abs_mean_error),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_reduction", &variance_reduction),
            ("observed_mask", &observed_mask),
            ("observation_samples", &observation_samples),
            ("nearest_observation_value", &nearest_observation_value),
            ("theta", &theta),
            ("phi", &phi),
        ],
    )?;

    Ok(())
}

fn write_observation_summary_csv(
    out_dir: &PathBuf,
    result: &Torus0FormConditioningResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(out_dir.join("observation_summary.csv"))?);
    writeln!(
        writer,
        "observation_index,vertex_index,x,y,z,theta,phi,observation_value,posterior_mean_at_observation,abs_error_at_observation,prior_variance_at_observation,posterior_variance_at_observation,neighbourhood_count,neighbourhood_mean,neighbourhood_mean_abs_deviation_from_observation,global_mean_abs_deviation_from_observation,neighbourhood_prior_variance_mean,neighbourhood_posterior_variance_mean,neighbourhood_variance_reduction_mean"
    )?;

    for summary in &result.observation_summaries {
        let coord = result.coords.coord(summary.vertex_index);
        writeln!(
            writer,
            "{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            summary.observation_index,
            summary.vertex_index,
            coord[0],
            coord[1],
            coord[2],
            summary.theta,
            summary.phi,
            summary.observation_value,
            summary.posterior_mean_at_observation,
            summary.abs_error_at_observation,
            summary.prior_variance_at_observation,
            summary.posterior_variance_at_observation,
            summary.neighbourhood_count,
            summary.neighbourhood_mean,
            summary.neighbourhood_mean_abs_deviation_from_observation,
            summary.global_mean_abs_deviation_from_observation,
            summary.neighbourhood_prior_variance_mean,
            summary.neighbourhood_posterior_variance_mean,
            summary.neighbourhood_variance_reduction_mean,
        )?;
    }

    Ok(())
}

fn write_summary_txt(
    out_dir: &PathBuf,
    config: &ExampleConfig,
    result: &Torus0FormConditioningResult,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(out_dir.join("summary.txt"))?);
    let max_obs_error = result
        .observation_summaries
        .iter()
        .map(|summary| summary.abs_error_at_observation)
        .fold(0.0, f64::max);
    let mean_neighbourhood_mae = mean(
        result
            .observation_summaries
            .iter()
            .map(|summary| summary.neighbourhood_mean_abs_deviation_from_observation),
    );
    let mean_local_to_global_mae_ratio = mean(result.observation_summaries.iter().map(|summary| {
        summary.neighbourhood_mean_abs_deviation_from_observation
            / summary.global_mean_abs_deviation_from_observation
    }));
    let mean_observed_variance_ratio = mean(result.observation_summaries.iter().map(|summary| {
        summary.posterior_variance_at_observation / summary.prior_variance_at_observation
    }));
    let mean_prior_variance = mean(result.prior_variance.iter().copied());
    let mean_posterior_variance = mean(result.posterior_variance.iter().copied());

    writeln!(writer, "Matern 0-form torus conditioning")?;
    writeln!(
        writer,
        "mesh_path={}",
        config.conditioning.mesh_path.display()
    )?;
    writeln!(writer, "kappa={}", config.conditioning.kappa)?;
    writeln!(writer, "tau={}", config.conditioning.tau)?;
    writeln!(
        writer,
        "noise_variance={}",
        config.conditioning.noise_variance
    )?;
    writeln!(
        writer,
        "major_radius={}, minor_radius={}",
        result.major_radius, result.minor_radius
    )?;
    writeln!(writer, "effective_range={}", result.effective_range)?;
    writeln!(
        writer,
        "neighbourhood_radius={}",
        result.neighbourhood_radius
    )?;
    writeln!(writer, "num_vertices={}", result.truth.len())?;
    writeln!(
        writer,
        "num_observations={}",
        result.observation_indices.len()
    )?;
    writeln!(writer, "max_observation_abs_error={max_obs_error}")?;
    writeln!(writer, "mean_neighbourhood_mae={mean_neighbourhood_mae}")?;
    writeln!(
        writer,
        "mean_local_to_global_mae_ratio={mean_local_to_global_mae_ratio}"
    )?;
    writeln!(
        writer,
        "mean_observed_variance_ratio={mean_observed_variance_ratio}"
    )?;
    writeln!(writer, "mean_prior_variance={mean_prior_variance}")?;
    writeln!(writer, "mean_posterior_variance={mean_posterior_variance}")?;

    Ok(())
}

fn print_summary(config: &ExampleConfig, result: &Torus0FormConditioningResult) {
    let max_obs_error = result
        .observation_summaries
        .iter()
        .map(|summary| summary.abs_error_at_observation)
        .fold(0.0, f64::max);
    let mean_neighbourhood_mae = mean(
        result
            .observation_summaries
            .iter()
            .map(|summary| summary.neighbourhood_mean_abs_deviation_from_observation),
    );
    let mean_local_to_global_mae_ratio = mean(result.observation_summaries.iter().map(|summary| {
        summary.neighbourhood_mean_abs_deviation_from_observation
            / summary.global_mean_abs_deviation_from_observation
    }));
    let mean_observed_variance_ratio = mean(result.observation_summaries.iter().map(|summary| {
        summary.posterior_variance_at_observation / summary.prior_variance_at_observation
    }));

    println!("mesh: {}", config.conditioning.mesh_path.display());
    println!(
        "vertices: {}, observations: {}",
        result.truth.len(),
        result.observation_indices.len()
    );
    println!(
        "kappa={}, tau={}, noise_variance={}",
        config.conditioning.kappa, config.conditioning.tau, config.conditioning.noise_variance
    );
    println!(
        "effective_range={:.6}, neighbourhood_radius={:.6}",
        result.effective_range, result.neighbourhood_radius
    );
    println!("max observation abs error: {:.6e}", max_obs_error);
    println!(
        "mean neighbourhood abs deviation: {:.6e}",
        mean_neighbourhood_mae
    );
    println!(
        "mean local/global neighbourhood deviation ratio: {:.6}",
        mean_local_to_global_mae_ratio
    );
    println!(
        "mean observed posterior/prior variance ratio: {:.6e}",
        mean_observed_variance_ratio
    );
    println!("per-observation summary:");
    for summary in &result.observation_summaries {
        println!(
            "  obs {} @ vertex {}: y={:.4}, mean={:.4}, abs_err={:.3e}, var_obs(prior/post)=({:.4e}/{:.4e}), neigh_mae={:.4e}, global_mae={:.4e}, neigh_var(prior/post)=({:.4e}/{:.4e})",
            summary.observation_index,
            summary.vertex_index,
            summary.observation_value,
            summary.posterior_mean_at_observation,
            summary.abs_error_at_observation,
            summary.prior_variance_at_observation,
            summary.posterior_variance_at_observation,
            summary.neighbourhood_mean_abs_deviation_from_observation,
            summary.global_mean_abs_deviation_from_observation,
            summary.neighbourhood_prior_variance_mean,
            summary.neighbourhood_posterior_variance_mean,
        );
    }
    println!("wrote outputs to {}", config.out_dir.display());
}

fn build_observation_samples(
    dimension: usize,
    observation_indices: &[usize],
    observation_values: &[f64],
) -> FeecVector {
    let mut samples = FeecVector::zeros(dimension);
    for (&idx, &value) in observation_indices.iter().zip(observation_values.iter()) {
        samples[idx] = value;
    }
    samples
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    values.iter().sum::<f64>() / values.len() as f64
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn print_help() {
    println!("Usage: cargo run --release -p feg-infer --example matern_0form_torus_conditioning [options]");
    println!("  --mesh-path PATH");
    println!("  --kappa FLOAT");
    println!("  --tau FLOAT");
    println!("  --noise-variance FLOAT");
    println!("  --radius-scale FLOAT");
    println!("  --out-dir PATH");
}
