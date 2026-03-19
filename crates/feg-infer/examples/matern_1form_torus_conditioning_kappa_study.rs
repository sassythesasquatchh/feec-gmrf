use feg_infer::torus_1form_conditioning::{
    run_torus_1form_conditioning, write_torus_1form_conditioning_outputs,
    SurfaceVectorVarianceMode, Torus1FormBranchResult, Torus1FormConditioningConfig,
    Torus1FormConditioningResult,
};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_KAPPAS: &[f64] = &[2.0, 4.0, 8.0];
const DEFAULT_DISTANCE_BAND_EDGES: &[f64] = &[0.2, 0.4, 0.8, 1.2];

#[derive(Debug, Clone)]
struct Config {
    conditioning: Torus1FormConditioningConfig,
    kappas: Vec<f64>,
    distance_band_edges: Vec<f64>,
    out_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct SummaryRow {
    kappa: f64,
    branch: &'static str,
    effective_range: f64,
    neighbourhood_radius: f64,
    far_radius: f64,
    observed_max_abs_error: f64,
    observed_variance_ratio_mean: f64,
    near_observation_deviation: f64,
    far_observation_deviation: f64,
    near_variance_ratio_mean: f64,
    far_variance_ratio_mean: f64,
    harmonic_coeff_truth_0: f64,
    harmonic_coeff_truth_1: f64,
    harmonic_coeff_posterior_0: f64,
    harmonic_coeff_posterior_1: f64,
}

#[derive(Debug, Clone)]
struct DistanceBandRow {
    kappa: f64,
    branch: &'static str,
    band_index: usize,
    distance_min: f64,
    distance_max: Option<f64>,
    count: usize,
    mean_observation_relative_deviation: f64,
    mean_variance_ratio: f64,
    mean_harmonic_free_variance_ratio: f64,
}

#[derive(Debug, Clone)]
struct VariancePatternRow {
    kappa: f64,
    branch: &'static str,
    object: &'static str,
    very_local_ratio: f64,
    local_ratio: f64,
    range_ratio: f64,
    far_ratio: f64,
    localization_auc: f64,
    monotonicity_score: f64,
    very_local_orientation_contrast: f64,
    local_orientation_contrast: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            conditioning: Torus1FormConditioningConfig::default(),
            kappas: DEFAULT_KAPPAS.to_vec(),
            distance_band_edges: DEFAULT_DISTANCE_BAND_EDGES.to_vec(),
            out_dir: PathBuf::from("out/matern_1form_torus_conditioning_kappa_study"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let total_start = Instant::now();
    let _ = fs::remove_dir_all(&config.out_dir);
    fs::create_dir_all(&config.out_dir)?;

    let mut summary_rows = Vec::new();
    let mut distance_rows = Vec::new();
    let mut variance_pattern_rows = Vec::new();
    for &kappa in &config.kappas {
        let mut conditioning = config.conditioning.clone();
        conditioning.kappa = kappa;

        let result = run_torus_1form_conditioning(&conditioning)?;
        let run_dir = config
            .out_dir
            .join(format!("kappa_{}", parameter_tag(kappa)));
        write_torus_1form_conditioning_outputs(&result, &run_dir)?;

        summary_rows.extend(summary_rows_for_result(kappa, &result));
        distance_rows.extend(distance_rows_for_result(
            kappa,
            &result,
            &config.distance_band_edges,
        ));
        variance_pattern_rows.extend(variance_pattern_rows_for_result(kappa, &result));

        println!(
            "[kappa={kappa}] effective_range={} wrote {}",
            result.effective_range,
            run_dir.display()
        );
    }

    write_summary_csv(&config.out_dir, &summary_rows)?;
    write_distance_profile_csv(&config.out_dir, &distance_rows)?;
    write_variance_pattern_summary_csv(&config.out_dir, &variance_pattern_rows)?;
    write_summary_txt(&config, &summary_rows, &variance_pattern_rows)?;

    println!("Torus 1-form Matérn conditioning kappa study");
    println!(
        "mesh={} tau={} noise_variance={}",
        config.conditioning.mesh_path.display(),
        config.conditioning.tau,
        config.conditioning.noise_variance
    );
    println!(
        "surface_vector_variance_mode={} rbmc_probes={} rbmc_batches={} seed={}",
        config.conditioning.surface_vector_variance_mode.as_str(),
        config.conditioning.num_rbmc_probes,
        config.conditioning.rbmc_batch_count,
        config.conditioning.rng_seed
    );
    println!("kappas={:?}", config.kappas);
    println!("distance_bands={:?}", config.distance_band_edges);
    println!("wrote outputs to {}", config.out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-path" => {
                config.conditioning.mesh_path = PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --mesh-path"))?,
                );
            }
            "--kappas" => {
                config.kappas = parse_f64_list(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --kappas"))?,
                    "--kappas",
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
            "--surface-variance-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --surface-variance-mode"))?;
                config.conditioning.surface_vector_variance_mode = value
                    .parse::<SurfaceVectorVarianceMode>()
                    .map_err(invalid_input)?;
            }
            "--radius-scale" => {
                config.conditioning.neighbourhood_radius_scale = parse_f64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --radius-scale"))?,
                    "--radius-scale",
                )?;
            }
            "--num-rbmc-probes" => {
                config.conditioning.num_rbmc_probes = parse_usize_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --num-rbmc-probes"))?,
                    "--num-rbmc-probes",
                )?;
            }
            "--rbmc-batch-count" => {
                config.conditioning.rbmc_batch_count = parse_usize_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --rbmc-batch-count"))?,
                    "--rbmc-batch-count",
                )?;
            }
            "--seed" => {
                config.conditioning.rng_seed = parse_u64_arg(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --seed"))?,
                    "--seed",
                )?;
            }
            "--distance-bands" => {
                config.distance_band_edges = parse_f64_list(
                    args.next()
                        .ok_or_else(|| invalid_input("missing value for --distance-bands"))?,
                    "--distance-bands",
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

    validate_args(&config)?;
    Ok(config)
}

fn validate_args(config: &Config) -> Result<(), Box<dyn std::error::Error>> {
    if config.kappas.is_empty() {
        return Err(invalid_input("at least one kappa value is required").into());
    }
    if config
        .kappas
        .iter()
        .any(|kappa| !kappa.is_finite() || *kappa <= 0.0)
    {
        return Err(invalid_input("kappa values must be finite and positive").into());
    }
    if config
        .distance_band_edges
        .iter()
        .any(|edge| !edge.is_finite() || *edge <= 0.0)
    {
        return Err(invalid_input("distance band edges must be finite and positive").into());
    }
    if config
        .distance_band_edges
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(invalid_input("distance band edges must be strictly increasing").into());
    }
    Ok(())
}

fn parse_f64_arg(value: String, flag: &str) -> Result<f64, io::Error> {
    value
        .parse::<f64>()
        .map_err(|_| invalid_input(format!("failed to parse {flag} as f64")))
}

fn parse_usize_arg(value: String, flag: &str) -> Result<usize, io::Error> {
    value
        .parse::<usize>()
        .map_err(|_| invalid_input(format!("failed to parse {flag} as usize")))
}

fn parse_u64_arg(value: String, flag: &str) -> Result<u64, io::Error> {
    value
        .parse::<u64>()
        .map_err(|_| invalid_input(format!("failed to parse {flag} as u64")))
}

fn parse_f64_list(value: String, flag: &str) -> Result<Vec<f64>, io::Error> {
    value
        .split(',')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            entry
                .trim()
                .parse::<f64>()
                .map_err(|_| invalid_input(format!("failed to parse {flag} entry as f64")))
        })
        .collect()
}

fn summary_rows_for_result(kappa: f64, result: &Torus1FormConditioningResult) -> Vec<SummaryRow> {
    [
        &result.harmonic_free_constrained,
        &result.full_unconstrained,
    ]
    .into_iter()
    .map(|branch| {
        let (near_observation_deviation, far_observation_deviation) =
            observation_relative_stats(branch, result.neighbourhood_radius, result.far_radius);
        SummaryRow {
            kappa,
            branch: branch.name,
            effective_range: result.effective_range,
            neighbourhood_radius: result.neighbourhood_radius,
            far_radius: result.far_radius,
            observed_max_abs_error: branch.summary.observed.max_abs_error,
            observed_variance_ratio_mean: branch.summary.observed.variance_ratio_mean,
            near_observation_deviation,
            far_observation_deviation,
            near_variance_ratio_mean: branch.summary.near.variance_ratio_mean,
            far_variance_ratio_mean: branch.summary.far.variance_ratio_mean,
            harmonic_coeff_truth_0: branch.harmonic_coefficients_truth[0],
            harmonic_coeff_truth_1: branch.harmonic_coefficients_truth[1],
            harmonic_coeff_posterior_0: branch.harmonic_coefficients_posterior_mean[0],
            harmonic_coeff_posterior_1: branch.harmonic_coefficients_posterior_mean[1],
        }
    })
    .collect()
}

fn distance_rows_for_result(
    kappa: f64,
    result: &Torus1FormConditioningResult,
    distance_band_edges: &[f64],
) -> Vec<DistanceBandRow> {
    let mut rows = Vec::new();
    for branch in [
        &result.harmonic_free_constrained,
        &result.full_unconstrained,
    ] {
        for (band_index, (distance_min, distance_max)) in
            distance_bands(distance_band_edges).into_iter().enumerate()
        {
            if let Some(row) =
                summarize_distance_band(kappa, branch, band_index, distance_min, distance_max)
            {
                rows.push(row);
            }
        }
    }
    rows
}

fn variance_pattern_rows_for_result(
    kappa: f64,
    result: &Torus1FormConditioningResult,
) -> Vec<VariancePatternRow> {
    [
        &result.harmonic_free_constrained,
        &result.full_unconstrained,
    ]
    .into_iter()
    .flat_map(|branch| {
        branch
            .variance_pattern
            .summary_rows
            .iter()
            .map(move |row| VariancePatternRow {
                kappa,
                branch: branch.name,
                object: row.object,
                very_local_ratio: row.very_local_ratio,
                local_ratio: row.local_ratio,
                range_ratio: row.range_ratio,
                far_ratio: row.far_ratio,
                localization_auc: row.localization_auc,
                monotonicity_score: row.monotonicity_score,
                very_local_orientation_contrast: row.very_local_orientation_contrast,
                local_orientation_contrast: row.local_orientation_contrast,
            })
    })
    .collect()
}

fn distance_bands(distance_band_edges: &[f64]) -> Vec<(f64, Option<f64>)> {
    let mut bands = Vec::with_capacity(distance_band_edges.len() + 1);
    let mut lower = 0.0;
    for &upper in distance_band_edges {
        bands.push((lower, Some(upper)));
        lower = upper;
    }
    bands.push((lower, None));
    bands
}

fn summarize_distance_band(
    kappa: f64,
    branch: &Torus1FormBranchResult,
    band_index: usize,
    distance_min: f64,
    distance_max: Option<f64>,
) -> Option<DistanceBandRow> {
    let mut count = 0_usize;
    let mut deviation_sum = 0.0;
    let mut variance_ratio_sum = 0.0;
    let mut harmonic_free_variance_ratio_sum = 0.0;

    for edge_index in 0..branch.posterior_mean.len() {
        let distance = branch.nearest_observation_distance[edge_index];
        let in_band = match distance_max {
            Some(distance_max) => distance > distance_min && distance <= distance_max,
            None => distance > distance_min,
        };
        if !in_band {
            continue;
        }

        deviation_sum += (branch.posterior_mean[edge_index]
            - branch.nearest_observation_value[edge_index])
            .abs();
        variance_ratio_sum += safe_ratio(
            branch.posterior_variance[edge_index],
            branch.prior_variance[edge_index],
        );
        harmonic_free_variance_ratio_sum += safe_ratio(
            branch.harmonic_free_posterior_variance[edge_index],
            branch.harmonic_free_prior_variance[edge_index],
        );
        count += 1;
    }

    if count == 0 {
        return None;
    }

    let count_f64 = count as f64;
    Some(DistanceBandRow {
        kappa,
        branch: branch.name,
        band_index,
        distance_min,
        distance_max,
        count,
        mean_observation_relative_deviation: deviation_sum / count_f64,
        mean_variance_ratio: variance_ratio_sum / count_f64,
        mean_harmonic_free_variance_ratio: harmonic_free_variance_ratio_sum / count_f64,
    })
}

fn observation_relative_stats(
    branch: &Torus1FormBranchResult,
    neighbourhood_radius: f64,
    far_radius: f64,
) -> (f64, f64) {
    let mut near_sum = 0.0;
    let mut near_count = 0_usize;
    let mut far_sum = 0.0;
    let mut far_count = 0_usize;

    for edge_index in 0..branch.posterior_mean.len() {
        let distance = branch.nearest_observation_distance[edge_index];
        let deviation = (branch.posterior_mean[edge_index]
            - branch.nearest_observation_value[edge_index])
            .abs();
        if distance <= neighbourhood_radius {
            near_sum += deviation;
            near_count += 1;
        }
        if distance > far_radius {
            far_sum += deviation;
            far_count += 1;
        }
    }

    (
        safe_mean(near_sum, near_count),
        safe_mean(far_sum, far_count),
    )
}

fn write_summary_csv(
    out_dir: &Path,
    rows: &[SummaryRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(out_dir.join("kappa_summary.csv"))?);
    writeln!(
        writer,
        "kappa,branch,effective_range,neighbourhood_radius,far_radius,observed_max_abs_error,observed_variance_ratio_mean,near_observation_deviation,far_observation_deviation,near_variance_ratio_mean,far_variance_ratio_mean,harmonic_coeff_truth_0,harmonic_coeff_truth_1,harmonic_coeff_posterior_0,harmonic_coeff_posterior_1"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.kappa,
            row.branch,
            row.effective_range,
            row.neighbourhood_radius,
            row.far_radius,
            row.observed_max_abs_error,
            row.observed_variance_ratio_mean,
            row.near_observation_deviation,
            row.far_observation_deviation,
            row.near_variance_ratio_mean,
            row.far_variance_ratio_mean,
            row.harmonic_coeff_truth_0,
            row.harmonic_coeff_truth_1,
            row.harmonic_coeff_posterior_0,
            row.harmonic_coeff_posterior_1,
        )?;
    }
    Ok(())
}

fn write_distance_profile_csv(
    out_dir: &Path,
    rows: &[DistanceBandRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(out_dir.join("distance_profile.csv"))?);
    writeln!(
        writer,
        "kappa,branch,band_index,distance_min,distance_max,count,mean_observation_relative_deviation,mean_variance_ratio,mean_harmonic_free_variance_ratio"
    )?;
    for row in rows {
        let distance_max = row
            .distance_max
            .map(|value| format!("{value:.12}"))
            .unwrap_or_else(|| "inf".to_string());
        writeln!(
            writer,
            "{:.12},{},{},{:.12},{},{},{:.12},{:.12},{:.12}",
            row.kappa,
            row.branch,
            row.band_index,
            row.distance_min,
            distance_max,
            row.count,
            row.mean_observation_relative_deviation,
            row.mean_variance_ratio,
            row.mean_harmonic_free_variance_ratio,
        )?;
    }
    Ok(())
}

fn write_variance_pattern_summary_csv(
    out_dir: &Path,
    rows: &[VariancePatternRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(
        out_dir.join("variance_pattern_kappa_summary.csv"),
    )?);
    writeln!(
        writer,
        "kappa,branch,object,very_local_ratio,local_ratio,range_ratio,far_ratio,localization_auc,monotonicity_score,very_local_orientation_contrast,local_orientation_contrast"
    )?;
    for row in rows {
        writeln!(
            writer,
            "{:.12},{},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}",
            row.kappa,
            row.branch,
            row.object,
            row.very_local_ratio,
            row.local_ratio,
            row.range_ratio,
            row.far_ratio,
            row.localization_auc,
            row.monotonicity_score,
            row.very_local_orientation_contrast,
            row.local_orientation_contrast,
        )?;
    }
    Ok(())
}

fn write_summary_txt(
    config: &Config,
    rows: &[SummaryRow],
    variance_pattern_rows: &[VariancePatternRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(config.out_dir.join("summary.txt"))?);
    writeln!(writer, "Torus 1-form Matérn conditioning kappa study")?;
    writeln!(
        writer,
        "mesh_path={}",
        config.conditioning.mesh_path.display()
    )?;
    writeln!(writer, "tau={}", config.conditioning.tau)?;
    writeln!(
        writer,
        "noise_variance={}",
        config.conditioning.noise_variance
    )?;
    writeln!(
        writer,
        "surface_vector_variance_mode={}",
        config.conditioning.surface_vector_variance_mode.as_str()
    )?;
    writeln!(
        writer,
        "num_rbmc_probes={}",
        config.conditioning.num_rbmc_probes
    )?;
    writeln!(
        writer,
        "rbmc_batch_count={}",
        config.conditioning.rbmc_batch_count
    )?;
    writeln!(writer, "seed={}", config.conditioning.rng_seed)?;
    writeln!(writer, "kappas={:?}", config.kappas)?;
    writeln!(
        writer,
        "distance_band_edges={:?}",
        config.distance_band_edges
    )?;
    for row in rows {
        writeln!(
            writer,
            "kappa={} branch={} effective_range={} observed_max_abs_error={} near_variance_ratio_mean={} far_variance_ratio_mean={}",
            row.kappa,
            row.branch,
            row.effective_range,
            row.observed_max_abs_error,
            row.near_variance_ratio_mean,
            row.far_variance_ratio_mean,
        )?;
    }
    writeln!(
        writer,
        "variance_pattern_note=smoothed_matched and circulation are the primary radial-decay diagnostics; edge_all orientation contrast explains anisotropy"
    )?;
    for row in variance_pattern_rows {
        writeln!(
            writer,
            "kappa={} branch={} object={} very_local_ratio={} local_ratio={} far_ratio={} localization_auc={} monotonicity_score={} very_local_orientation_contrast={} local_orientation_contrast={}",
            row.kappa,
            row.branch,
            row.object,
            row.very_local_ratio,
            row.local_ratio,
            row.far_ratio,
            row.localization_auc,
            row.monotonicity_score,
            row.very_local_orientation_contrast,
            row.local_orientation_contrast,
        )?;
    }
    Ok(())
}

fn parameter_tag(value: f64) -> String {
    format!("{value:.5}").replace('.', "p")
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= 1e-12 {
        0.0
    } else {
        numerator / denominator
    }
}

fn safe_mean(sum: f64, count: usize) -> f64 {
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p feg-infer --example matern_1form_torus_conditioning_kappa_study [options]"
    );
    println!("  --mesh-path <path>            Path to the torus mesh");
    println!("  --kappas <list>               Comma-separated kappa values");
    println!("  --tau <value>                 Matérn tau parameter");
    println!("  --noise-variance <value>      Observation noise variance");
    println!(
        "  --surface-variance-mode <mode>  Surface-vector variance mode: exact | rbmc | rbmc-clipped"
    );
    println!(
        "  --radius-scale <value>        Neighbourhood radius as a multiple of effective range"
    );
    println!("  --num-rbmc-probes <value>     Number of RBMC probes per run");
    println!("  --rbmc-batch-count <value>    Number of RBMC batches");
    println!("  --seed <value>                RBMC RNG seed");
    println!("  --distance-bands <list>       Comma-separated intrinsic distance cutoffs");
    println!("  --out-dir <path>              Output directory");
}
