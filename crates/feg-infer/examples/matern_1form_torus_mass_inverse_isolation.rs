use feg_infer::matern_1form::build_hodge_laplacian_1form;
use feg_infer::torus_1form_mass_inverse_isolation::{
    compute_torus_1form_mass_inverse_isolation_report, default_torus_shell_coarse_mesh_path,
    IsolationStrategyReport, Torus1FormMassInverseIsolationReport,
};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct Config {
    mesh_path: PathBuf,
    kappa: f64,
    tau: f64,
    nominal_mesh_size: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mesh_path: default_torus_shell_coarse_mesh_path(),
            kappa: 20.0,
            tau: 1.0,
            nominal_mesh_size: 0.30,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;

    let out_dir = PathBuf::from("out/matern_1form_torus_mass_inverse_isolation");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let report = compute_torus_1form_mass_inverse_isolation_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        config.kappa,
        config.tau,
    )
    .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;

    write_summary(&out_dir.join("summary.txt"), &report, &config)?;
    write_edge_csv(&out_dir.join("edge_comparison.csv"), &report)?;
    print_summary(&report, &config);

    println!("wrote outputs to {}", out_dir.display());
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mesh" => {
                i += 1;
                config.mesh_path = PathBuf::from(parse_next::<String>(&args, i, "--mesh")?);
            }
            "--kappa" => {
                i += 1;
                config.kappa = parse_next(&args, i, "--kappa")?;
            }
            "--tau" => {
                i += 1;
                config.tau = parse_next(&args, i, "--tau")?;
            }
            "--mesh-size" => {
                i += 1;
                config.nominal_mesh_size = parse_next(&args, i, "--mesh-size")?;
            }
            flag => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown flag {flag}"),
                )
                .into())
            }
        }
        i += 1;
    }
    Ok(config)
}

fn parse_next<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let value = args.get(index).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value after {flag}"),
        )
    })?;
    value.parse::<T>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {flag}: {err}"),
        )
        .into()
    })
}

fn write_summary(
    path: &std::path::Path,
    report: &Torus1FormMassInverseIsolationReport,
    config: &Config,
) -> io::Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "mesh_path={}", config.mesh_path.display())?;
    writeln!(file, "nominal_mesh_size={}", config.nominal_mesh_size)?;
    writeln!(file, "edge_dofs={}", report.edge_lengths.len())?;
    writeln!(
        file,
        "harmonic_constraint_rank={}",
        report.harmonic_constraint_rank
    )?;
    writeln!(file, "kappa={}", config.kappa)?;
    writeln!(file, "tau={}", config.tau)?;
    writeln!(file)?;
    write_strategy_summary(&mut file, &report.exact_consistent)?;
    write_strategy_summary(&mut file, &report.projected)?;
    write_strategy_summary(&mut file, &report.row_sum)?;
    writeln!(file)?;
    writeln!(file, "interpretation={}", interpretation_line(report))?;
    Ok(())
}

fn write_strategy_summary(
    mut writer: impl Write,
    strategy: &IsolationStrategyReport,
) -> io::Result<()> {
    writeln!(writer, "[{}]", strategy.kind.label())?;
    writeln!(writer, "H_raw={}", strategy.h_raw)?;
    writeln!(writer, "H_hf={}", strategy.h_hf)?;
    writeln!(writer, "H_post_len={}", strategy.h_post_len)?;
    writeln!(
        writer,
        "harmonic_removed_fraction_mean={}",
        strategy.harmonic_removed_fraction_mean
    )?;
    if let Some(distance) = strategy.distance_to_exact {
        writeln!(writer, "distance_to_exact={distance}")?;
    }
    if let Some(delta) = strategy.delta_h_hf_vs_exact {
        writeln!(writer, "delta_H_hf_vs_exact={delta}")?;
    }
    if let Some(delta) = strategy.delta_h_post_len_vs_exact {
        writeln!(writer, "delta_H_post_len_vs_exact={delta}")?;
    }
    writeln!(writer)?;
    Ok(())
}

fn write_edge_csv(
    path: &std::path::Path,
    report: &Torus1FormMassInverseIsolationReport,
) -> io::Result<()> {
    let mut csv = BufWriter::new(File::create(path)?);
    writeln!(
        csv,
        "edge_index,edge_length,row_sum_unconstrained_variance,projected_unconstrained_variance,exact_unconstrained_variance,row_sum_harmonic_free_variance,projected_harmonic_free_variance,exact_harmonic_free_variance,row_sum_log_hf_per_length2,projected_log_hf_per_length2,exact_log_hf_per_length2,row_sum_harmonic_free_minus_exact,projected_harmonic_free_minus_exact,row_sum_centered_log_hf_per_length2_minus_exact,projected_centered_log_hf_per_length2_minus_exact"
    )?;
    for i in 0..report.edge_lengths.len() {
        writeln!(
            csv,
            "{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            i,
            report.edge_lengths[i],
            report.row_sum.unconstrained_variances[i],
            report.projected.unconstrained_variances[i],
            report.exact_consistent.unconstrained_variances[i],
            report.row_sum.harmonic_free_variances[i],
            report.projected.harmonic_free_variances[i],
            report.exact_consistent.harmonic_free_variances[i],
            report.row_sum.log_harmonic_free_variance_per_length2[i],
            report.projected.log_harmonic_free_variance_per_length2[i],
            report.exact_consistent.log_harmonic_free_variance_per_length2[i],
            report.row_sum.harmonic_free_variances[i] - report.exact_consistent.harmonic_free_variances[i],
            report.projected.harmonic_free_variances[i] - report.exact_consistent.harmonic_free_variances[i],
            report.row_sum.centered_log_harmonic_free_variance_per_length2[i]
                - report.exact_consistent.centered_log_harmonic_free_variance_per_length2[i],
            report.projected.centered_log_harmonic_free_variance_per_length2[i]
                - report.exact_consistent.centered_log_harmonic_free_variance_per_length2[i],
        )?;
    }
    Ok(())
}

fn print_summary(report: &Torus1FormMassInverseIsolationReport, config: &Config) {
    println!("mesh: {}", config.mesh_path.display());
    println!("nominal mesh size: {}", config.nominal_mesh_size);
    println!("edge dofs: {}", report.edge_lengths.len());
    println!(
        "harmonic constraint rank: {}",
        report.harmonic_constraint_rank
    );
    println!("kappa={}, tau={}", config.kappa, config.tau);
    print_strategy(report, &report.exact_consistent);
    print_strategy(report, &report.projected);
    print_strategy(report, &report.row_sum);
    println!("{}", interpretation_line(report));
}

fn print_strategy(
    report: &Torus1FormMassInverseIsolationReport,
    strategy: &IsolationStrategyReport,
) {
    println!("[{}]", strategy.kind.label());
    println!(
        "  H_raw={:.6e}, H_hf={:.6e}, H_post_len={:.6e}",
        strategy.h_raw, strategy.h_hf, strategy.h_post_len
    );
    println!(
        "  harmonic_removed_fraction_mean={:.6e}",
        strategy.harmonic_removed_fraction_mean
    );
    if let Some(distance) = strategy.distance_to_exact {
        println!("  D_to_exact={distance:.6e}");
    }
    if let Some(delta) = strategy.delta_h_hf_vs_exact {
        println!("  delta_H_hf_vs_exact={delta:.6e}");
    }
    if let Some(delta) = strategy.delta_h_post_len_vs_exact {
        println!("  delta_H_post_len_vs_exact={delta:.6e}");
    }
    let _ = report;
}

fn interpretation_line(report: &Torus1FormMassInverseIsolationReport) -> String {
    let row_distance = report.row_sum.distance_to_exact.unwrap_or(f64::NAN);
    let projected_distance = report.projected.distance_to_exact.unwrap_or(f64::NAN);
    if !(row_distance.is_finite() && projected_distance.is_finite()) {
        return "interpretation unavailable because at least one distance is non-finite"
            .to_string();
    }
    if projected_distance <= 0.5 * row_distance {
        format!(
            "projected is materially closer to exact consistent than row-sum (distance ratio {:.3})",
            projected_distance / row_distance
        )
    } else if projected_distance < row_distance {
        format!(
            "projected is closer to exact consistent than row-sum, but only modestly (distance ratio {:.3})",
            projected_distance / row_distance
        )
    } else {
        format!(
            "projected is not closer to exact consistent than row-sum (distance ratio {:.3})",
            projected_distance / row_distance
        )
    }
}
