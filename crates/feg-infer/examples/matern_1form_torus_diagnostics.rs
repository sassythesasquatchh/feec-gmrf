use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use feg_infer::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    compute_matern_1form_torus_inhomogeneity_report, pearson_correlation, ContributionEstimate,
    EstimateWithError, LinearEqualityConstraints, Matern1FormTorusAttributionReport,
};
use feg_infer::matern_1form::{build_hodge_laplacian_1form, MaternConfig, MaternMassInverse};
use feg_infer::util::convert_whittle_params_to_matern;
use feg_infer::vtk::write_1cochain_vtk_fields;
use gmrf_core::types::Vector as GmrfVector;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::time::Instant;

const DEFAULT_RBMC_BATCH_COUNT: usize = 8;

#[derive(Debug, Clone, Copy)]
enum MassInverseMode {
    RowSumLumped,
    Nc1Projected,
    Compare,
}

#[derive(Debug, Clone, Copy)]
struct Config {
    kappa: f64,
    tau: f64,
    num_rbmc_probes: usize,
    rng_seed: u64,
    mass_inverse_mode: MassInverseMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kappa: 20.0,
            tau: 1.0,
            num_rbmc_probes: 1024,
            rng_seed: 13,
            mass_inverse_mode: MassInverseMode::Compare,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let total_start = Instant::now();

    let out_dir = "out/matern_1form_torus_diagnostics";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let mesh_path = "meshes/torus_shell.msh";
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let (_nu, _variance, euclidean_effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    let t = Instant::now();
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    println!(
        "assemble hodge operators: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let harmonic_basis = build_analytic_torus_harmonic_basis(&topology, &coords, &metric)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let harmonic_constraint_matrix =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let harmonic_constraint_rhs = GmrfVector::zeros(harmonic_constraint_matrix.nrows());
    let harmonic_constraints = LinearEqualityConstraints {
        matrix: &harmonic_constraint_matrix,
        rhs: &harmonic_constraint_rhs,
    };
    println!(
        "harmonic constraints build ({}x{}): {:.3}s",
        harmonic_constraint_matrix.nrows(),
        harmonic_constraint_matrix.ncols(),
        t.elapsed().as_secs_f64()
    );

    match config.mass_inverse_mode {
        MassInverseMode::RowSumLumped => {
            let report = run_strategy(
                &format!("{out_dir}/row_sum_lumped"),
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::RowSumLumped,
                harmonic_constraints,
            )?;
            println!(
                "single-strategy output: {}",
                strategy_dir_name(report.mass_inverse)
            );
        }
        MassInverseMode::Nc1Projected => {
            let report = run_strategy(
                &format!("{out_dir}/nc1_projected"),
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::Nc1ProjectedSparseInverse,
                harmonic_constraints,
            )?;
            println!(
                "single-strategy output: {}",
                strategy_dir_name(report.mass_inverse)
            );
        }
        MassInverseMode::Compare => {
            let row_sum = run_strategy(
                &format!("{out_dir}/row_sum_lumped"),
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::RowSumLumped,
                harmonic_constraints,
            )?;
            let projected = run_strategy(
                &format!("{out_dir}/nc1_projected"),
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::Nc1ProjectedSparseInverse,
                harmonic_constraints,
            )?;
            write_comparison_summary(out_dir, &row_sum, &projected)?;
            print_comparison_summary(&row_sum, &projected);
            println!("comparison outputs: {out_dir}");
        }
    }

    println!("edge dofs: {}", hodge.mass_u.nrows());
    println!("kappa={}, tau={}", config.kappa, config.tau);
    println!(
        "RBMC probes={}, batches={}",
        config.num_rbmc_probes, DEFAULT_RBMC_BATCH_COUNT
    );
    println!("Euclidean effective range: {euclidean_effective_range}");
    println!("Loaded mesh from {mesh_path}");
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn run_strategy(
    out_dir: &str,
    topology: &manifold::topology::complex::Complex,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    hodge: &feg_infer::matern_1form::HodgeLaplacian1Form,
    config: Config,
    mass_inverse: MaternMassInverse,
    constraints: LinearEqualityConstraints<'_>,
) -> Result<Matern1FormTorusAttributionReport, Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    let t = Instant::now();
    let report = compute_matern_1form_torus_inhomogeneity_report(
        topology,
        coords,
        metric,
        hodge,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse,
        },
        config.num_rbmc_probes,
        config.rng_seed,
        DEFAULT_RBMC_BATCH_COUNT,
        constraints,
    )
    .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
    println!(
        "[{}] attribution build: {:.3}s",
        strategy_dir_name(mass_inverse),
        t.elapsed().as_secs_f64()
    );

    write_strategy_vtk(out_dir, coords, topology, &report)?;
    write_strategy_csv(out_dir, &report)?;
    print_strategy_summary(&report);
    Ok(report)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut config = Config::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--kappa" => {
                i += 1;
                config.kappa = parse_next::<f64>(&args, i, "--kappa")?;
            }
            "--tau" => {
                i += 1;
                config.tau = parse_next::<f64>(&args, i, "--tau")?;
            }
            "--samples" => {
                i += 1;
                config.num_rbmc_probes = parse_next::<usize>(&args, i, "--samples")?;
            }
            "--seed" => {
                i += 1;
                config.rng_seed = parse_next::<u64>(&args, i, "--seed")?;
            }
            "--mass-inverse" => {
                i += 1;
                config.mass_inverse_mode =
                    parse_mass_inverse_mode(args.get(i).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "missing value for --mass-inverse",
                        )
                    })?)?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown flag: {flag}"),
                )
                .into());
            }
        }
        i += 1;
    }
    Ok(config)
}

fn parse_next<T: std::str::FromStr>(
    args: &[String],
    idx: usize,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    <T as std::str::FromStr>::Err: std::error::Error + 'static,
{
    let raw = args.get(idx).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value for {flag}"),
        )
    })?;
    Ok(raw.parse::<T>()?)
}

fn parse_mass_inverse_mode(raw: &str) -> Result<MassInverseMode, Box<dyn std::error::Error>> {
    match raw {
        "row-sum" => Ok(MassInverseMode::RowSumLumped),
        "projected" => Ok(MassInverseMode::Nc1Projected),
        "compare" => Ok(MassInverseMode::Compare),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for --mass-inverse: {raw}"),
        )
        .into()),
    }
}

fn print_usage() {
    println!("Usage: cargo run --release -p feg-infer --example matern_1form_torus_diagnostics -- [options]");
    println!("Options:");
    println!("  --kappa <f64>          Matérn kappa parameter (default: 20.0)");
    println!("  --tau <f64>            Matérn tau parameter (default: 1.0)");
    println!(
        "  --samples <usize>      RBMC probe count for prior-variance estimation (default: 1024)"
    );
    println!("  --seed <u64>           RNG seed for RBMC probes and secondary MC (default: 13)");
    println!("  --mass-inverse <mode>  row-sum | projected | compare (default: compare)");
}

fn write_strategy_vtk(
    out_dir: &str,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    report: &Matern1FormTorusAttributionReport,
) -> io::Result<()> {
    let diagnostics = &report.edge_diagnostics;
    let standardized = &report.standardized_forcing;
    let attribution = &report.torus_attribution;
    let harmonic_free = &attribution.harmonic_free_edge_diagnostics;
    let removed = &attribution.harmonic_removed_edge_diagnostics;
    let geometry = &attribution.geometry;
    let field = &attribution.field_decomposition;

    let variance = Cochain::new(1, diagnostics.variances.clone());
    let harmonic_free_variance = Cochain::new(1, harmonic_free.variances.clone());
    let removed_variance = Cochain::new(1, removed.variances.clone());
    let removed_fraction = Cochain::new(1, attribution.harmonic_removed_fraction.clone());
    let variance_per_length2 = Cochain::new(1, diagnostics.variance_per_length2.clone());
    let harmonic_free_variance_per_length2 =
        Cochain::new(1, harmonic_free.variance_per_length2.clone());
    let standardized_mean = Cochain::new(1, standardized.mean.clone());
    let standardized_variance = Cochain::new(1, standardized.variances.clone());
    let edge_length = Cochain::new(1, diagnostics.edge_lengths.clone());
    let rho = Cochain::new(1, geometry.midpoint_rho.clone());
    let theta = Cochain::new(1, geometry.midpoint_theta.clone());
    let curvature = Cochain::new(1, geometry.gaussian_curvature.clone());
    let toroidal_alignment_sq = Cochain::new(1, geometry.toroidal_alignment_sq.clone());

    write_1cochain_vtk_fields(
        format!("{out_dir}/prior_variance_diagnostics.vtk"),
        coords,
        topology,
        &[
            ("prior_variance_rbmc", &variance),
            ("harmonic_free_variance_est", &harmonic_free_variance),
            ("harmonic_removed_variance_exact", &removed_variance),
            ("harmonic_removed_fraction", &removed_fraction),
            ("variance_per_length2_rbmc", &variance_per_length2),
            (
                "harmonic_free_variance_per_length2_est",
                &harmonic_free_variance_per_length2,
            ),
            ("standardized_forcing_mean_mc", &standardized_mean),
            ("standardized_forcing_variance_mc", &standardized_variance),
            ("edge_length", &edge_length),
            ("midpoint_rho", &rho),
            ("midpoint_theta", &theta),
            ("gaussian_curvature", &curvature),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;

    let log_variance = Cochain::new(1, field.log_unconstrained_variance.clone());
    let log_harmonic_free = Cochain::new(1, field.log_harmonic_free_variance.clone());
    let log_harmonic_free_per_length2 =
        Cochain::new(1, field.log_harmonic_free_variance_per_length2.clone());
    let minor_angle_component = Cochain::new(1, field.minor_angle_component.clone());
    let direction_component = Cochain::new(1, field.direction_component.clone());
    let interaction_component = Cochain::new(1, field.interaction_component.clone());
    let residual_component = Cochain::new(1, field.residual_component.clone());

    write_1cochain_vtk_fields(
        format!("{out_dir}/prior_variance_diagnostics_inhomogeneity.vtk"),
        coords,
        topology,
        &[
            ("log_variance_rbmc", &log_variance),
            ("log_harmonic_free_variance_est", &log_harmonic_free),
            (
                "log_harmonic_free_variance_per_length2",
                &log_harmonic_free_per_length2,
            ),
            ("minor_angle_component", &minor_angle_component),
            ("direction_component", &direction_component),
            ("interaction_component", &interaction_component),
            ("residual_component", &residual_component),
            ("midpoint_rho", &rho),
            ("midpoint_theta", &theta),
            ("gaussian_curvature", &curvature),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )?;

    Ok(())
}

fn write_strategy_csv(out_dir: &str, report: &Matern1FormTorusAttributionReport) -> io::Result<()> {
    let diagnostics = &report.edge_diagnostics;
    let standardized = &report.standardized_forcing;
    let attribution = &report.torus_attribution;
    let harmonic_free = &attribution.harmonic_free_edge_diagnostics;
    let removed = &attribution.harmonic_removed_edge_diagnostics;
    let geometry = &attribution.geometry;
    let field = &attribution.field_decomposition;

    let file = File::create(format!("{out_dir}/edge_diagnostics.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "edge_idx,length,midpoint_rho,midpoint_theta,gaussian_curvature,toroidal_alignment_sq,mass_diag,forcing_scale_diag,variance_rbmc,std_rbmc,harmonic_free_variance_est,harmonic_removed_variance_exact,harmonic_removed_fraction,variance_per_length2_rbmc,harmonic_free_variance_per_length2_est,standardized_mean_mc,standardized_variance_mc"
    )?;
    for i in 0..diagnostics.variances.len() {
        writeln!(
            w,
            "{i},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            diagnostics.edge_lengths[i],
            geometry.midpoint_rho[i],
            geometry.midpoint_theta[i],
            geometry.gaussian_curvature[i],
            geometry.toroidal_alignment_sq[i],
            diagnostics.mass_diag[i],
            diagnostics.forcing_scale_diag[i],
            diagnostics.variances[i],
            diagnostics.std_devs[i],
            harmonic_free.variances[i],
            removed.variances[i],
            attribution.harmonic_removed_fraction[i],
            diagnostics.variance_per_length2[i],
            harmonic_free.variance_per_length2[i],
            standardized.mean[i],
            standardized.variances[i],
        )?;
    }

    let geometry_file = File::create(format!("{out_dir}/edge_geometry.csv"))?;
    let mut gw = BufWriter::new(geometry_file);
    writeln!(
        gw,
        "edge_idx,length,midpoint_rho,midpoint_theta,gaussian_curvature,toroidal_alignment_sq"
    )?;
    for i in 0..diagnostics.edge_lengths.len() {
        writeln!(
            gw,
            "{i},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            diagnostics.edge_lengths[i],
            geometry.midpoint_rho[i],
            geometry.midpoint_theta[i],
            geometry.gaussian_curvature[i],
            geometry.toroidal_alignment_sq[i],
        )?;
    }

    let attribution_file = File::create(format!("{out_dir}/edge_inhomogeneity_attribution.csv"))?;
    let mut aw = BufWriter::new(attribution_file);
    writeln!(
        aw,
        "edge_idx,log_variance_rbmc,log_harmonic_free_variance_est,log_harmonic_free_variance_per_length2,minor_angle_component,direction_component,interaction_component,residual_component"
    )?;
    for i in 0..field.log_unconstrained_variance.len() {
        writeln!(
            aw,
            "{i},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            field.log_unconstrained_variance[i],
            field.log_harmonic_free_variance[i],
            field.log_harmonic_free_variance_per_length2[i],
            field.minor_angle_component[i],
            field.direction_component[i],
            field.interaction_component[i],
            field.residual_component[i],
        )?;
    }

    Ok(())
}

fn print_strategy_summary(report: &Matern1FormTorusAttributionReport) {
    let label = strategy_dir_name(report.mass_inverse);
    let attribution = &report.torus_attribution;
    let geometry = &attribution.geometry;
    let field = &attribution.field_decomposition;

    println!("[{label}]");
    print_estimate("H(log variance)", &attribution.contribution_summary.total);
    print_contribution("C_harm", &attribution.contribution_summary.harmonic);
    print_contribution("C_len", &attribution.contribution_summary.edge_length);
    print_contribution("C_minor", &attribution.contribution_summary.minor_angle);
    print_contribution("C_direction", &attribution.contribution_summary.direction);
    print_contribution(
        "C_interaction",
        &attribution.contribution_summary.interaction,
    );
    print_contribution("C_residual", &attribution.contribution_summary.residual);
    println!(
        "RBMC floor hits={} harmonic-free floor hits={}",
        attribution.rbmc_floor_hits, attribution.harmonic_free_floor_hits
    );
    print_corr(
        "corr(log variance, rho)",
        pearson_correlation(&field.log_unconstrained_variance, &geometry.midpoint_rho),
    );
    print_corr(
        "corr(log harmonic-free variance / length^2, curvature)",
        pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.gaussian_curvature,
        ),
    );
    print_corr(
        "corr(log harmonic-free variance / length^2, toroidal_alignment_sq)",
        pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.toroidal_alignment_sq,
        ),
    );
    let removed_log = positive_log_vector(&attribution.harmonic_removed_edge_diagnostics.variances);
    print_corr(
        "corr(log harmonic-removed variance, curvature)",
        pearson_correlation(&removed_log, &geometry.gaussian_curvature),
    );
}

fn write_comparison_summary(
    out_dir: &str,
    row_sum: &Matern1FormTorusAttributionReport,
    projected: &Matern1FormTorusAttributionReport,
) -> io::Result<()> {
    let mut txt = BufWriter::new(File::create(format!("{out_dir}/comparison_summary.txt"))?);
    writeln!(txt, "Matérn 1-form torus inhomogeneity attribution")?;
    writeln!(
        txt,
        "row_sum strategy: {}",
        strategy_dir_name(row_sum.mass_inverse)
    )?;
    writeln!(
        txt,
        "projected strategy: {}",
        strategy_dir_name(projected.mass_inverse)
    )?;
    writeln!(txt, "primary score: H(v) = Var(log variance)")?;
    writeln!(
        txt,
        "variance field: paired RBMC estimate; harmonic correction: exact low-rank diagonal"
    )?;
    writeln!(
        txt,
        "uncertainty: mean ± 2SE across {} RBMC batches",
        row_sum.torus_attribution.rbmc_batch_sizes.len()
    )?;
    writeln!(txt)?;

    write_strategy_block(&mut txt, "row_sum", row_sum)?;
    write_strategy_block(&mut txt, "projected", projected)?;

    writeln!(txt, "comparison:")?;
    write_comparison_line(
        &mut txt,
        "H(log variance)",
        row_sum.torus_attribution.contribution_summary.total,
        projected.torus_attribution.contribution_summary.total,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_harm",
        row_sum.torus_attribution.contribution_summary.harmonic,
        projected.torus_attribution.contribution_summary.harmonic,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_len",
        row_sum.torus_attribution.contribution_summary.edge_length,
        projected.torus_attribution.contribution_summary.edge_length,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_minor",
        row_sum.torus_attribution.contribution_summary.minor_angle,
        projected.torus_attribution.contribution_summary.minor_angle,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_direction",
        row_sum.torus_attribution.contribution_summary.direction,
        projected.torus_attribution.contribution_summary.direction,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_interaction",
        row_sum.torus_attribution.contribution_summary.interaction,
        projected.torus_attribution.contribution_summary.interaction,
    )?;
    write_comparison_contribution(
        &mut txt,
        "C_residual",
        row_sum.torus_attribution.contribution_summary.residual,
        projected.torus_attribution.contribution_summary.residual,
    )?;

    let mut csv = BufWriter::new(File::create(format!(
        "{out_dir}/comparison_attribution.csv"
    ))?);
    writeln!(
        csv,
        "metric,row_sum_mean,row_sum_2se,row_sum_fraction_mean,row_sum_fraction_2se,projected_mean,projected_2se,projected_fraction_mean,projected_fraction_2se"
    )?;
    write_comparison_csv_total(
        &mut csv,
        "H(log variance)",
        row_sum.torus_attribution.contribution_summary.total,
        projected.torus_attribution.contribution_summary.total,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_harm",
        row_sum.torus_attribution.contribution_summary.harmonic,
        projected.torus_attribution.contribution_summary.harmonic,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_len",
        row_sum.torus_attribution.contribution_summary.edge_length,
        projected.torus_attribution.contribution_summary.edge_length,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_minor",
        row_sum.torus_attribution.contribution_summary.minor_angle,
        projected.torus_attribution.contribution_summary.minor_angle,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_direction",
        row_sum.torus_attribution.contribution_summary.direction,
        projected.torus_attribution.contribution_summary.direction,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_interaction",
        row_sum.torus_attribution.contribution_summary.interaction,
        projected.torus_attribution.contribution_summary.interaction,
    )?;
    write_comparison_csv_contribution(
        &mut csv,
        "C_residual",
        row_sum.torus_attribution.contribution_summary.residual,
        projected.torus_attribution.contribution_summary.residual,
    )?;

    Ok(())
}

fn write_strategy_block(
    writer: &mut impl Write,
    label: &str,
    report: &Matern1FormTorusAttributionReport,
) -> io::Result<()> {
    let attribution = &report.torus_attribution;
    let geometry = &attribution.geometry;
    let field = &attribution.field_decomposition;
    writeln!(writer, "{label}:")?;
    writeln!(
        writer,
        "  H(log variance)={}",
        format_estimate(&attribution.contribution_summary.total)
    )?;
    writeln!(
        writer,
        "  C_harm={}",
        format_contribution(&attribution.contribution_summary.harmonic)
    )?;
    writeln!(
        writer,
        "  C_len={}",
        format_contribution(&attribution.contribution_summary.edge_length)
    )?;
    writeln!(
        writer,
        "  C_minor={}",
        format_contribution(&attribution.contribution_summary.minor_angle)
    )?;
    writeln!(
        writer,
        "  C_direction={}",
        format_contribution(&attribution.contribution_summary.direction)
    )?;
    writeln!(
        writer,
        "  C_interaction={}",
        format_contribution(&attribution.contribution_summary.interaction)
    )?;
    writeln!(
        writer,
        "  C_residual={}",
        format_contribution(&attribution.contribution_summary.residual)
    )?;
    writeln!(
        writer,
        "  rbmc_floor_hits={} harmonic_free_floor_hits={}",
        attribution.rbmc_floor_hits, attribution.harmonic_free_floor_hits
    )?;
    writeln!(
        writer,
        "  corr_log_variance_rho={}",
        format_optional_float(pearson_correlation(
            &field.log_unconstrained_variance,
            &geometry.midpoint_rho
        ))
    )?;
    writeln!(
        writer,
        "  corr_log_harmonic_free_variance_per_length2_curvature={}",
        format_optional_float(pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.gaussian_curvature
        ))
    )?;
    writeln!(
        writer,
        "  corr_log_harmonic_free_variance_per_length2_toroidal_alignment_sq={}",
        format_optional_float(pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.toroidal_alignment_sq
        ))
    )?;
    Ok(())
}

fn print_comparison_summary(
    row_sum: &Matern1FormTorusAttributionReport,
    projected: &Matern1FormTorusAttributionReport,
) {
    println!("[comparison]");
    print_comparison_line(
        "H(log variance)",
        row_sum.torus_attribution.contribution_summary.total,
        projected.torus_attribution.contribution_summary.total,
    );
    print_comparison_contribution(
        "C_harm",
        row_sum.torus_attribution.contribution_summary.harmonic,
        projected.torus_attribution.contribution_summary.harmonic,
    );
    print_comparison_contribution(
        "C_len",
        row_sum.torus_attribution.contribution_summary.edge_length,
        projected.torus_attribution.contribution_summary.edge_length,
    );
    print_comparison_contribution(
        "C_minor",
        row_sum.torus_attribution.contribution_summary.minor_angle,
        projected.torus_attribution.contribution_summary.minor_angle,
    );
    print_comparison_contribution(
        "C_direction",
        row_sum.torus_attribution.contribution_summary.direction,
        projected.torus_attribution.contribution_summary.direction,
    );
    print_comparison_contribution(
        "C_interaction",
        row_sum.torus_attribution.contribution_summary.interaction,
        projected.torus_attribution.contribution_summary.interaction,
    );
    print_comparison_contribution(
        "C_residual",
        row_sum.torus_attribution.contribution_summary.residual,
        projected.torus_attribution.contribution_summary.residual,
    );
}

fn print_estimate(label: &str, estimate: &EstimateWithError) {
    println!("{label}={}", format_estimate(estimate));
}

fn print_contribution(label: &str, estimate: &ContributionEstimate) {
    println!("{label}={}", format_contribution(estimate));
}

fn print_corr(label: &str, value: Option<f64>) {
    println!("{label}={}", format_optional_float(value));
}

fn print_comparison_line(label: &str, row_sum: EstimateWithError, projected: EstimateWithError) {
    println!(
        "{label}: row_sum={} projected={} delta={:.6e}",
        format_estimate(&row_sum),
        format_estimate(&projected),
        projected.mean - row_sum.mean
    );
}

fn print_comparison_contribution(
    label: &str,
    row_sum: ContributionEstimate,
    projected: ContributionEstimate,
) {
    println!(
        "{label}: row_sum={} projected={} delta={:.6e}",
        format_contribution(&row_sum),
        format_contribution(&projected),
        projected.absolute.mean - row_sum.absolute.mean
    );
}

fn write_comparison_line(
    writer: &mut impl Write,
    label: &str,
    row_sum: EstimateWithError,
    projected: EstimateWithError,
) -> io::Result<()> {
    writeln!(
        writer,
        "  {label}: row_sum={} projected={} delta={:.6e}",
        format_estimate(&row_sum),
        format_estimate(&projected),
        projected.mean - row_sum.mean
    )
}

fn write_comparison_contribution(
    writer: &mut impl Write,
    label: &str,
    row_sum: ContributionEstimate,
    projected: ContributionEstimate,
) -> io::Result<()> {
    writeln!(
        writer,
        "  {label}: row_sum={} projected={} delta={:.6e}",
        format_contribution(&row_sum),
        format_contribution(&projected),
        projected.absolute.mean - row_sum.absolute.mean
    )
}

fn write_comparison_csv_total(
    writer: &mut impl Write,
    label: &str,
    row_sum: EstimateWithError,
    projected: EstimateWithError,
) -> io::Result<()> {
    writeln!(
        writer,
        "{label},{:.12e},{:.12e},,,{:.12e},{:.12e},,",
        row_sum.mean,
        2.0 * row_sum.standard_error,
        projected.mean,
        2.0 * projected.standard_error,
    )
}

fn write_comparison_csv_contribution(
    writer: &mut impl Write,
    label: &str,
    row_sum: ContributionEstimate,
    projected: ContributionEstimate,
) -> io::Result<()> {
    writeln!(
        writer,
        "{label},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
        row_sum.absolute.mean,
        2.0 * row_sum.absolute.standard_error,
        row_sum.fraction_of_total.mean,
        2.0 * row_sum.fraction_of_total.standard_error,
        projected.absolute.mean,
        2.0 * projected.absolute.standard_error,
        projected.fraction_of_total.mean,
        2.0 * projected.fraction_of_total.standard_error,
    )
}

fn format_estimate(estimate: &EstimateWithError) -> String {
    format!(
        "{:.6e} ± {:.6e}",
        estimate.mean,
        2.0 * estimate.standard_error
    )
}

fn format_contribution(estimate: &ContributionEstimate) -> String {
    format!(
        "{:.6e} ± {:.6e} ({} ± {})",
        estimate.absolute.mean,
        2.0 * estimate.absolute.standard_error,
        format_percent(estimate.fraction_of_total.mean),
        format_percent(2.0 * estimate.fraction_of_total.standard_error)
    )
}

fn format_optional_float(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_percent(value: f64) -> String {
    format!("{:.6}%", value * 100.0)
}

fn positive_log_vector(values: &FeecVector) -> FeecVector {
    let positive_mean = values
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .sum::<f64>()
        / values.iter().filter(|value| **value > 0.0).count().max(1) as f64;
    let floor = positive_mean.abs().max(1.0) * 1e-12;
    values.map(|value| value.max(floor).ln())
}

fn strategy_dir_name(mass_inverse: MaternMassInverse) -> &'static str {
    match mass_inverse {
        MaternMassInverse::RowSumLumped => "row_sum_lumped",
        MaternMassInverse::Nc1ProjectedSparseInverse => "nc1_projected",
    }
}
