use ddf::cochain::Cochain;
use feg_infer::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    compute_matern_1form_torus_residual_study_report, pearson_correlation, ContributionEstimate,
    EstimateWithError, LinearEqualityConstraints, Matern1FormTorusResidualStudyReport,
};
use feg_infer::matern_1form::{build_hodge_laplacian_1form, MaternConfig, MaternMassInverse};
use feg_infer::util::{
    convert_whittle_params_to_matern, generate_torus_surface_mesh, mesh_size_tag,
};
use feg_infer::vtk::write_1cochain_vtk_fields;
use gmrf_core::types::Vector as GmrfVector;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_MESH_PATH: &str = "meshes/torus_shell.msh";
const DEFAULT_MAJOR_RADIUS: f64 = 1.0;
const DEFAULT_MINOR_RADIUS: f64 = 0.3;
const DEFAULT_RBMC_BATCH_COUNT: usize = 8;
const DEFAULT_STUDY_MESH_SIZES: &[f64] = &[0.30, 0.20, 0.14, 0.10];

#[derive(Debug, Clone, Copy)]
enum MassInverseMode {
    RowSumLumped,
    Nc1Projected,
    Compare,
}

#[derive(Debug, Clone)]
struct Config {
    kappa: f64,
    tau: f64,
    num_rbmc_probes: usize,
    rng_seed: u64,
    mass_inverse_mode: MassInverseMode,
    mesh_path: PathBuf,
    mesh_size: Option<f64>,
    study_mesh_sizes: Vec<f64>,
    major_radius: f64,
    minor_radius: f64,
}

#[derive(Debug, Clone)]
struct MeshRunSpec {
    label: String,
    display_name: String,
    mesh_path: PathBuf,
    mesh_size: Option<f64>,
}

#[derive(Debug, Clone)]
struct RefinementRow {
    mesh_label: String,
    mesh_size: Option<f64>,
    edge_dofs: usize,
    strategy: MaternMassInverse,
    total_post_length: EstimateWithError,
    position_even_fourier: ContributionEstimate,
    direction_legendre: ContributionEstimate,
    interaction_even: ContributionEstimate,
    discrete_surrogates: ContributionEstimate,
    unexplained: ContributionEstimate,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kappa: 20.0,
            tau: 1.0,
            num_rbmc_probes: 1024,
            rng_seed: 13,
            mass_inverse_mode: MassInverseMode::Compare,
            mesh_path: PathBuf::from(DEFAULT_MESH_PATH),
            mesh_size: None,
            study_mesh_sizes: DEFAULT_STUDY_MESH_SIZES.to_vec(),
            major_radius: DEFAULT_MAJOR_RADIUS,
            minor_radius: DEFAULT_MINOR_RADIUS,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let total_start = Instant::now();
    let out_dir = PathBuf::from("out/matern_1form_torus_residual_study");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let mesh_runs = prepare_mesh_runs(&config, &out_dir)?;
    let (_nu, _variance, euclidean_effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    let mut rows = Vec::new();
    for mesh_run in &mesh_runs {
        let mesh_rows = run_mesh(&config, &out_dir, mesh_run)?;
        rows.extend(mesh_rows);
    }

    write_refinement_summary(&out_dir, &rows)?;

    println!("Matérn 1-form torus residual study");
    println!("kappa={}, tau={}", config.kappa, config.tau);
    println!(
        "RBMC probes={}, batches={}",
        config.num_rbmc_probes, DEFAULT_RBMC_BATCH_COUNT
    );
    println!("Euclidean effective range: {euclidean_effective_range}");
    println!("mesh runs={}", mesh_runs.len());
    println!("output={}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());
    Ok(())
}

fn prepare_mesh_runs(
    config: &Config,
    out_dir: &Path,
) -> Result<Vec<MeshRunSpec>, Box<dyn std::error::Error>> {
    let mut runs = Vec::new();
    let mesh_dir = out_dir.join("meshes");
    fs::create_dir_all(&mesh_dir)?;

    let primary = if let Some(mesh_size) = config.mesh_size {
        let label = format!("base_h_{}", mesh_size_tag(mesh_size));
        let mesh_path = mesh_dir.join(format!("{label}.msh"));
        generate_torus_surface_mesh(
            &mesh_path,
            config.major_radius,
            config.minor_radius,
            mesh_size,
        )?;
        MeshRunSpec {
            label,
            display_name: format!("base h={mesh_size:.5}"),
            mesh_path,
            mesh_size: Some(mesh_size),
        }
    } else {
        MeshRunSpec {
            label: "input_mesh".to_string(),
            display_name: format!("input {}", config.mesh_path.display()),
            mesh_path: config.mesh_path.clone(),
            mesh_size: None,
        }
    };
    runs.push(primary);

    let mut seen_sizes = runs
        .iter()
        .filter_map(|run| run.mesh_size)
        .collect::<Vec<_>>();
    for &mesh_size in &config.study_mesh_sizes {
        if seen_sizes
            .iter()
            .any(|seen| (seen - mesh_size).abs() <= 1e-12)
        {
            continue;
        }
        seen_sizes.push(mesh_size);
        let label = format!("study_h_{}", mesh_size_tag(mesh_size));
        let mesh_path = mesh_dir.join(format!("{label}.msh"));
        generate_torus_surface_mesh(
            &mesh_path,
            config.major_radius,
            config.minor_radius,
            mesh_size,
        )?;
        runs.push(MeshRunSpec {
            label,
            display_name: format!("study h={mesh_size:.5}"),
            mesh_path,
            mesh_size: Some(mesh_size),
        });
    }

    Ok(runs)
}

fn run_mesh(
    config: &Config,
    out_dir: &Path,
    mesh_run: &MeshRunSpec,
) -> Result<Vec<RefinementRow>, Box<dyn std::error::Error>> {
    let mesh_bytes = fs::read(&mesh_run.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let t = Instant::now();
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    println!(
        "[{}] assemble hodge operators: {:.3}s",
        mesh_run.display_name,
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
        "[{}] harmonic constraints build ({}x{}): {:.3}s",
        mesh_run.display_name,
        harmonic_constraint_matrix.nrows(),
        harmonic_constraint_matrix.ncols(),
        t.elapsed().as_secs_f64()
    );

    let mesh_out_dir = out_dir.join(&mesh_run.label);
    fs::create_dir_all(&mesh_out_dir)?;

    let mut rows = Vec::new();
    match config.mass_inverse_mode {
        MassInverseMode::RowSumLumped => {
            rows.push(run_strategy(
                &mesh_out_dir.join("row_sum_lumped"),
                mesh_run,
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::RowSumLumped,
                harmonic_constraints,
            )?);
        }
        MassInverseMode::Nc1Projected => {
            rows.push(run_strategy(
                &mesh_out_dir.join("nc1_projected"),
                mesh_run,
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::Nc1ProjectedSparseInverse,
                harmonic_constraints,
            )?);
        }
        MassInverseMode::Compare => {
            rows.push(run_strategy(
                &mesh_out_dir.join("row_sum_lumped"),
                mesh_run,
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::RowSumLumped,
                harmonic_constraints,
            )?);
            rows.push(run_strategy(
                &mesh_out_dir.join("nc1_projected"),
                mesh_run,
                &topology,
                &coords,
                &metric,
                &hodge,
                config,
                MaternMassInverse::Nc1ProjectedSparseInverse,
                harmonic_constraints,
            )?);
        }
    }

    Ok(rows)
}

fn run_strategy(
    out_dir: &Path,
    mesh_run: &MeshRunSpec,
    topology: &manifold::topology::complex::Complex,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    metric: &manifold::geometry::metric::mesh::MeshLengths,
    hodge: &feg_infer::matern_1form::HodgeLaplacian1Form,
    config: &Config,
    mass_inverse: MaternMassInverse,
    constraints: LinearEqualityConstraints<'_>,
) -> Result<RefinementRow, Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;

    let t = Instant::now();
    let report = compute_matern_1form_torus_residual_study_report(
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
        "[{}:{}] residual study build: {:.3}s",
        mesh_run.label,
        strategy_dir_name(mass_inverse),
        t.elapsed().as_secs_f64()
    );

    write_strategy_vtk(out_dir, coords, topology, &report)?;
    write_strategy_csv(out_dir, &report)?;
    write_strategy_summary(out_dir, mesh_run, &report)?;
    print_strategy_summary(mesh_run, &report);

    Ok(RefinementRow {
        mesh_label: mesh_run.label.clone(),
        mesh_size: mesh_run.mesh_size,
        edge_dofs: report.edge_diagnostics.variances.len(),
        strategy: report.mass_inverse,
        total_post_length: report.contribution_summary.total_post_length,
        position_even_fourier: report.contribution_summary.position_even_fourier,
        direction_legendre: report.contribution_summary.direction_legendre,
        interaction_even: report.contribution_summary.interaction_even,
        discrete_surrogates: report.contribution_summary.discrete_surrogates,
        unexplained: report.contribution_summary.unexplained,
    })
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
            "--mesh" => {
                i += 1;
                config.mesh_path = PathBuf::from(parse_next::<String>(&args, i, "--mesh")?);
            }
            "--mesh-size" => {
                i += 1;
                config.mesh_size = Some(parse_next::<f64>(&args, i, "--mesh-size")?);
            }
            "--study-mesh-sizes" => {
                i += 1;
                config.study_mesh_sizes = parse_mesh_sizes(args.get(i).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "missing value for --study-mesh-sizes",
                    )
                })?)?;
            }
            "--major-radius" => {
                i += 1;
                config.major_radius = parse_next::<f64>(&args, i, "--major-radius")?;
            }
            "--minor-radius" => {
                i += 1;
                config.minor_radius = parse_next::<f64>(&args, i, "--minor-radius")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown flag: {other}"),
                )
                .into());
            }
        }
        i += 1;
    }

    if config.kappa <= 0.0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "kappa must be positive").into());
    }
    if config.tau <= 0.0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "tau must be positive").into());
    }
    if config.num_rbmc_probes == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "samples must be at least 1").into(),
        );
    }
    if let Some(mesh_size) = config.mesh_size {
        if mesh_size <= 0.0 {
            return Err(
                io::Error::new(io::ErrorKind::InvalidInput, "mesh-size must be positive").into(),
            );
        }
    }
    if config.major_radius <= 0.0 || config.minor_radius <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "major-radius and minor-radius must be positive",
        )
        .into());
    }

    Ok(config)
}

fn parse_mesh_sizes(raw: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = trimmed.parse::<f64>().map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid mesh size '{trimmed}': {err}"),
            )
        })?;
        if value <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("mesh sizes must be positive, got {value}"),
            )
            .into());
        }
        out.push(value);
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "study-mesh-sizes must contain at least one positive value",
        )
        .into());
    }
    Ok(out)
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
    println!(
        "Usage: cargo run --release -p feg-infer --example matern_1form_torus_residual_study -- [options]"
    );
    println!("Options:");
    println!("  --kappa <f64>               Matérn kappa parameter (default: 20.0)");
    println!("  --tau <f64>                 Matérn tau parameter (default: 1.0)");
    println!("  --samples <usize>           RBMC probe count (default: 1024)");
    println!("  --seed <u64>                RNG seed for RBMC probes (default: 13)");
    println!("  --mass-inverse <mode>       row-sum | projected | compare (default: compare)");
    println!(
        "  --mesh <path>               Input torus mesh path (default: meshes/torus_shell.msh)"
    );
    println!(
        "  --mesh-size <f64>           Generate a primary torus mesh with this target characteristic length"
    );
    println!(
        "  --study-mesh-sizes <list>   Comma-separated refinement study mesh sizes (default: 0.30,0.20,0.14,0.10)"
    );
    println!("  --major-radius <f64>        Major radius for generated meshes (default: 1.0)");
    println!("  --minor-radius <f64>        Minor radius for generated meshes (default: 0.3)");
}

fn write_strategy_vtk(
    out_dir: &Path,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    report: &Matern1FormTorusResidualStudyReport,
) -> io::Result<()> {
    let geometry = &report.geometry;
    let field = &report.field_decomposition;
    let harmonic_free_variance =
        Cochain::new(1, report.harmonic_free_edge_diagnostics.variances.clone());
    let harmonic_removed_variance = Cochain::new(
        1,
        report.harmonic_removed_edge_diagnostics.variances.clone(),
    );
    let harmonic_removed_fraction = Cochain::new(1, report.harmonic_removed_fraction.clone());
    let rho = Cochain::new(1, geometry.midpoint_rho.clone());
    let theta = Cochain::new(1, geometry.midpoint_theta.clone());
    let curvature = Cochain::new(1, geometry.gaussian_curvature.clone());
    let toroidal_alignment_sq = Cochain::new(1, geometry.toroidal_alignment_sq.clone());
    let log_harmonic_free_per_length2 =
        Cochain::new(1, field.log_harmonic_free_variance_per_length2.clone());
    let position_even_fourier = Cochain::new(1, field.position_even_fourier_component.clone());
    let direction_legendre = Cochain::new(1, field.direction_legendre_component.clone());
    let interaction_even = Cochain::new(1, field.interaction_even_component.clone());
    let discrete_surrogate = Cochain::new(1, field.discrete_surrogate_component.clone());
    let unexplained = Cochain::new(1, field.unexplained_residual.clone());

    write_1cochain_vtk_fields(
        out_dir.join("residual_study_edge_fields.vtk"),
        coords,
        topology,
        &[
            ("harmonic_free_variance_est", &harmonic_free_variance),
            (
                "harmonic_removed_variance_exact",
                &harmonic_removed_variance,
            ),
            ("harmonic_removed_fraction", &harmonic_removed_fraction),
            (
                "log_harmonic_free_variance_per_length2",
                &log_harmonic_free_per_length2,
            ),
            ("position_even_fourier_component", &position_even_fourier),
            ("direction_legendre_component", &direction_legendre),
            ("interaction_even_component", &interaction_even),
            ("discrete_surrogate_component", &discrete_surrogate),
            ("unexplained_residual", &unexplained),
            ("midpoint_rho", &rho),
            ("midpoint_theta", &theta),
            ("gaussian_curvature", &curvature),
            ("toroidal_alignment_sq", &toroidal_alignment_sq),
        ],
    )
}

fn write_strategy_csv(
    out_dir: &Path,
    report: &Matern1FormTorusResidualStudyReport,
) -> io::Result<()> {
    let diagnostics = &report.edge_diagnostics;
    let geometry = &report.geometry;
    let field = &report.field_decomposition;
    let harmonic_free = &report.harmonic_free_edge_diagnostics;
    let harmonic_removed = &report.harmonic_removed_edge_diagnostics;

    let file = File::create(out_dir.join("residual_study_edge_fields.csv"))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "edge_idx,length,rho,theta,gaussian_curvature,toroidal_alignment_sq,mass_diag,mass_lumped_diag,forcing_scale_diag,harmonic_free_variance_est,harmonic_removed_variance_exact,harmonic_removed_fraction,log_harmonic_free_variance_per_length2,position_even_fourier_component,direction_legendre_component,interaction_even_component,discrete_surrogate_component,unexplained_residual"
    )?;
    for i in 0..diagnostics.edge_lengths.len() {
        writeln!(
            w,
            "{i},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            diagnostics.edge_lengths[i],
            geometry.midpoint_rho[i],
            geometry.midpoint_theta[i],
            geometry.gaussian_curvature[i],
            geometry.toroidal_alignment_sq[i],
            diagnostics.mass_diag[i],
            diagnostics.mass_lumped_diag[i],
            diagnostics.forcing_scale_diag[i],
            harmonic_free.variances[i],
            harmonic_removed.variances[i],
            report.harmonic_removed_fraction[i],
            field.log_harmonic_free_variance_per_length2[i],
            field.position_even_fourier_component[i],
            field.direction_legendre_component[i],
            field.interaction_even_component[i],
            field.discrete_surrogate_component[i],
            field.unexplained_residual[i],
        )?;
    }
    Ok(())
}

fn write_strategy_summary(
    out_dir: &Path,
    mesh_run: &MeshRunSpec,
    report: &Matern1FormTorusResidualStudyReport,
) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(out_dir.join("residual_study_summary.txt"))?);
    let summary = &report.contribution_summary;
    let field = &report.field_decomposition;
    let geometry = &report.geometry;
    writeln!(w, "Matérn 1-form torus residual study")?;
    writeln!(w, "mesh_label={}", mesh_run.label)?;
    writeln!(w, "mesh_display_name={}", mesh_run.display_name)?;
    writeln!(w, "strategy={}", strategy_dir_name(report.mass_inverse))?;
    if let Some(mesh_size) = mesh_run.mesh_size {
        writeln!(w, "mesh_size={mesh_size:.5}")?;
    }
    writeln!(w, "response=log(harmonic_free_variance / length^2)")?;
    writeln!(
        w,
        "uncertainty=mean +- 2SE across {} RBMC batches",
        report.rbmc_batch_sizes.len()
    )?;
    writeln!(
        w,
        "H_post_len={}",
        format_estimate(&summary.total_post_length)
    )?;
    writeln!(
        w,
        "C_position_even_fourier={}",
        format_contribution(&summary.position_even_fourier)
    )?;
    writeln!(
        w,
        "C_direction_legendre={}",
        format_contribution(&summary.direction_legendre)
    )?;
    writeln!(
        w,
        "C_interaction_even={}",
        format_contribution(&summary.interaction_even)
    )?;
    writeln!(
        w,
        "C_discrete_surrogates={}",
        format_contribution(&summary.discrete_surrogates)
    )?;
    writeln!(
        w,
        "C_unexplained={}",
        format_contribution(&summary.unexplained)
    )?;
    writeln!(
        w,
        "rbmc_floor_hits={} harmonic_free_floor_hits={}",
        report.rbmc_floor_hits, report.harmonic_free_floor_hits
    )?;
    writeln!(
        w,
        "corr_response_curvature={}",
        format_optional_float(pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.gaussian_curvature,
        ))
    )?;
    writeln!(
        w,
        "corr_response_toroidal_alignment_sq={}",
        format_optional_float(pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.toroidal_alignment_sq,
        ))
    )?;
    writeln!(
        w,
        "corr_unexplained_curvature={}",
        format_optional_float(pearson_correlation(
            &field.unexplained_residual,
            &geometry.gaussian_curvature,
        ))
    )?;
    writeln!(
        w,
        "corr_unexplained_toroidal_alignment_sq={}",
        format_optional_float(pearson_correlation(
            &field.unexplained_residual,
            &geometry.toroidal_alignment_sq,
        ))
    )?;
    Ok(())
}

fn write_refinement_summary(out_dir: &Path, rows: &[RefinementRow]) -> io::Result<()> {
    let mut csv = BufWriter::new(File::create(out_dir.join("residual_study_refinement.csv"))?);
    writeln!(
        csv,
        "mesh_label,mesh_size,strategy,edge_dofs,total_post_length_mean,total_post_length_2se,position_even_fourier_mean,position_even_fourier_2se,position_even_fourier_fraction_mean,position_even_fourier_fraction_2se,direction_legendre_mean,direction_legendre_2se,direction_legendre_fraction_mean,direction_legendre_fraction_2se,interaction_even_mean,interaction_even_2se,interaction_even_fraction_mean,interaction_even_fraction_2se,discrete_surrogates_mean,discrete_surrogates_2se,discrete_surrogates_fraction_mean,discrete_surrogates_fraction_2se,unexplained_mean,unexplained_2se,unexplained_fraction_mean,unexplained_fraction_2se"
    )?;
    for row in rows {
        writeln!(
            csv,
            "{},{},{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            row.mesh_label,
            format_optional_mesh_size(row.mesh_size),
            strategy_dir_name(row.strategy),
            row.edge_dofs,
            row.total_post_length.mean,
            two_se(row.total_post_length.standard_error),
            row.position_even_fourier.absolute.mean,
            two_se(row.position_even_fourier.absolute.standard_error),
            row.position_even_fourier.fraction_of_total.mean,
            two_se(row.position_even_fourier.fraction_of_total.standard_error),
            row.direction_legendre.absolute.mean,
            two_se(row.direction_legendre.absolute.standard_error),
            row.direction_legendre.fraction_of_total.mean,
            two_se(row.direction_legendre.fraction_of_total.standard_error),
            row.interaction_even.absolute.mean,
            two_se(row.interaction_even.absolute.standard_error),
            row.interaction_even.fraction_of_total.mean,
            two_se(row.interaction_even.fraction_of_total.standard_error),
            row.discrete_surrogates.absolute.mean,
            two_se(row.discrete_surrogates.absolute.standard_error),
            row.discrete_surrogates.fraction_of_total.mean,
            two_se(row.discrete_surrogates.fraction_of_total.standard_error),
            row.unexplained.absolute.mean,
            two_se(row.unexplained.absolute.standard_error),
            row.unexplained.fraction_of_total.mean,
            two_se(row.unexplained.fraction_of_total.standard_error),
        )?;
    }

    let mut txt = BufWriter::new(File::create(out_dir.join("residual_study_refinement.txt"))?);
    writeln!(txt, "Matérn 1-form torus residual study refinement summary")?;
    writeln!(
        txt,
        "metrics are reported as mean +- 2SE over the RBMC batches"
    )?;
    for row in rows {
        writeln!(
            txt,
            "{} {} ndofs={} H_post_len={} C_position={} C_direction={} C_interaction={} C_discrete={} C_unexplained={}",
            row.mesh_label,
            strategy_dir_name(row.strategy),
            row.edge_dofs,
            format_estimate(&row.total_post_length),
            format_contribution(&row.position_even_fourier),
            format_contribution(&row.direction_legendre),
            format_contribution(&row.interaction_even),
            format_contribution(&row.discrete_surrogates),
            format_contribution(&row.unexplained),
        )?;
    }
    Ok(())
}

fn print_strategy_summary(mesh_run: &MeshRunSpec, report: &Matern1FormTorusResidualStudyReport) {
    let summary = &report.contribution_summary;
    let field = &report.field_decomposition;
    let geometry = &report.geometry;
    println!(
        "[{}:{}]",
        mesh_run.label,
        strategy_dir_name(report.mass_inverse)
    );
    print_estimate("H_post_len", &summary.total_post_length);
    print_contribution("C_position_even_fourier", &summary.position_even_fourier);
    print_contribution("C_direction_legendre", &summary.direction_legendre);
    print_contribution("C_interaction_even", &summary.interaction_even);
    print_contribution("C_discrete_surrogates", &summary.discrete_surrogates);
    print_contribution("C_unexplained", &summary.unexplained);
    println!(
        "RBMC floor hits={} harmonic-free floor hits={}",
        report.rbmc_floor_hits, report.harmonic_free_floor_hits
    );
    print_corr(
        "corr(response, curvature)",
        pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.gaussian_curvature,
        ),
    );
    print_corr(
        "corr(response, toroidal_alignment_sq)",
        pearson_correlation(
            &field.log_harmonic_free_variance_per_length2,
            &geometry.toroidal_alignment_sq,
        ),
    );
    print_corr(
        "corr(unexplained, curvature)",
        pearson_correlation(&field.unexplained_residual, &geometry.gaussian_curvature),
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

fn format_estimate(estimate: &EstimateWithError) -> String {
    format!(
        "{:.6e} +- {:.6e}",
        estimate.mean,
        two_se(estimate.standard_error)
    )
}

fn format_contribution(estimate: &ContributionEstimate) -> String {
    format!(
        "{:.6e} +- {:.6e} ({:.3}% +- {:.3}%)",
        estimate.absolute.mean,
        two_se(estimate.absolute.standard_error),
        100.0 * estimate.fraction_of_total.mean,
        100.0 * two_se(estimate.fraction_of_total.standard_error),
    )
}

fn format_optional_float(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6e}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn format_optional_mesh_size(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.5}"))
        .unwrap_or_else(|| "NA".to_string())
}

fn two_se(standard_error: f64) -> f64 {
    2.0 * standard_error
}

fn strategy_dir_name(strategy: MaternMassInverse) -> &'static str {
    match strategy {
        MaternMassInverse::RowSumLumped => "row_sum_lumped",
        MaternMassInverse::Nc1ProjectedSparseInverse => "nc1_projected",
    }
}
