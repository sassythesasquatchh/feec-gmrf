use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use feg_infer::diagnostics::{
    compute_matern_0form_diagnostics, infer_torus_radii, pearson_correlation, summarize_vector,
    vertex_rho, Matern0FormDiagnosticsReport, SummaryStats,
};
use feg_infer::matern_0form::{
    build_laplace_beltrami_0form, feec_csr_to_gmrf, MaternConfig, MaternMassInverse,
};
use feg_infer::util::{
    convert_whittle_params_to_matern, generate_torus_surface_mesh, mesh_size_tag,
};
use feg_infer::vtk::write_0cochain_vtk_fields;
use gmrf_core::types::Vector as GmrfVector;
use gmrf_core::Gmrf;
use rand::SeedableRng;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_MESH_PATH: &str = "meshes/torus_shell.msh";
const DEFAULT_MAJOR_RADIUS: f64 = 1.0;
const DEFAULT_MINOR_RADIUS: f64 = 0.3;
const DEFAULT_RHO_BUCKETS: usize = 6;

#[derive(Debug, Clone)]
struct Config {
    kappa: f64,
    tau: f64,
    num_mc_samples: usize,
    rng_seed: u64,
    mesh_path: PathBuf,
    mesh_size: Option<f64>,
    study_mesh_sizes: Vec<f64>,
    rho_buckets: usize,
    major_radius: f64,
    minor_radius: f64,
}

#[derive(Debug, Clone)]
struct TorusVertexGeometry {
    major_radius: f64,
    minor_radius: f64,
    rho: FeecVector,
    gaussian_curvature: FeecVector,
}

#[derive(Debug, Clone)]
struct BucketSummary {
    bucket_idx: usize,
    count: usize,
    rho_min: f64,
    rho_max: f64,
    rho_mean: f64,
    curvature_mean: f64,
    mass_diag_mean: f64,
    forcing_scale_mean: f64,
    variance_mean: f64,
    variance_per_mass_diag_mean: f64,
    variance_per_forcing_scale_mean: f64,
    standardized_variance_mean: f64,
}

#[derive(Debug, Clone)]
struct MeshRun {
    report: Matern0FormDiagnosticsReport,
    geometry: TorusVertexGeometry,
    bucket_summaries: Vec<BucketSummary>,
}

#[derive(Debug, Clone)]
struct RefinementStudyRow {
    mesh_size: f64,
    vertex_dofs: usize,
    corr_mass_rho: Option<f64>,
    corr_variance_rho: Option<f64>,
    corr_variance_per_forcing_rho: Option<f64>,
    corr_standardized_variance_rho: Option<f64>,
    corr_variance_curvature: Option<f64>,
    corr_variance_per_forcing_curvature: Option<f64>,
    corr_standardized_variance_curvature: Option<f64>,
    high_over_low_mass: Option<f64>,
    high_over_low_variance: Option<f64>,
    high_over_low_variance_per_forcing: Option<f64>,
    high_over_low_standardized_variance: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kappa: 20.0,
            tau: 1.0,
            num_mc_samples: 1024,
            rng_seed: 13,
            mesh_path: PathBuf::from(DEFAULT_MESH_PATH),
            mesh_size: None,
            study_mesh_sizes: Vec::new(),
            rho_buckets: DEFAULT_RHO_BUCKETS,
            major_radius: DEFAULT_MAJOR_RADIUS,
            minor_radius: DEFAULT_MINOR_RADIUS,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_args()?;
    let total_start = Instant::now();

    let out_dir = PathBuf::from("out/matern_0form_torus_diagnostics");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let mesh_path = if let Some(mesh_size) = config.mesh_size {
        generate_torus_surface_mesh(
            &out_dir.join(format!(
                "generated_torus_h_{}.msh",
                mesh_size_tag(mesh_size)
            )),
            config.major_radius,
            config.minor_radius,
            mesh_size,
        )?
    } else {
        config.mesh_path.clone()
    };

    let mesh_run = run_single_mesh(
        &mesh_path,
        "primary",
        &out_dir,
        &config,
        Some(String::new()),
    )?;

    let mut study_rows = Vec::new();
    if !config.study_mesh_sizes.is_empty() {
        let study_dir = out_dir.join("refinement_study");
        fs::create_dir_all(&study_dir)?;
        for &mesh_size in &config.study_mesh_sizes {
            let mesh_path = generate_torus_surface_mesh(
                &study_dir.join(format!("torus_h_{}.msh", mesh_size_tag(mesh_size))),
                config.major_radius,
                config.minor_radius,
                mesh_size,
            )?;
            let label = format!("h={mesh_size:.5}");
            let run = run_single_mesh(
                &mesh_path,
                &label,
                &study_dir,
                &config,
                Some(format!("mesh_{}", mesh_size_tag(mesh_size))),
            )?;
            study_rows.push(build_refinement_study_row(mesh_size, &run));
        }
        write_refinement_study(&out_dir, &study_rows)?;
        print_refinement_study(&study_rows);
    }

    let (_nu, _variance, euclidean_effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    println!("vertex dofs: {}", mesh_run.report.prior_precision.nrows());
    println!("kappa={}, tau={}", config.kappa, config.tau);
    println!("Euclidean effective range: {euclidean_effective_range}");
    println!("Loaded mesh from {}", mesh_path.display());
    if !study_rows.is_empty() {
        println!(
            "refinement study mesh sizes: {}",
            config
                .study_mesh_sizes
                .iter()
                .map(|h| format!("{h:.5}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("Wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn run_single_mesh(
    mesh_path: &Path,
    mesh_label: &str,
    out_dir: &Path,
    config: &Config,
    prefix: Option<String>,
) -> Result<MeshRun, Box<dyn std::error::Error>> {
    let t = Instant::now();
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    println!(
        "[{mesh_label}] load mesh + metric: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let laplace = build_laplace_beltrami_0form(&topology, &metric);
    println!(
        "[{mesh_label}] assemble scalar Laplace-Beltrami operators: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let report = compute_matern_0form_diagnostics(
        &coords,
        &laplace,
        MaternConfig {
            kappa: config.kappa,
            tau: config.tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
        config.num_mc_samples,
        config.rng_seed,
    )
    .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
    println!(
        "[{mesh_label}] diagnostics build: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let prior_sample = sample_prior_field(&report, config.rng_seed.wrapping_add(2_000_003))?;
    println!(
        "[{mesh_label}] prior sample draw: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let geometry = build_torus_geometry(&coords)?;
    let bucket_summaries = build_rho_bucket_summaries(&geometry, &report, config.rho_buckets)
        .map_err(|msg| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to build rho bucket summaries: {msg}"),
            )
        })?;

    if let Some(prefix) = prefix {
        write_diagnostics_vtk(
            out_dir,
            &coords,
            &topology,
            &report,
            &prior_sample,
            &geometry,
            &prefix,
        )?;
        write_diagnostics_csv(out_dir, &report, &prior_sample, &geometry, &prefix)?;
        write_bucket_summary_csv(out_dir, &bucket_summaries, &prefix)?;
        write_summary_txt(
            out_dir,
            &report,
            &geometry,
            &bucket_summaries,
            config,
            mesh_path,
            &prefix,
        )?;
    }

    print_summary(mesh_label, &report, &geometry, &bucket_summaries);

    Ok(MeshRun {
        report,
        geometry,
        bucket_summaries,
    })
}

fn build_torus_geometry(
    coords: &manifold::geometry::coord::mesh::MeshCoords,
) -> Result<TorusVertexGeometry, Box<dyn std::error::Error>> {
    let (major_radius, minor_radius) =
        infer_torus_radii(coords).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let rho = vertex_rho(coords);
    let gaussian_curvature = rho.map(|rho_i| gaussian_curvature(major_radius, minor_radius, rho_i));
    Ok(TorusVertexGeometry {
        major_radius,
        minor_radius,
        rho,
        gaussian_curvature,
    })
}

fn gaussian_curvature(major_radius: f64, minor_radius: f64, rho: f64) -> f64 {
    let denom = minor_radius * minor_radius * rho;
    if denom.abs() <= 1e-12 {
        0.0
    } else {
        (rho - major_radius) / denom
    }
}

fn build_rho_bucket_summaries(
    geometry: &TorusVertexGeometry,
    report: &Matern0FormDiagnosticsReport,
    num_buckets: usize,
) -> Result<Vec<BucketSummary>, String> {
    if num_buckets == 0 {
        return Err("num_buckets must be >= 1".to_string());
    }
    let diagnostics = &report.node_diagnostics;
    let standardized = &report.standardized_forcing;
    let n = diagnostics.variances.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        geometry.rho[i]
            .partial_cmp(&geometry.rho[j])
            .expect("finite rho values")
    });

    let mut buckets = Vec::new();
    for bucket_idx in 0..num_buckets {
        let start = bucket_idx * n / num_buckets;
        let end = (bucket_idx + 1) * n / num_buckets;
        if start == end {
            continue;
        }
        let slice = &order[start..end];
        let rho_values = slice.iter().map(|&i| geometry.rho[i]).collect::<Vec<_>>();
        buckets.push(BucketSummary {
            bucket_idx,
            count: slice.len(),
            rho_min: rho_values.iter().copied().fold(f64::INFINITY, f64::min),
            rho_max: rho_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            rho_mean: mean(slice.iter().map(|&i| geometry.rho[i])),
            curvature_mean: mean(slice.iter().map(|&i| geometry.gaussian_curvature[i])),
            mass_diag_mean: mean(slice.iter().map(|&i| diagnostics.mass_diag[i])),
            forcing_scale_mean: mean(slice.iter().map(|&i| diagnostics.forcing_scale_diag[i])),
            variance_mean: mean(slice.iter().map(|&i| diagnostics.variances[i])),
            variance_per_mass_diag_mean: mean(
                slice.iter().map(|&i| diagnostics.variance_per_mass_diag[i]),
            ),
            variance_per_forcing_scale_mean: mean(
                slice
                    .iter()
                    .map(|&i| diagnostics.variance_per_forcing_scale_diag[i]),
            ),
            standardized_variance_mean: mean(slice.iter().map(|&i| standardized.variances[i])),
        });
    }

    Ok(buckets)
}

fn mean(iter: impl Iterator<Item = f64>) -> f64 {
    let values = iter.collect::<Vec<_>>();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn high_over_low_bucket_mean_ratio(
    buckets: &[BucketSummary],
    selector: impl Fn(&BucketSummary) -> f64,
) -> Option<f64> {
    let low = buckets.first()?;
    let high = buckets.last()?;
    let low_value = selector(low);
    if low_value.abs() <= 1e-12 {
        None
    } else {
        Some(selector(high) / low_value)
    }
}

fn sample_prior_field(
    report: &Matern0FormDiagnosticsReport,
    rng_seed: u64,
) -> Result<FeecVector, Box<dyn std::error::Error>> {
    let q_prior = feec_csr_to_gmrf(&report.prior_precision);
    let q_factor = q_prior.cholesky_sqrt_lower()?;
    let prior =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(report.prior_precision.nrows()), q_prior)?
            .with_precision_sqrt(q_factor);

    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);
    let sample = prior.sample_one_solve(&mut rng)?;
    Ok(FeecVector::from_vec(sample.iter().copied().collect()))
}

fn write_diagnostics_vtk(
    out_dir: &Path,
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    report: &Matern0FormDiagnosticsReport,
    prior_sample: &FeecVector,
    geometry: &TorusVertexGeometry,
    prefix: &str,
) -> io::Result<()> {
    let diagnostics = &report.node_diagnostics;
    let standardized = &report.standardized_forcing;

    let prior_sample = Cochain::new(0, prior_sample.clone());
    let variance = Cochain::new(0, diagnostics.variances.clone());
    let mass_diag = Cochain::new(0, diagnostics.mass_diag.clone());
    let forcing_scale_diag = Cochain::new(0, diagnostics.forcing_scale_diag.clone());
    let var_per_mass_diag = Cochain::new(0, diagnostics.variance_per_mass_diag.clone());
    let var_per_forcing_scale_diag =
        Cochain::new(0, diagnostics.variance_per_forcing_scale_diag.clone());
    let standardized_var = Cochain::new(0, standardized.variances.clone());
    let standardized_mean_abs = Cochain::new(0, standardized.mean.map(|v| v.abs()));
    let rho = Cochain::new(0, geometry.rho.clone());
    let curvature = Cochain::new(0, geometry.gaussian_curvature.clone());

    write_0cochain_vtk_fields(
        out_dir.join(output_file_name(prefix, "diagnostics", "vtk")),
        coords,
        topology,
        &[
            ("prior_sample", &prior_sample),
            ("prior_variance_mc", &variance),
            ("mass_diag", &mass_diag),
            ("forcing_scale_diag", &forcing_scale_diag),
            ("variance_per_mass_diag", &var_per_mass_diag),
            (
                "variance_per_forcing_scale_diag",
                &var_per_forcing_scale_diag,
            ),
            ("standardized_forcing_var_mc", &standardized_var),
            ("abs_standardized_forcing_mean_mc", &standardized_mean_abs),
            ("vertex_rho", &rho),
            ("gaussian_curvature", &curvature),
        ],
    )
}

fn write_diagnostics_csv(
    out_dir: &Path,
    report: &Matern0FormDiagnosticsReport,
    prior_sample: &FeecVector,
    geometry: &TorusVertexGeometry,
    prefix: &str,
) -> io::Result<()> {
    let diagnostics = &report.node_diagnostics;
    let standardized = &report.standardized_forcing;

    let file = File::create(out_dir.join(output_file_name(prefix, "node_diagnostics", "csv")))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "vertex_idx,vertex_rho,gaussian_curvature,prior_sample,mass_diag,forcing_scale_diag,variance_mc,variance_per_mass_diag,variance_per_forcing_scale_diag,standardized_variance_mc,abs_standardized_mean_mc"
    )?;

    for i in 0..diagnostics.variances.len() {
        writeln!(
            w,
            "{i},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            geometry.rho[i],
            geometry.gaussian_curvature[i],
            prior_sample[i],
            diagnostics.mass_diag[i],
            diagnostics.forcing_scale_diag[i],
            diagnostics.variances[i],
            diagnostics.variance_per_mass_diag[i],
            diagnostics.variance_per_forcing_scale_diag[i],
            standardized.variances[i],
            standardized.mean[i].abs(),
        )?;
    }

    Ok(())
}

fn write_bucket_summary_csv(
    out_dir: &Path,
    bucket_summaries: &[BucketSummary],
    prefix: &str,
) -> io::Result<()> {
    let file = File::create(out_dir.join(output_file_name(prefix, "rho_bucket_summary", "csv")))?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "bucket_idx,count,rho_min,rho_max,rho_mean,curvature_mean,mass_diag_mean,forcing_scale_mean,variance_mean,variance_per_mass_diag_mean,variance_per_forcing_scale_mean,standardized_variance_mean"
    )?;
    for bucket in bucket_summaries {
        writeln!(
            w,
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}",
            bucket.bucket_idx,
            bucket.count,
            bucket.rho_min,
            bucket.rho_max,
            bucket.rho_mean,
            bucket.curvature_mean,
            bucket.mass_diag_mean,
            bucket.forcing_scale_mean,
            bucket.variance_mean,
            bucket.variance_per_mass_diag_mean,
            bucket.variance_per_forcing_scale_mean,
            bucket.standardized_variance_mean,
        )?;
    }
    Ok(())
}

fn write_summary_txt(
    out_dir: &Path,
    report: &Matern0FormDiagnosticsReport,
    geometry: &TorusVertexGeometry,
    bucket_summaries: &[BucketSummary],
    config: &Config,
    mesh_path: &Path,
    prefix: &str,
) -> io::Result<()> {
    let diagnostics = &report.node_diagnostics;
    let standardized = &report.standardized_forcing;
    let (_nu, _variance, euclidean_effective_range) =
        convert_whittle_params_to_matern(2.0, config.tau, config.kappa, 2);

    let mut w = BufWriter::new(File::create(
        out_dir.join(output_file_name(prefix, "summary", "txt")),
    )?);
    writeln!(w, "Matérn 0-form torus diagnostics")?;
    writeln!(w, "mesh={}", mesh_path.display())?;
    writeln!(w, "mass_inverse={}", strategy_name(report.mass_inverse))?;
    writeln!(w, "kappa={}", config.kappa)?;
    writeln!(w, "tau={}", config.tau)?;
    writeln!(w, "samples={}", config.num_mc_samples)?;
    writeln!(w, "seed={}", config.rng_seed)?;
    writeln!(w, "major_radius={}", geometry.major_radius)?;
    writeln!(w, "minor_radius={}", geometry.minor_radius)?;
    writeln!(w, "rho_bucket_count={}", bucket_summaries.len())?;
    writeln!(w, "euclidean_effective_range={euclidean_effective_range}")?;
    writeln!(w)?;

    writeln!(w, "geometry-driven local scale:")?;
    write_stats(
        &mut w,
        "mass diag",
        summarize_vector(&diagnostics.mass_diag),
    )?;
    write_stats(
        &mut w,
        "forcing scale diag",
        summarize_vector(&diagnostics.forcing_scale_diag),
    )?;
    write_corr(
        &mut w,
        "corr(mass_diag, vertex_rho)",
        pearson_correlation(&diagnostics.mass_diag, &geometry.rho),
    )?;
    write_corr(
        &mut w,
        "corr(forcing_scale_diag, vertex_rho)",
        pearson_correlation(&diagnostics.forcing_scale_diag, &geometry.rho),
    )?;
    write_corr(
        &mut w,
        "corr(mass_diag, gaussian_curvature)",
        pearson_correlation(&diagnostics.mass_diag, &geometry.gaussian_curvature),
    )?;
    writeln!(w)?;

    writeln!(w, "field variance diagnostics:")?;
    write_stats(
        &mut w,
        "prior variance (mc)",
        summarize_vector(&diagnostics.variances),
    )?;
    write_stats(
        &mut w,
        "variance / mass_diag",
        summarize_vector(&diagnostics.variance_per_mass_diag),
    )?;
    write_stats(
        &mut w,
        "variance / forcing_scale_diag",
        summarize_vector(&diagnostics.variance_per_forcing_scale_diag),
    )?;
    write_stats(
        &mut w,
        "standardized forcing variance",
        summarize_vector(&standardized.variances),
    )?;
    write_stats(
        &mut w,
        "abs standardized forcing mean",
        summarize_vector(&standardized.mean.map(|v| v.abs())),
    )?;
    write_corr(
        &mut w,
        "corr(variance, vertex_rho)",
        pearson_correlation(&diagnostics.variances, &geometry.rho),
    )?;
    write_corr(
        &mut w,
        "corr(variance / forcing_scale_diag, vertex_rho)",
        pearson_correlation(&diagnostics.variance_per_forcing_scale_diag, &geometry.rho),
    )?;
    write_corr(
        &mut w,
        "corr(standardized variance, vertex_rho)",
        pearson_correlation(&standardized.variances, &geometry.rho),
    )?;
    write_corr(
        &mut w,
        "corr(variance, gaussian_curvature)",
        pearson_correlation(&diagnostics.variances, &geometry.gaussian_curvature),
    )?;
    write_corr(
        &mut w,
        "corr(variance / forcing_scale_diag, gaussian_curvature)",
        pearson_correlation(
            &diagnostics.variance_per_forcing_scale_diag,
            &geometry.gaussian_curvature,
        ),
    )?;
    write_corr(
        &mut w,
        "corr(standardized variance, gaussian_curvature)",
        pearson_correlation(&standardized.variances, &geometry.gaussian_curvature),
    )?;
    writeln!(w)?;

    writeln!(w, "low-vs-high rho bucket mean ratios:")?;
    write_ratio_line(
        &mut w,
        "mass_diag",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| bucket.mass_diag_mean),
    )?;
    write_ratio_line(
        &mut w,
        "forcing_scale_diag",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| bucket.forcing_scale_mean),
    )?;
    write_ratio_line(
        &mut w,
        "raw_variance",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| bucket.variance_mean),
    )?;
    write_ratio_line(
        &mut w,
        "variance_per_mass_diag",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| {
            bucket.variance_per_mass_diag_mean
        }),
    )?;
    write_ratio_line(
        &mut w,
        "variance_per_forcing_scale_diag",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| {
            bucket.variance_per_forcing_scale_mean
        }),
    )?;
    write_ratio_line(
        &mut w,
        "standardized_variance",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| {
            bucket.standardized_variance_mean
        }),
    )?;
    writeln!(w)?;

    writeln!(w, "rho bucket means:")?;
    for bucket in bucket_summaries {
        writeln!(
            w,
            "  bucket={} count={} rho=[{:.6e}, {:.6e}] rho_mean={:.6e} curvature_mean={:.6e} mass_mean={:.6e} forcing_mean={:.6e} variance_mean={:.6e} variance_per_forcing_mean={:.6e} standardized_variance_mean={:.6e}",
            bucket.bucket_idx,
            bucket.count,
            bucket.rho_min,
            bucket.rho_max,
            bucket.rho_mean,
            bucket.curvature_mean,
            bucket.mass_diag_mean,
            bucket.forcing_scale_mean,
            bucket.variance_mean,
            bucket.variance_per_forcing_scale_mean,
            bucket.standardized_variance_mean,
        )?;
    }

    Ok(())
}

fn print_summary(
    mesh_label: &str,
    report: &Matern0FormDiagnosticsReport,
    geometry: &TorusVertexGeometry,
    bucket_summaries: &[BucketSummary],
) {
    let diagnostics = &report.node_diagnostics;
    let standardized = &report.standardized_forcing;
    println!("[{mesh_label} | {}]", strategy_name(report.mass_inverse));
    print_stats("mass diag", summarize_vector(&diagnostics.mass_diag));
    print_stats(
        "forcing scale diag",
        summarize_vector(&diagnostics.forcing_scale_diag),
    );
    print_stats(
        "prior variance (mc)",
        summarize_vector(&diagnostics.variances),
    );
    print_stats(
        "variance / forcing_scale_diag",
        summarize_vector(&diagnostics.variance_per_forcing_scale_diag),
    );
    print_stats(
        "standardized forcing variance",
        summarize_vector(&standardized.variances),
    );
    print_corr(
        "corr(mass_diag, vertex_rho)",
        pearson_correlation(&diagnostics.mass_diag, &geometry.rho),
    );
    print_corr(
        "corr(variance, vertex_rho)",
        pearson_correlation(&diagnostics.variances, &geometry.rho),
    );
    print_corr(
        "corr(variance / forcing_scale_diag, vertex_rho)",
        pearson_correlation(&diagnostics.variance_per_forcing_scale_diag, &geometry.rho),
    );
    print_corr(
        "corr(standardized variance, vertex_rho)",
        pearson_correlation(&standardized.variances, &geometry.rho),
    );
    print_ratio_console(
        "high/low rho mean ratio (mass_diag)",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| bucket.mass_diag_mean),
    );
    print_ratio_console(
        "high/low rho mean ratio (raw variance)",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| bucket.variance_mean),
    );
    print_ratio_console(
        "high/low rho mean ratio (variance / forcing_scale_diag)",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| {
            bucket.variance_per_forcing_scale_mean
        }),
    );
    print_ratio_console(
        "high/low rho mean ratio (standardized variance)",
        high_over_low_bucket_mean_ratio(bucket_summaries, |bucket| {
            bucket.standardized_variance_mean
        }),
    );
}

fn build_refinement_study_row(mesh_size: f64, run: &MeshRun) -> RefinementStudyRow {
    let diagnostics = &run.report.node_diagnostics;
    let standardized = &run.report.standardized_forcing;
    RefinementStudyRow {
        mesh_size,
        vertex_dofs: run.report.prior_precision.nrows(),
        corr_mass_rho: pearson_correlation(&diagnostics.mass_diag, &run.geometry.rho),
        corr_variance_rho: pearson_correlation(&diagnostics.variances, &run.geometry.rho),
        corr_variance_per_forcing_rho: pearson_correlation(
            &diagnostics.variance_per_forcing_scale_diag,
            &run.geometry.rho,
        ),
        corr_standardized_variance_rho: pearson_correlation(
            &standardized.variances,
            &run.geometry.rho,
        ),
        corr_variance_curvature: pearson_correlation(
            &diagnostics.variances,
            &run.geometry.gaussian_curvature,
        ),
        corr_variance_per_forcing_curvature: pearson_correlation(
            &diagnostics.variance_per_forcing_scale_diag,
            &run.geometry.gaussian_curvature,
        ),
        corr_standardized_variance_curvature: pearson_correlation(
            &standardized.variances,
            &run.geometry.gaussian_curvature,
        ),
        high_over_low_mass: high_over_low_bucket_mean_ratio(&run.bucket_summaries, |bucket| {
            bucket.mass_diag_mean
        }),
        high_over_low_variance: high_over_low_bucket_mean_ratio(&run.bucket_summaries, |bucket| {
            bucket.variance_mean
        }),
        high_over_low_variance_per_forcing: high_over_low_bucket_mean_ratio(
            &run.bucket_summaries,
            |bucket| bucket.variance_per_forcing_scale_mean,
        ),
        high_over_low_standardized_variance: high_over_low_bucket_mean_ratio(
            &run.bucket_summaries,
            |bucket| bucket.standardized_variance_mean,
        ),
    }
}

fn write_refinement_study(
    out_dir: &Path,
    rows: &[RefinementStudyRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut csv = BufWriter::new(File::create(out_dir.join("refinement_study.csv"))?);
    writeln!(
        csv,
        "mesh_size,vertex_dofs,corr_mass_rho,corr_variance_rho,corr_variance_per_forcing_rho,corr_standardized_variance_rho,corr_variance_curvature,corr_variance_per_forcing_curvature,corr_standardized_variance_curvature,high_over_low_mass,high_over_low_variance,high_over_low_variance_per_forcing,high_over_low_standardized_variance"
    )?;
    for row in rows {
        writeln!(
            csv,
            "{:.12e},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.mesh_size,
            row.vertex_dofs,
            format_optional(row.corr_mass_rho),
            format_optional(row.corr_variance_rho),
            format_optional(row.corr_variance_per_forcing_rho),
            format_optional(row.corr_standardized_variance_rho),
            format_optional(row.corr_variance_curvature),
            format_optional(row.corr_variance_per_forcing_curvature),
            format_optional(row.corr_standardized_variance_curvature),
            format_optional(row.high_over_low_mass),
            format_optional(row.high_over_low_variance),
            format_optional(row.high_over_low_variance_per_forcing),
            format_optional(row.high_over_low_standardized_variance),
        )?;
    }

    let mut txt = BufWriter::new(File::create(out_dir.join("refinement_study.txt"))?);
    writeln!(txt, "Matérn 0-form torus refinement study")?;
    writeln!(txt)?;
    for row in rows {
        writeln!(
            txt,
            "h={:.5} ndofs={} corr(mass,rho)={} corr(var,rho)={} corr(var/forcing,rho)={} corr(standardized_var,rho)={} high/low mass={} high/low raw_var={} high/low var/forcing={} high/low standardized_var={}",
            row.mesh_size,
            row.vertex_dofs,
            format_optional(row.corr_mass_rho),
            format_optional(row.corr_variance_rho),
            format_optional(row.corr_variance_per_forcing_rho),
            format_optional(row.corr_standardized_variance_rho),
            format_optional(row.high_over_low_mass),
            format_optional(row.high_over_low_variance),
            format_optional(row.high_over_low_variance_per_forcing),
            format_optional(row.high_over_low_standardized_variance),
        )?;
    }

    Ok(())
}

fn print_refinement_study(rows: &[RefinementStudyRow]) {
    println!("[refinement study]");
    for row in rows {
        println!(
            "h={:.5} ndofs={} corr(var,rho)={} corr(var/forcing,rho)={} corr(std_var,rho)={} high/low raw_var={} high/low var/forcing={} high/low std_var={}",
            row.mesh_size,
            row.vertex_dofs,
            format_optional(row.corr_variance_rho),
            format_optional(row.corr_variance_per_forcing_rho),
            format_optional(row.corr_standardized_variance_rho),
            format_optional(row.high_over_low_variance),
            format_optional(row.high_over_low_variance_per_forcing),
            format_optional(row.high_over_low_standardized_variance),
        );
    }
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
                config.num_mc_samples = parse_next::<usize>(&args, i, "--samples")?;
            }
            "--seed" => {
                i += 1;
                config.rng_seed = parse_next::<u64>(&args, i, "--seed")?;
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
            "--rho-buckets" => {
                i += 1;
                config.rho_buckets = parse_next::<usize>(&args, i, "--rho-buckets")?;
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
                    format!("unknown argument: {other}"),
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
    if config.num_mc_samples == 0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "samples must be at least 1").into(),
        );
    }
    if config.rho_buckets == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "rho-buckets must be at least 1",
        )
        .into());
    }
    if config.major_radius <= 0.0 || config.minor_radius <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "major-radius and minor-radius must be positive",
        )
        .into());
    }
    if config.mesh_size.is_some() && config.mesh_size.unwrap() <= 0.0 {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "mesh-size must be positive").into(),
        );
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
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let raw = args.get(idx).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("missing value for {flag}"),
        )
    })?;
    raw.parse::<T>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {flag}: {e}"),
        )
        .into()
    })
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p feg-infer --example matern_0form_torus_diagnostics -- [options]"
    );
    println!("Options:");
    println!("  --kappa <f64>               Matérn kappa parameter (default: 20.0)");
    println!("  --tau <f64>                 Matérn tau parameter (default: 1.0)");
    println!("  --samples <usize>           Monte Carlo sample count (default: 1024)");
    println!("  --seed <u64>                RNG seed for MC variances and the sample field (default: 13)");
    println!(
        "  --mesh <path>               Input torus mesh path (default: meshes/torus_shell.msh)"
    );
    println!(
        "  --mesh-size <f64>           Generate a torus mesh with this target characteristic length"
    );
    println!(
        "  --study-mesh-sizes <list>   Comma-separated mesh sizes for an optional refinement study"
    );
    println!("  --rho-buckets <usize>       Number of equal-count rho buckets (default: 6)");
    println!("  --major-radius <f64>        Major radius for generated meshes (default: 1.0)");
    println!("  --minor-radius <f64>        Minor radius for generated meshes (default: 0.3)");
}

fn strategy_name(strategy: MaternMassInverse) -> &'static str {
    match strategy {
        MaternMassInverse::RowSumLumped => "row_sum_lumped",
    }
}

fn print_stats(label: &str, stats: Option<SummaryStats>) {
    if let Some(stats) = stats {
        println!(
            "{label}: min={:.6e} max={:.6e} ratio={:.3} mean={:.6e} std={:.6e}",
            stats.min,
            stats.max,
            stats.ratio(),
            stats.mean,
            stats.std
        );
    }
}

fn print_corr(label: &str, corr: Option<f64>) {
    if let Some(corr) = corr {
        println!("{label}={corr:.6}");
    }
}

fn print_ratio_console(label: &str, ratio: Option<f64>) {
    if let Some(ratio) = ratio {
        println!("{label}={ratio:.6}");
    }
}

fn write_stats(
    writer: &mut impl Write,
    label: &str,
    stats: Option<SummaryStats>,
) -> io::Result<()> {
    match stats {
        Some(stats) => writeln!(
            writer,
            "{label}: min={:.6e} max={:.6e} ratio={:.3} mean={:.6e} std={:.6e}",
            stats.min,
            stats.max,
            stats.ratio(),
            stats.mean,
            stats.std
        ),
        None => writeln!(writer, "{label}: n/a"),
    }
}

fn write_corr(writer: &mut impl Write, label: &str, corr: Option<f64>) -> io::Result<()> {
    match corr {
        Some(corr) => writeln!(writer, "{label}={corr:.6}"),
        None => writeln!(writer, "{label}=n/a"),
    }
}

fn write_ratio_line(writer: &mut impl Write, label: &str, ratio: Option<f64>) -> io::Result<()> {
    match ratio {
        Some(ratio) => writeln!(writer, "  {label}={ratio:.6e}"),
        None => writeln!(writer, "  {label}=n/a"),
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.12e}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn output_file_name(prefix: &str, stem: &str, ext: &str) -> String {
    if prefix.is_empty() {
        format!("{stem}.{ext}")
    } else {
        format!("{prefix}_{stem}.{ext}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_curvature_has_expected_inner_and_outer_signs() {
        let major = 1.0;
        let minor = 0.3;
        let outer = gaussian_curvature(major, minor, major + minor);
        let inner = gaussian_curvature(major, minor, major - minor);
        let mid = gaussian_curvature(major, minor, major);

        assert!(outer > 0.0);
        assert!(inner < 0.0);
        assert!(mid.abs() <= 1e-12);
    }

    #[test]
    fn high_over_low_bucket_mean_ratio_uses_end_buckets() {
        let buckets = vec![
            BucketSummary {
                bucket_idx: 0,
                count: 2,
                rho_min: 0.0,
                rho_max: 0.0,
                rho_mean: 0.0,
                curvature_mean: 0.0,
                mass_diag_mean: 2.0,
                forcing_scale_mean: 0.0,
                variance_mean: 0.0,
                variance_per_mass_diag_mean: 0.0,
                variance_per_forcing_scale_mean: 0.0,
                standardized_variance_mean: 0.0,
            },
            BucketSummary {
                bucket_idx: 1,
                count: 2,
                rho_min: 0.0,
                rho_max: 0.0,
                rho_mean: 0.0,
                curvature_mean: 0.0,
                mass_diag_mean: 3.0,
                forcing_scale_mean: 0.0,
                variance_mean: 0.0,
                variance_per_mass_diag_mean: 0.0,
                variance_per_forcing_scale_mean: 0.0,
                standardized_variance_mean: 0.0,
            },
            BucketSummary {
                bucket_idx: 2,
                count: 2,
                rho_min: 0.0,
                rho_max: 0.0,
                rho_mean: 0.0,
                curvature_mean: 0.0,
                mass_diag_mean: 5.0,
                forcing_scale_mean: 0.0,
                variance_mean: 0.0,
                variance_per_mass_diag_mean: 0.0,
                variance_per_forcing_scale_mean: 0.0,
                standardized_variance_mean: 0.0,
            },
        ];

        let ratio = high_over_low_bucket_mean_ratio(&buckets, |bucket| bucket.mass_diag_mean)
            .expect("ratio should exist");
        assert!((ratio - 2.5).abs() <= 1e-12);
    }

    #[test]
    fn parse_mesh_sizes_accepts_comma_separated_values() {
        let values = parse_mesh_sizes("0.2, 0.1,0.05").expect("mesh sizes should parse");
        assert_eq!(values, vec![0.2, 0.1, 0.05]);
    }
}
