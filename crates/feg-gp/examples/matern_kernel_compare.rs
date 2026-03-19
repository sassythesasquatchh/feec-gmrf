use faer::Mat;
use feg_gp::{
    condition_full_covariance_with_covariance, matern_covariance_matrix_euclidean,
    EuclideanMaternConfig, SpectralMaternConfig, SpectralMaternGp,
};
use libm::tgamma;
use manifold::gen::cartesian::CartesianMeshInfo;
use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return Ok(());
    }

    let out_dir = "out/matern_kernel_compare";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    // 2D grid with a buffer to reduce boundary effects.
    let dim = 2;
    // let core_points = 16;
    let core_points = 30;
    let buffer = 4;
    let grid_size = core_points + 2 * buffer;
    let cells_per_axis = grid_size - 1;

    let mesh = CartesianMeshInfo::new_unit_scaled(dim, cells_per_axis, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let dx = 1.0 / cells_per_axis as f64;
    let dy = dx;

    // Ensure hyperparams are consistent
    let alpha = 2.; // This is the alpha when using precision KT*M*K
    let nu = alpha - dim as f64 / 2.;
    let tau = 1.;
    let kappa: f64 = 10.;
    let variance = tgamma(nu)
        / (tau * tau * tgamma(alpha) * (4. * PI).powf(dim as f64 / 2.) * kappa.powf(2. * nu));
    // let length_scale = (2. * nu).sqrt() / kappa;
    let effective_range = (8. * nu).sqrt() / kappa;
    println!("Effective range: {}", effective_range);

    let noise_variance = 1e-7;

    let ndofs = coords.nvertices();
    // let spectral_k = 64.min(ndofs).max(1);
    let spectral_k = 128.min(ndofs).max(1);

    let obs_points = [
        (core_points / 4, core_points / 4),
        (core_points / 2, core_points / 2),
        (3 * core_points / 4, core_points / 4),
        (core_points / 4, 3 * core_points / 4),
        (3 * core_points / 4, 3 * core_points / 4),
    ];
    let obs_indices: Vec<usize> = obs_points
        .iter()
        .map(|(x, y)| grid_index(grid_size, x + buffer, y + buffer))
        .collect();
    let obs_values: Vec<f64> = obs_points
        .iter()
        .map(|(x, y)| {
            let idx = grid_index(grid_size, x + buffer, y + buffer);
            let p = coords.coord(idx);
            let xx = p[0];
            let yy = p[1];
            (2.0 * PI * xx).sin() * (2.0 * PI * yy).cos()
        })
        .collect();

    let spectral_gp = SpectralMaternGp::from_hodge_laplace(
        &topology,
        &metric,
        0,
        SpectralMaternConfig {
            kappa,
            alpha,
            tau,
            k: spectral_k,
        },
    )?;
    let spectral_cov = spectral_gp.covariance_matrix();

    let points = coords
        .coord_iter()
        .map(|coord| vec![coord[0], coord[1]])
        .collect::<Vec<_>>();
    let kernel_cov = matern_covariance_matrix_euclidean(
        &points,
        EuclideanMaternConfig {
            kappa,
            nu,
            variance,
        },
    )?;

    let spectral_post = condition_full_covariance_with_covariance(
        &spectral_cov,
        &obs_indices,
        &obs_values,
        noise_variance,
    )?;
    let kernel_post = condition_full_covariance_with_covariance(
        &kernel_cov,
        &obs_indices,
        &obs_values,
        noise_variance,
    )?;

    let var_spectral = diagonal_variance(&spectral_post.covariance)?;
    let var_kernel = diagonal_variance(&kernel_post.covariance)?;

    let core_indices = core_indices_2d(grid_size, buffer, core_points);
    let mean_spectral_core = select_vec_values(&spectral_post.mean, &core_indices);
    let var_spectral_core = select_vec_values(&var_spectral, &core_indices);

    let mean_kernel_core = select_vec_values(&kernel_post.mean, &core_indices);
    let var_kernel_core = select_vec_values(&var_kernel, &core_indices);

    let mean_diff: Vec<f64> = mean_kernel_core
        .iter()
        .zip(mean_spectral_core.iter())
        .map(|(kernel, spectral)| kernel - spectral)
        .collect();
    let var_diff: Vec<f64> = var_kernel_core
        .iter()
        .zip(var_spectral_core.iter())
        .map(|(kernel, spectral)| kernel - spectral)
        .collect();

    let origin_x = buffer as f64 * dx;
    let origin_y = buffer as f64 * dy;
    write_structured_points_2d_fields(
        &format!("{out_dir}/posterior_spectral.vtk"),
        origin_x,
        origin_y,
        dx,
        dy,
        core_points,
        &[
            ("mean_spectral", &mean_spectral_core),
            ("var_spectral", &var_spectral_core),
        ],
    )?;
    write_structured_points_2d_fields(
        &format!("{out_dir}/posterior_kernel.vtk"),
        origin_x,
        origin_y,
        dx,
        dy,
        core_points,
        &[
            ("mean_kernel", &mean_kernel_core),
            ("var_kernel", &var_kernel_core),
        ],
    )?;
    write_structured_points_2d_fields(
        &format!("{out_dir}/posterior_diff.vtk"),
        origin_x,
        origin_y,
        dx,
        dy,
        core_points,
        &[("mean_diff", &mean_diff), ("var_diff", &var_diff)],
    )?;

    println!("Spectral Matern GP vs Euclidean kernel GP (2D)");
    println!("kappa={kappa:.3} nu={nu:.3} variance={variance:.3} noise={noise_variance:.1e}");
    println!(
        "grid: {grid_size}x{grid_size} (core={core_points}x{core_points}, buffer={buffer}, dx={dx}, dy={dy})"
    );
    println!("spectral k={}", spectral_gp.k());
    println!("observations at {obs_points:?}");
    println!("Wrote VTK outputs to {out_dir}");

    Ok(())
}

fn write_structured_points_2d_fields(
    path: &str,
    origin_x: f64,
    origin_y: f64,
    spacing_x: f64,
    spacing_y: f64,
    grid_size: usize,
    fields: &[(&str, &[f64])],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    let expected_len = grid_size * grid_size;
    for (name, values) in fields {
        if values.len() != expected_len {
            return Err(format!(
                "field '{name}' length {} does not match {}",
                values.len(),
                expected_len
            )
            .into());
        }
    }

    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "Matern kernel GP posterior fields (2D)")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET STRUCTURED_POINTS")?;
    writeln!(writer, "DIMENSIONS {} {} 1", grid_size, grid_size)?;
    writeln!(writer, "ORIGIN {:.6} {:.6} 0", origin_x, origin_y)?;
    writeln!(writer, "SPACING {:.6} {:.6} 1", spacing_x, spacing_y)?;
    writeln!(writer, "POINT_DATA {}", expected_len)?;

    for (name, values) in fields {
        writeln!(writer, "SCALARS {} float 1", name)?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for value in *values {
            writeln!(writer, "{:.6}", value)?;
        }
    }

    Ok(())
}

fn grid_index(grid_size: usize, x: usize, y: usize) -> usize {
    y * grid_size + x
}

fn core_indices_2d(grid_size: usize, buffer: usize, core_points: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(core_points * core_points);
    for y in 0..core_points {
        for x in 0..core_points {
            indices.push(grid_index(grid_size, x + buffer, y + buffer));
        }
    }
    indices
}

fn select_vec_values(values: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| values[i]).collect()
}

fn diagonal_variance(cov: &Mat<f64>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if cov.nrows() != cov.ncols() {
        return Err(format!(
            "covariance must be square, got {}x{}",
            cov.nrows(),
            cov.ncols()
        )
        .into());
    }
    let n = cov.nrows();
    let mut variances = Vec::with_capacity(n);
    for i in 0..n {
        variances.push(cov[(i, i)].max(0.0));
    }
    Ok(variances)
}

fn petsc_solver_available() -> bool {
    if let Ok(path) = std::env::var("PETSC_SOLVER_PATH") {
        let candidate = PathBuf::from(path).join("ghiep.out");
        return candidate.exists();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../feec/petsc-solver/ghiep.out")
        .exists()
}
