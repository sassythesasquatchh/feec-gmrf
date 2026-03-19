use faer::{Mat, Side};
use feg_gp::{
    condition_full_covariance_with_covariance, matern_covariance_matrix_euclidean,
    EuclideanMaternConfig,
};
use feg_infer::matern_0form::{
    build_laplace_beltrami_0form_from_galmats, build_matern_precision_0form, feec_csr_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::types::Vector;
use gmrf_core::Gmrf;
use libm::tgamma;
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;
use std::fs;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "out/matern_grid_compare";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    // 2D grid with a buffer to reduce boundary effects.
    let dim = 2;
    let core_points = 16;
    let buffer = 4;
    let grid_size = core_points + 2 * buffer;
    let cells_per_axis = grid_size - 1;

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

    let mesh = CartesianMeshInfo::new_unit_scaled(dim, cells_per_axis, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let dx = 1.0 / cells_per_axis as f64;
    let dy = dx;

    let noise_variance = 1e-7;
    let num_mc_samples = 256;

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

    let laplace = build_laplace_beltrami_0form_from_galmats(
        &formoniq::problems::laplace_beltrami::LaplaceBeltramiGalmats::compute(&topology, &metric),
    );

    let prior_precision = build_matern_precision_0form(
        &laplace,
        MaternConfig {
            kappa,
            tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );

    let q_prior = feec_csr_to_gmrf(&prior_precision);
    let ndofs = q_prior.nrows();
    let observation_matrix = observation_selector(ndofs, &obs_indices);
    let observations = Vector::from_vec(obs_values.clone());
    let (q_post, info) = apply_gaussian_observations(
        &q_prior,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );

    let q_factor = q_post.cholesky_sqrt_lower()?;
    let mut gmrf = Gmrf::from_information_and_precision_with_sqrt(info, q_post, q_factor)?;
    let mean_gmrf = gmrf.mean().clone();
    let mut rng = StdRng::seed_from_u64(7);
    let var_gmrf = gmrf.mc_variances(num_mc_samples, &mut rng)?;

    let points = coords
        .coord_iter()
        .map(|coord| vec![coord[0], coord[1]])
        .collect::<Vec<_>>();
    let gp_cov = matern_covariance_matrix_euclidean(
        &points,
        EuclideanMaternConfig {
            kappa,
            nu,
            variance,
        },
    )?;
    let conditioned = condition_full_covariance_with_covariance(
        &gp_cov,
        &obs_indices,
        &obs_values,
        noise_variance,
    )?;

    let mut gp_rng = StdRng::seed_from_u64(11);
    let var_gp = mc_variance_from_covariance(&conditioned.covariance, num_mc_samples, &mut gp_rng)?;

    let core_indices = core_indices_2d(grid_size, buffer, core_points);
    let mean_gmrf_core = select_values(&mean_gmrf, &core_indices);
    let var_gmrf_core = select_values(&var_gmrf, &core_indices);

    let mean_gp_core = select_vec_values(&conditioned.mean, &core_indices);
    let var_gp_core = select_vec_values(&var_gp, &core_indices);
    let mean_diff: Vec<f64> = mean_gp_core
        .iter()
        .zip(mean_gmrf_core.iter())
        .map(|(gp, gmrf)| gp - gmrf)
        .collect();
    let var_diff: Vec<f64> = var_gp_core
        .iter()
        .zip(var_gmrf_core.iter())
        .map(|(gp, gmrf)| gp - gmrf)
        .collect();

    let origin_x = buffer as f64 * dx;
    let origin_y = buffer as f64 * dy;
    write_structured_points_2d_fields(
        &format!("{out_dir}/posterior_gp.vtk"),
        origin_x,
        origin_y,
        dx,
        dy,
        core_points,
        &[("mean_gp", &mean_gp_core), ("var_gp_mc", &var_gp_core)],
    )?;
    write_structured_points_2d_fields(
        &format!("{out_dir}/posterior_gmrf.vtk"),
        origin_x,
        origin_y,
        dx,
        dy,
        core_points,
        &[
            ("mean_gmrf", &mean_gmrf_core),
            ("var_gmrf_mc", &var_gmrf_core),
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

    println!("Matern GP vs GMRF comparison (2D)");
    println!("kappa={kappa:.3} nu={nu:.3} variance={variance:.3} noise={noise_variance:.1e}");
    println!(
        "grid: {grid_size}x{grid_size} (core={core_points}x{core_points}, buffer={buffer}, dx={dx}, dy={dy})"
    );
    println!("observations at {obs_points:?}");
    println!("MC variances: {num_mc_samples} samples");
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
    writeln!(writer, "Matern grid posterior fields (2D)")?;
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

fn select_values(values: &Vector, indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| values[i]).collect()
}

fn select_vec_values(values: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| values[i]).collect()
}

fn mc_variance_from_covariance(
    cov: &Mat<f64>,
    num_samples: usize,
    rng: &mut impl Rng,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    if num_samples == 0 {
        return Err("num_samples must be positive".into());
    }
    if cov.nrows() != cov.ncols() {
        return Err("covariance must be square".into());
    }
    let n = cov.nrows();
    let chol = cov
        .llt(Side::Lower)
        .map_err(|_| "posterior covariance not SPD")?;
    let l = chol.L();
    let normal = StandardNormal;
    let mut variances = vec![0.0; n];
    let mut z = vec![0.0; n];
    for _ in 0..num_samples {
        for zi in &mut z {
            *zi = normal.sample(rng);
        }
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..=i {
                acc += l[(i, j)] * z[j];
            }
            variances[i] += acc * acc;
        }
    }
    let denom = num_samples as f64;
    for v in &mut variances {
        *v /= denom;
    }
    Ok(variances)
}
