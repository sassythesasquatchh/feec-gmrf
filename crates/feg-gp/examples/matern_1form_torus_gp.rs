use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use exterior::field::DiffFormClosure;
use feg_gp::{condition_full_covariance, SpectralMaternConfig, SpectralMaternGp};
use formoniq::assemble::assemble_galvec;
use formoniq::io::{write_1form_vector_field_vtk, write_cochain_vtk};
use formoniq::operators::SourceElVec;
use formoniq::problems::hodge_laplace;
use manifold::io::gmsh::gmsh2coord_complex;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return Ok(());
    }

    let total_start = Instant::now();
    let out_dir = "out/matern_1form_torus_gp";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let mesh_path = "meshes/torus_shell.msh";
    let t = Instant::now();
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    println!("mesh load + metric: {:.3}s", t.elapsed().as_secs_f64());

    let grade = 1;
    let homology_dim = 2;

    let t = Instant::now();
    let source_form = DiffFormClosure::one_form(
        |p| {
            let x = p[0];
            let y = p[1];
            let rho = (x * x + y * y).sqrt().max(1e-12);
            FeecVector::from_column_slice(&[-y / rho, x / rho, 0.0])
        },
        topology.dim(),
    );
    let galmats = hodge_laplace::MixedGalmats::compute(&topology, &metric, grade);
    let source_data = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new(&source_form, &coords, None),
    );
    println!(
        "assembly (galmats + source): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let gp = SpectralMaternGp::from_hodge_laplace(
        &topology,
        &metric,
        grade,
        SpectralMaternConfig {
            kappa: 2.0,
            alpha: 2.0,
            tau: 1.0,
            k: 64,
        },
    )?;
    println!("spectral GP build: {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let (_sigma, u, _harmonics) = hodge_laplace::solve_hodge_laplace_source_with_galmats(
        &topology,
        &galmats,
        source_data,
        grade,
        homology_dim,
    );
    println!(
        "FEEC solve (hodge laplace): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let cov = gp.covariance_matrix();
    println!(
        "explicit covariance build: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let ndofs = cov.nrows();
    let t = Instant::now();
    let prior_variance: Vec<f64> = (0..ndofs).map(|i| cov[(i, i)].max(0.0)).collect();
    let prior_std: Vec<f64> = prior_variance.iter().map(|v| v.sqrt()).collect();

    let mut prior_rng = StdRng::seed_from_u64(5);
    let normal = StandardNormal;
    let mut z = vec![0.0; gp.k()];
    for zi in &mut z {
        *zi = normal.sample(&mut prior_rng);
    }
    let prior_sample = gp.sample_from_standard_normal(&z)?;
    println!("prior sample + std: {:.3}s", t.elapsed().as_secs_f64());

    let prior_sample_vec = FeecVector::from_vec(prior_sample);
    let prior_sample_cochain = Cochain::new(grade, prior_sample_vec);
    write_cochain_vtk(
        format!("{out_dir}/prior_sample.vtk"),
        &coords,
        &topology,
        &prior_sample_cochain,
        "prior_sample",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/prior_sample_vector_field.vtk"),
        &coords,
        &topology,
        &prior_sample_cochain,
        "prior_sample_vector_field",
    )?;

    let prior_variance_vec = FeecVector::from_vec(prior_variance);
    let prior_variance_cochain = Cochain::new(grade, prior_variance_vec);
    write_cochain_vtk(
        format!("{out_dir}/prior_variance.vtk"),
        &coords,
        &topology,
        &prior_variance_cochain,
        "prior_variance",
    )?;

    let prior_std_vec = FeecVector::from_vec(prior_std);
    let prior_std_cochain = Cochain::new(grade, prior_std_vec);
    write_cochain_vtk(
        format!("{out_dir}/prior_std.vtk"),
        &coords,
        &topology,
        &prior_std_cochain,
        "prior_std",
    )?;

    let u_vals: Vec<f64> = u.coeffs.iter().copied().collect();

    let obs_fraction = 0.10;
    let mut num_obs = (ndofs as f64 * obs_fraction).round() as usize;
    num_obs = num_obs.clamp(1, ndofs);
    let mut obs_indices: Vec<usize> = (0..ndofs).collect();
    let mut obs_rng = StdRng::seed_from_u64(11);
    obs_indices.shuffle(&mut obs_rng);
    obs_indices.truncate(num_obs);
    obs_indices.sort_unstable();

    let t = Instant::now();
    let obs_values: Vec<f64> = obs_indices.iter().map(|&idx| u_vals[idx]).collect();
    let noise_variance = 1e-9;
    let conditioned = condition_full_covariance(&cov, &obs_indices, &obs_values, noise_variance)?;
    let (max_abs_err, mean_abs_err) = obs_indices.iter().zip(obs_values.iter()).fold(
        (0.0_f64, 0.0_f64),
        |(max_err, sum_err), (&idx, &obs)| {
            let err = (conditioned.mean[idx] - obs).abs();
            (max_err.max(err), sum_err + err)
        },
    );
    let mean_abs_err = mean_abs_err / num_obs as f64;
    let obs_tol = if noise_variance == 0.0 {
        1e-10
    } else {
        noise_variance.sqrt() * 0.5
    };
    if max_abs_err > obs_tol {
        return Err(format!(
            "posterior mean mismatch at observations: max abs error {max_abs_err:.3e} \
             (mean {mean_abs_err:.3e}) exceeds tolerance {obs_tol:.3e}"
        )
        .into());
    }
    let posterior_std: Vec<f64> = conditioned.variance.iter().map(|v| v.sqrt()).collect();
    println!("GP conditioning: {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    write_cochain_vtk(
        format!("{out_dir}/solution.vtk"),
        &coords,
        &topology,
        &u,
        "solution",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/solution_vector_field.vtk"),
        &coords,
        &topology,
        &u,
        "solution_vector_field",
    )?;

    let posterior_mean_vec = FeecVector::from_vec(conditioned.mean);
    let posterior_mean_cochain = Cochain::new(grade, posterior_mean_vec);
    write_cochain_vtk(
        format!("{out_dir}/posterior_mean.vtk"),
        &coords,
        &topology,
        &posterior_mean_cochain,
        "posterior_mean",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/posterior_mean_vector_field.vtk"),
        &coords,
        &topology,
        &posterior_mean_cochain,
        "posterior_mean_vector_field",
    )?;

    let posterior_std_vec = FeecVector::from_vec(posterior_std);
    let posterior_std_cochain = Cochain::new(grade, posterior_std_vec);
    write_cochain_vtk(
        format!("{out_dir}/posterior_std.vtk"),
        &coords,
        &topology,
        &posterior_std_cochain,
        "posterior_std",
    )?;

    let mut obs_mask = vec![0.0; ndofs];
    for &idx in &obs_indices {
        obs_mask[idx] = 1.0;
    }
    let obs_mask_cochain = Cochain::new(grade, FeecVector::from_vec(obs_mask));
    write_cochain_vtk(
        format!("{out_dir}/observation_mask.vtk"),
        &coords,
        &topology,
        &obs_mask_cochain,
        "observation_mask",
    )?;

    println!("write VTK outputs: {:.3}s", t.elapsed().as_secs_f64());
    println!("1-form dofs: {ndofs}");
    println!(
        "observations: {num_obs} of {ndofs} (~{:.1}%)",
        100.0 * num_obs as f64 / ndofs as f64
    );
    println!("Loaded mesh from {mesh_path}");
    println!("Wrote VTK outputs to {out_dir}");
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

    Ok(())
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
