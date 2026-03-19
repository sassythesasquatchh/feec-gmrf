use common::linalg::nalgebra::Vector as FeecVector;
use ddf::cochain::Cochain;
use exterior::field::DiffFormClosure;
use feg_infer::matern_1form::{
    build_hodge_laplacian_1form_from_galmats, build_matern_precision_1form, feec_csr_to_gmrf,
    feec_vec_to_gmrf, MaternConfig, MaternMassInverse,
};
use feg_infer::vtk::write_1cochain_vtk_fields;
use formoniq::{
    assemble::assemble_galvec,
    io::{write_1form_vector_field_vtk, write_cochain_vtk},
    operators::SourceElVec,
    problems::hodge_laplace,
};
use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::Gmrf;
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "out/matern_1form";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let dim = 2;
    let cells_per_axis = 20;
    let mesh = CartesianMeshInfo::new_unit_scaled(dim, cells_per_axis, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let grade = 1;
    let homology_dim = 0;
    let galmats = hodge_laplace::MixedGalmats::compute(&topology, &metric, grade);
    let source_form = DiffFormClosure::one_form(
        |p| {
            FeecVector::from_iterator(
                p.len(),
                p.iter().map(|&x| (2.0 * std::f64::consts::PI * x).sin()),
            )
        },
        dim,
    );
    let source_data = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new(&source_form, &coords, None),
    );

    let (_sigma, u, _harmonics) = hodge_laplace::solve_hodge_laplace_source_with_galmats(
        &topology,
        &galmats,
        source_data,
        grade,
        homology_dim,
    );

    let hodge = build_hodge_laplacian_1form_from_galmats(&galmats);
    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 2.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );

    let ndofs = hodge.mass_u.nrows();
    let u_sol = u.coeffs.clone();
    let u_gmrf = feec_vec_to_gmrf(&u_sol);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);

    let obs_fraction = 0.05;
    let mut num_obs = (ndofs as f64 * obs_fraction).round() as usize;
    num_obs = num_obs.clamp(1, ndofs);
    let mut obs_indices: Vec<usize> = (0..ndofs).collect();
    let mut obs_rng = StdRng::seed_from_u64(11);
    obs_indices.shuffle(&mut obs_rng);
    obs_indices.truncate(num_obs);
    obs_indices.sort_unstable();

    let observation_matrix = observation_selector(ndofs, &obs_indices);
    let observations = &observation_matrix * &u_gmrf;

    let noise_variance = 1e-3;
    let (posterior_precision, information) = apply_gaussian_observations(
        &q_prior_gmrf,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );

    let q_factor = posterior_precision.cholesky_sqrt_lower()?;
    let mut posterior =
        Gmrf::from_information_and_precision_with_sqrt(information, posterior_precision, q_factor)?;

    let num_mc_samples = 256;
    let mut mc_rng = StdRng::seed_from_u64(99);
    let posterior_variances = posterior.mc_variances(num_mc_samples, &mut mc_rng)?;
    let posterior_std = posterior_variances.map(|v| v.sqrt());

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

    let posterior_mean = gmrf_vec_to_feec(posterior.mean());
    let posterior_var = gmrf_vec_to_feec(&posterior_variances);
    let posterior_std = gmrf_vec_to_feec(&posterior_std);

    let posterior_mean_cochain = Cochain::new(grade, posterior_mean);
    let posterior_var_cochain = Cochain::new(grade, posterior_var);
    let posterior_std_cochain = Cochain::new(grade, posterior_std);
    write_1cochain_vtk_fields(
        format!("{out_dir}/posterior_mean.vtk"),
        &coords,
        &topology,
        &[
            ("posterior_mean", &posterior_mean_cochain),
            ("posterior_variance_mc", &posterior_var_cochain),
            ("posterior_std_mc", &posterior_std_cochain),
        ],
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/posterior_mean_vector_field.vtk"),
        &coords,
        &topology,
        &posterior_mean_cochain,
        "posterior_mean_vector_field",
    )?;

    let mut rng = StdRng::seed_from_u64(7);
    for (idx, sample_vec) in (0..3)
        .map(|_| posterior.sample_one_solve(&mut rng))
        .enumerate()
    {
        let sample_vec = sample_vec?;
        let sample_cochain = Cochain::new(grade, gmrf_vec_to_feec(&sample_vec));
        write_1cochain_vtk_fields(
            format!("{out_dir}/posterior_sample_{idx}.vtk"),
            &coords,
            &topology,
            &[
                ("posterior_sample", &sample_cochain),
                ("posterior_variance_mc", &posterior_var_cochain),
                ("posterior_std_mc", &posterior_std_cochain),
            ],
        )?;
        write_1form_vector_field_vtk(
            format!("{out_dir}/posterior_sample_{idx}_vector_field.vtk"),
            &coords,
            &topology,
            &sample_cochain,
            "posterior_sample_vector_field",
        )?;
    }

    println!("1-form dofs: {ndofs}");
    println!(
        "observations: {num_obs} of {ndofs} (~{:.1}%)",
        100.0 * num_obs as f64 / ndofs as f64
    );
    let (mean_mean, mean_std) = summarize(posterior.mean());
    let (var_mean, var_std) = summarize(&posterior_variances);
    println!("posterior mean stats: mean={mean_mean:.6} std={mean_std:.6}");
    println!("mc variances (n={num_mc_samples}) stats: mean={var_mean:.6} std={var_std:.6}");
    println!("Wrote VTK outputs to {out_dir}");

    Ok(())
}

fn summarize(v: &gmrf_core::types::Vector) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

fn gmrf_vec_to_feec(vec: &gmrf_core::types::Vector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}
