use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector};
use ddf::cochain::Cochain;
use exterior::field::DiffFormClosure;
use faer::linalg::solvers::Solve;
use faer::Side;
use feg_infer::matern_1form::{
    build_hodge_laplacian_1form_from_galmats, build_matern_precision_1form,
    build_reconstructed_barycenter_field_operator, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse, ReconstructedBarycenterField,
};
use feg_infer::util::convert_whittle_params_to_matern;
use feg_infer::vtk::write_1cochain_vtk_fields;
use formoniq::{
    assemble::assemble_galvec,
    io::{
        sample_1form_cell_vectors, write_1form_vector_field_vtk, write_cochain_vtk,
        write_top_cell_vtk_fields,
    },
    operators::SourceElVec,
    problems::hodge_laplace,
};
use gmrf_core::observation::{apply_gaussian_observations, observation_selector};
use gmrf_core::types::{DenseMatrix as GmrfDenseMatrix, Vector as GmrfVector};
use gmrf_core::Gmrf;
use manifold::{geometry::coord::mesh::MeshCoords, topology::complex::Complex};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example overview:
    // 1) Solve a FEEC Hodge-Laplace problem on a torus mesh (reference solution).
    // 2) Build a Matérn prior precision on 1-forms.
    // 3) Construct two posteriors:
    //    - PDE-observation posterior (uses the Laplacian as observation operator).
    //    - Selector-observation posterior (uses ~10% of FEEC solution entries).
    // 4) Estimate all variances via Monte Carlo sampling.
    // 5) Write cochain fields and vector fields for solution, prior, and posteriors.
    let total_start = Instant::now();
    let out_dir = "out/matern_1form_torus";
    let t = Instant::now();
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;
    println!("out dir prep: {:.3}s", t.elapsed().as_secs_f64());

    let mesh_path = "meshes/torus_shell.msh";
    let t = Instant::now();
    let mesh_bytes = fs::read(mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    println!("mesh load + metric: {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let reconstructed_field_operator =
        build_reconstructed_barycenter_field_operator(&topology, &coords).map_err(invalid_data)?;
    println!(
        "barycenter reconstruction operator (ambient dim={}, cells={}): {:.3}s",
        reconstructed_field_operator.ambient_dim(),
        reconstructed_field_operator.cell_count(),
        t.elapsed().as_secs_f64()
    );

    let grade = 1;
    // Torus has two harmonic 1-forms (dim H1 = 2).
    let homology_dim = 2;

    let kappa = 20.;
    let tau = 1.;

    let (_nu, _variance, euclidean_effective_range) =
        convert_whittle_params_to_matern(2., tau, kappa, 2);

    println!("Euclidean effective range: {}", euclidean_effective_range);

    let t = Instant::now();
    let source_form = DiffFormClosure::one_form(
        |p| {
            let x = p[0];
            let y = p[1];
            let rho = (x * x + y * y).sqrt().max(1e-12);
            // Swirl around the major circle (tangent to the torus).
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
    let harmonic_basis = hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
        &topology,
        &galmats,
        grade,
        homology_dim,
        None,
        None,
    );
    println!(
        "harmonic basis build (dim={}): {:.3}s",
        harmonic_basis.ncols(),
        t.elapsed().as_secs_f64()
    );

    // ---- FEEC solve (reference solution) ----
    let t = Instant::now();
    let (_sigma, u, _harmonics) = hodge_laplace::solve_hodge_laplace_source_with_galmats(
        &topology,
        &galmats,
        source_data.clone(),
        grade,
        homology_dim,
    );
    println!(
        "FEEC solve (hodge laplace): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Matérn prior precision ----
    let t = Instant::now();
    let hodge = build_hodge_laplacian_1form_from_galmats(&galmats);
    let prior_precision = build_matern_precision_1form(
        &topology,
        &metric,
        &hodge,
        MaternConfig {
            kappa: kappa,
            tau: tau,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
    );
    println!("prior precision build: {:.3}s", t.elapsed().as_secs_f64());

    let ndofs = hodge.mass_u.nrows();
    let u_sol = u.coeffs.clone();
    let u_gmrf = feec_vec_to_gmrf(&u_sol);

    // ---- Convert FEEC matrices/vectors to GMRF types ----
    let t = Instant::now();
    let h_gmrf = feec_csr_to_gmrf(&hodge.laplacian);
    let y_gmrf = feec_vec_to_gmrf(&source_data.clone());
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);

    println!("convert matrix types: {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let harmonic_constraint_matrix =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u);
    let harmonic_constraint_rhs = GmrfVector::zeros(harmonic_constraint_matrix.nrows());
    println!(
        "harmonic constraints build ({}x{}): {:.3}s",
        harmonic_constraint_matrix.nrows(),
        harmonic_constraint_matrix.ncols(),
        t.elapsed().as_secs_f64()
    );

    let num_mc_samples = 256;

    // ---- Prior (mean=0) and MC variance estimate ----
    let t = Instant::now();
    let q_prior_factor = q_prior_gmrf.cholesky_sqrt_lower()?;
    let mut prior = Gmrf::from_mean_and_precision(GmrfVector::zeros(ndofs), q_prior_gmrf.clone())?
        .with_precision_sqrt(q_prior_factor);
    println!("build prior: {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut prior_rng_unconstrained = StdRng::seed_from_u64(13);
    let prior_variances_unconstrained =
        prior.mc_variances(num_mc_samples, &mut prior_rng_unconstrained)?;
    let mut prior_rng_constrained = StdRng::seed_from_u64(113);
    let prior_variances_constrained = constrained_mc_variances(
        &mut prior,
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
        num_mc_samples,
        &mut prior_rng_constrained,
    )?;
    println!(
        "prior mc variances (unconstrained + constrained): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Posterior A: PDE observations (H = Laplacian) ----
    let t = Instant::now();
    let noise_variance = 1e-8;
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, noise_variance);

    println!(
        "apply observations (pde): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let q_factor = posterior_precision.cholesky_sqrt_lower()?;
    println!("precision factor (pde): {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut posterior =
        Gmrf::from_information_and_precision_with_sqrt(information, posterior_precision, q_factor)?;
    println!("build posterior (pde): {:.3}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let mut posterior_rng_unconstrained = StdRng::seed_from_u64(17);
    let posterior_variances_unconstrained =
        posterior.mc_variances(num_mc_samples, &mut posterior_rng_unconstrained)?;
    let mut posterior_rng_constrained = StdRng::seed_from_u64(117);
    let posterior_variances_constrained = constrained_mc_variances(
        &mut posterior,
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
        num_mc_samples,
        &mut posterior_rng_constrained,
    )?;
    println!(
        "posterior mc variances (pde, unconstrained + constrained): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Posterior B: selector observations (~10% of FEEC solution) ----
    let obs_fraction = 0.1;
    let mut num_obs = (ndofs as f64 * obs_fraction).round() as usize;
    num_obs = num_obs.clamp(1, ndofs);
    let mut obs_indices: Vec<usize> = (0..ndofs).collect();
    let mut obs_rng = StdRng::seed_from_u64(23);
    obs_indices.shuffle(&mut obs_rng);
    obs_indices.truncate(num_obs);
    obs_indices.sort_unstable();

    let t = Instant::now();
    let observation_matrix = observation_selector(ndofs, &obs_indices);
    let observations = &observation_matrix * &u_gmrf;
    let (posterior_obs_precision, information_obs) = apply_gaussian_observations(
        &q_prior_gmrf,
        &observation_matrix,
        &observations,
        None,
        noise_variance,
    );
    println!(
        "apply observations (selector): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let q_factor_obs = posterior_obs_precision.cholesky_sqrt_lower()?;
    println!(
        "precision factor (selector): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let mut posterior_obs = Gmrf::from_information_and_precision_with_sqrt(
        information_obs,
        posterior_obs_precision,
        q_factor_obs,
    )?;
    println!(
        "build posterior (selector): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let mut posterior_obs_rng_unconstrained = StdRng::seed_from_u64(29);
    let posterior_obs_variances_unconstrained =
        posterior_obs.mc_variances(num_mc_samples, &mut posterior_obs_rng_unconstrained)?;
    let mut posterior_obs_rng_constrained = StdRng::seed_from_u64(129);
    let posterior_obs_variances_constrained = constrained_mc_variances(
        &mut posterior_obs,
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
        num_mc_samples,
        &mut posterior_obs_rng_constrained,
    )?;
    println!(
        "posterior mc variances (selector, unconstrained + constrained): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Quick sample (sanity check) ----
    let t = Instant::now();
    let mut rng_unconstrained = StdRng::seed_from_u64(7);
    let sample_unconstrained = posterior.sample_one_solve(&mut rng_unconstrained)?;
    let harmonic_residual_unconstrained =
        dense_matvec(&harmonic_constraint_matrix, &sample_unconstrained).norm();
    let mut rng_constrained = StdRng::seed_from_u64(7);
    let sample_constrained = posterior.sample_constrained(
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
        &mut rng_constrained,
    )?;
    let harmonic_residual_constrained =
        dense_matvec(&harmonic_constraint_matrix, &sample_constrained).norm();
    println!(
        "gmrf samples: {:.3}s (harmonic residual l2: unconstrained={harmonic_residual_unconstrained:.3e}, constrained={harmonic_residual_constrained:.3e})",
        t.elapsed().as_secs_f64()
    );

    // ---- Write FEEC solution fields ----
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
    println!("write solution VTK: {:.3}s", t.elapsed().as_secs_f64());

    // ---- Write prior mean + variance ----
    let t = Instant::now();
    let prior_mean = gmrf_vec_to_feec(prior.mean());
    let prior_var_unconstrained = gmrf_vec_to_feec(&prior_variances_unconstrained);
    let prior_var_constrained = gmrf_vec_to_feec(&prior_variances_constrained);
    let prior_mean_cochain = Cochain::new(grade, prior_mean);
    let prior_var_unconstrained_cochain = Cochain::new(grade, prior_var_unconstrained);
    let prior_var_constrained_cochain = Cochain::new(grade, prior_var_constrained);
    write_1cochain_vtk_fields(
        format!("{out_dir}/prior_mean.vtk"),
        &coords,
        &topology,
        &[
            ("prior_mean", &prior_mean_cochain),
            (
                "prior_variance_mc_unconstrained",
                &prior_var_unconstrained_cochain,
            ),
            (
                "prior_variance_mc_constrained",
                &prior_var_constrained_cochain,
            ),
        ],
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/prior_mean_vector_field.vtk"),
        &coords,
        &topology,
        &prior_mean_cochain,
        "prior_mean_vector_field",
    )?;
    println!(
        "write prior mean/var VTK: {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Write posterior A (PDE) mean + variance ----
    let t = Instant::now();
    let posterior_mean_harmonic_residual_unconstrained =
        dense_matvec(&harmonic_constraint_matrix, posterior.mean()).norm();
    let posterior_mean_constrained = constrained_mean(
        &mut posterior,
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
    )?;
    let posterior_mean_harmonic_residual_constrained =
        dense_matvec(&harmonic_constraint_matrix, &posterior_mean_constrained).norm();
    println!(
        "posterior mean (pde) harmonic residual l2: unconstrained={posterior_mean_harmonic_residual_unconstrained:.3e}, constrained={posterior_mean_harmonic_residual_constrained:.3e}"
    );
    let posterior_mean = gmrf_vec_to_feec(&posterior_mean_constrained);
    let posterior_var_unconstrained = gmrf_vec_to_feec(&posterior_variances_unconstrained);
    let posterior_var_constrained = gmrf_vec_to_feec(&posterior_variances_constrained);
    let posterior_mean_cochain = Cochain::new(grade, posterior_mean);
    let posterior_var_unconstrained_cochain = Cochain::new(grade, posterior_var_unconstrained);
    let posterior_var_constrained_cochain = Cochain::new(grade, posterior_var_constrained);
    let posterior_diff = &posterior_mean_cochain.coeffs - &u_sol;
    let posterior_diff_cochain = Cochain::new(grade, posterior_diff);
    write_1cochain_vtk_fields(
        format!("{out_dir}/posterior_mean.vtk"),
        &coords,
        &topology,
        &[
            ("posterior_mean", &posterior_mean_cochain),
            (
                "posterior_variance_mc_unconstrained",
                &posterior_var_unconstrained_cochain,
            ),
            (
                "posterior_variance_mc_constrained",
                &posterior_var_constrained_cochain,
            ),
        ],
    )?;
    write_cochain_vtk(
        format!("{out_dir}/posterior_mean_diff.vtk"),
        &coords,
        &topology,
        &posterior_diff_cochain,
        "posterior_mean_minus_solution",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/posterior_mean_vector_field.vtk"),
        &coords,
        &topology,
        &posterior_mean_cochain,
        "posterior_mean_vector_field",
    )?;
    let posterior_mean_unconstrained = posterior.mean().clone();
    let mut posterior_reconstructed_rng_unconstrained = StdRng::seed_from_u64(217);
    let posterior_reconstructed_variance_unconstrained =
        estimate_reconstructed_field_component_variances(
            &reconstructed_field_operator,
            &posterior_mean_unconstrained,
            num_mc_samples,
            &mut posterior_reconstructed_rng_unconstrained,
            |rng| posterior.sample_one_solve(rng),
        )?;
    let mut posterior_reconstructed_rng_constrained = StdRng::seed_from_u64(317);
    let posterior_reconstructed_variance_constrained =
        estimate_reconstructed_field_component_variances(
            &reconstructed_field_operator,
            &posterior_mean_constrained,
            num_mc_samples,
            &mut posterior_reconstructed_rng_constrained,
            |rng| {
                posterior.sample_constrained(
                    &harmonic_constraint_matrix,
                    &harmonic_constraint_rhs,
                    rng,
                )
            },
        )?;
    write_reconstructed_field_vtk(
        format!("{out_dir}/posterior_surface_vector_field.vtk"),
        &coords,
        &topology,
        "posterior_surface_vector",
        &posterior_mean_cochain,
        &posterior_reconstructed_variance_unconstrained,
        &posterior_reconstructed_variance_constrained,
    )?;
    println!(
        "write posterior mean/var VTK (+ reconstructed field uncertainty): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    // ---- Write posterior B (selector) mean + variance ----
    let t = Instant::now();
    let posterior_obs_mean_harmonic_residual_unconstrained =
        dense_matvec(&harmonic_constraint_matrix, posterior_obs.mean()).norm();
    let posterior_obs_mean_constrained = constrained_mean(
        &mut posterior_obs,
        &harmonic_constraint_matrix,
        &harmonic_constraint_rhs,
    )?;
    let posterior_obs_mean_harmonic_residual_constrained =
        dense_matvec(&harmonic_constraint_matrix, &posterior_obs_mean_constrained).norm();
    println!(
        "posterior mean (selector) harmonic residual l2: unconstrained={posterior_obs_mean_harmonic_residual_unconstrained:.3e}, constrained={posterior_obs_mean_harmonic_residual_constrained:.3e}"
    );
    let posterior_obs_mean = gmrf_vec_to_feec(&posterior_obs_mean_constrained);
    let posterior_obs_var_unconstrained = gmrf_vec_to_feec(&posterior_obs_variances_unconstrained);
    let posterior_obs_var_constrained = gmrf_vec_to_feec(&posterior_obs_variances_constrained);
    let posterior_obs_mean_cochain = Cochain::new(grade, posterior_obs_mean);
    let posterior_obs_var_unconstrained_cochain =
        Cochain::new(grade, posterior_obs_var_unconstrained);
    let posterior_obs_var_constrained_cochain = Cochain::new(grade, posterior_obs_var_constrained);
    let posterior_obs_diff = &posterior_obs_mean_cochain.coeffs - &u_sol;
    let posterior_obs_diff_cochain = Cochain::new(grade, posterior_obs_diff);
    write_1cochain_vtk_fields(
        format!("{out_dir}/posterior_obs10pct_mean.vtk"),
        &coords,
        &topology,
        &[
            ("posterior_obs10pct_mean", &posterior_obs_mean_cochain),
            (
                "posterior_obs10pct_variance_mc_unconstrained",
                &posterior_obs_var_unconstrained_cochain,
            ),
            (
                "posterior_obs10pct_variance_mc_constrained",
                &posterior_obs_var_constrained_cochain,
            ),
        ],
    )?;
    write_cochain_vtk(
        format!("{out_dir}/posterior_obs10pct_mean_diff.vtk"),
        &coords,
        &topology,
        &posterior_obs_diff_cochain,
        "posterior_obs10pct_mean_minus_solution",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/posterior_obs10pct_mean_vector_field.vtk"),
        &coords,
        &topology,
        &posterior_obs_mean_cochain,
        "posterior_obs10pct_mean_vector_field",
    )?;
    let posterior_obs_mean_unconstrained = posterior_obs.mean().clone();
    let mut posterior_obs_reconstructed_rng_unconstrained = StdRng::seed_from_u64(229);
    let posterior_obs_reconstructed_variance_unconstrained =
        estimate_reconstructed_field_component_variances(
            &reconstructed_field_operator,
            &posterior_obs_mean_unconstrained,
            num_mc_samples,
            &mut posterior_obs_reconstructed_rng_unconstrained,
            |rng| posterior_obs.sample_one_solve(rng),
        )?;
    let mut posterior_obs_reconstructed_rng_constrained = StdRng::seed_from_u64(329);
    let posterior_obs_reconstructed_variance_constrained =
        estimate_reconstructed_field_component_variances(
            &reconstructed_field_operator,
            &posterior_obs_mean_constrained,
            num_mc_samples,
            &mut posterior_obs_reconstructed_rng_constrained,
            |rng| {
                posterior_obs.sample_constrained(
                    &harmonic_constraint_matrix,
                    &harmonic_constraint_rhs,
                    rng,
                )
            },
        )?;
    write_reconstructed_field_vtk(
        format!("{out_dir}/posterior_obs10pct_surface_vector_field.vtk"),
        &coords,
        &topology,
        "posterior_obs10pct_surface_vector",
        &posterior_obs_mean_cochain,
        &posterior_obs_reconstructed_variance_unconstrained,
        &posterior_obs_reconstructed_variance_constrained,
    )?;
    println!(
        "write selector posterior mean/var VTK (+ reconstructed field uncertainty): {:.3}s",
        t.elapsed().as_secs_f64()
    );

    let (mean_unconstrained, std_unconstrained) = summarize(&sample_unconstrained);
    let (mean_constrained, std_constrained) = summarize(&sample_constrained);
    let (prior_var_unconstrained_mean, prior_var_unconstrained_std) =
        summarize(&prior_variances_unconstrained);
    let (prior_var_constrained_mean, prior_var_constrained_std) =
        summarize(&prior_variances_constrained);
    let (posterior_var_unconstrained_mean, posterior_var_unconstrained_std) =
        summarize(&posterior_variances_unconstrained);
    let (posterior_var_constrained_mean, posterior_var_constrained_std) =
        summarize(&posterior_variances_constrained);
    let (posterior_obs_var_unconstrained_mean, posterior_obs_var_unconstrained_std) =
        summarize(&posterior_obs_variances_unconstrained);
    let (posterior_obs_var_constrained_mean, posterior_obs_var_constrained_std) =
        summarize(&posterior_obs_variances_constrained);
    println!("1-form dofs: {ndofs}");
    println!(
        "selector observations: {num_obs} of {ndofs} (~{:.1}%)",
        100.0 * num_obs as f64 / ndofs as f64
    );
    println!(
        "posterior sample unconstrained mean={mean_unconstrained:.6} std={std_unconstrained:.6}"
    );
    println!("posterior sample constrained mean={mean_constrained:.6} std={std_constrained:.6}");
    println!(
        "prior variances (unconstrained mc, n={num_mc_samples}) stats: mean={prior_var_unconstrained_mean:.6} std={prior_var_unconstrained_std:.6}"
    );
    println!(
        "prior variances (constrained mc, n={num_mc_samples}) stats: mean={prior_var_constrained_mean:.6} std={prior_var_constrained_std:.6}"
    );
    println!(
        "posterior variances (pde, unconstrained mc, n={num_mc_samples}) stats: mean={posterior_var_unconstrained_mean:.6} std={posterior_var_unconstrained_std:.6}"
    );
    println!(
        "posterior variances (pde, constrained mc, n={num_mc_samples}) stats: mean={posterior_var_constrained_mean:.6} std={posterior_var_constrained_std:.6}"
    );
    println!(
        "posterior variances (selector, unconstrained mc, n={num_mc_samples}) stats: mean={posterior_obs_var_unconstrained_mean:.6} std={posterior_obs_var_unconstrained_std:.6}"
    );
    println!(
        "posterior variances (selector, constrained mc, n={num_mc_samples}) stats: mean={posterior_obs_var_constrained_mean:.6} std={posterior_obs_var_constrained_std:.6}"
    );
    println!("Loaded mesh from {mesh_path}");
    println!("Wrote VTK outputs to {out_dir}");
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());

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

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn estimate_reconstructed_field_component_variances<R, F>(
    operator: &feg_infer::matern_1form::ReconstructedBarycenterFieldOperator,
    mean: &GmrfVector,
    num_samples: usize,
    rng: &mut R,
    mut sample_draw: F,
) -> Result<ReconstructedBarycenterField, Box<dyn std::error::Error>>
where
    R: rand::Rng + ?Sized,
    F: FnMut(&mut R) -> Result<GmrfVector, gmrf_core::GmrfError>,
{
    if num_samples == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at least one sample is required for reconstructed field variances",
        )
        .into());
    }

    let mean_field = operator
        .apply_to_slice(mean.as_slice())
        .map_err(invalid_data)?;
    let mut component_sums = (0..mean_field.ambient_dim())
        .map(|_| FeecVector::zeros(mean_field.cell_count()))
        .collect::<Vec<_>>();

    for _ in 0..num_samples {
        let draw = sample_draw(rng)?;
        let draw_field = operator
            .apply_to_slice(draw.as_slice())
            .map_err(invalid_data)?;
        for (sum, (draw_component, mean_component)) in component_sums.iter_mut().zip(
            draw_field
                .components()
                .iter()
                .zip(mean_field.components().iter()),
        ) {
            let centered = draw_component - mean_component;
            *sum += centered.component_mul(&centered);
        }
    }

    for component_sum in &mut component_sums {
        *component_sum /= num_samples as f64;
    }

    ReconstructedBarycenterField::from_components(component_sums)
        .map_err(invalid_data)
        .map_err(Into::into)
}

fn write_reconstructed_field_vtk(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    mean_cochain: &Cochain,
    variance_unconstrained: &ReconstructedBarycenterField,
    variance_constrained: &ReconstructedBarycenterField,
) -> Result<(), Box<dyn std::error::Error>> {
    if mean_cochain.dim() != 1 {
        return Err(invalid_data("surface vector output expects a 1-cochain").into());
    }
    if variance_unconstrained.ambient_dim() != 3 || variance_constrained.ambient_dim() != 3 {
        return Err(invalid_data(
            "reconstructed field VTK output currently expects three ambient components",
        )
        .into());
    }

    let mean_vectors = sample_1form_cell_vectors(coords, topology, mean_cochain)?;
    let mean_magnitude = vector_magnitudes(&mean_vectors);
    let variance_vectors_unconstrained = variance_unconstrained.vtk_vectors();
    let variance_vectors_constrained = variance_constrained.vtk_vectors();
    let trace_unconstrained = variance_unconstrained.trace();
    let trace_constrained = variance_constrained.trace();
    let std_trace_unconstrained = trace_unconstrained.map(|value| value.max(0.0).sqrt());
    let std_trace_constrained = trace_constrained.map(|value| value.max(0.0).sqrt());

    write_top_cell_vtk_fields(
        path,
        coords,
        topology,
        &[
            (vector_name, mean_vectors.as_slice()),
            (
                "variance_vector_unconstrained",
                variance_vectors_unconstrained.as_slice(),
            ),
            (
                "variance_vector_constrained",
                variance_vectors_constrained.as_slice(),
            ),
        ],
        &[
            ("magnitude", mean_magnitude.as_slice()),
            (
                "marginal_variance_unconstrained",
                trace_unconstrained.as_slice(),
            ),
            (
                "marginal_std_unconstrained",
                std_trace_unconstrained.as_slice(),
            ),
            (
                "marginal_variance_constrained",
                trace_constrained.as_slice(),
            ),
            ("marginal_std_constrained", std_trace_constrained.as_slice()),
        ],
    )?;

    Ok(())
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
        .collect()
}

fn build_harmonic_orthogonality_constraints(
    harmonic_basis: &FeecMatrix,
    mass_u: &FeecCsr,
) -> GmrfDenseMatrix {
    let weighted_basis = mass_u * harmonic_basis;
    // Enforce H^T M x = 0, i.e. orthogonality to span(H) in the FEEC mass inner product.
    GmrfDenseMatrix::from_fn(harmonic_basis.ncols(), harmonic_basis.nrows(), |i, j| {
        weighted_basis[(j, i)]
    })
}

fn dense_matvec(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> GmrfVector {
    assert_eq!(matrix.ncols(), vector.len());
    let mut out = GmrfVector::zeros(matrix.nrows());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let xj = vector[j];
        if xj == 0.0 {
            continue;
        }
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        for (i, value) in col.as_slice().iter().enumerate() {
            out[i] += *value * xj;
        }
    }
    out
}

fn constrained_mc_variances<R: rand::Rng + ?Sized>(
    posterior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
    num_samples: usize,
    rng: &mut R,
) -> Result<GmrfVector, gmrf_core::GmrfError> {
    if num_samples == 0 {
        return Err(gmrf_core::GmrfError::DimensionMismatch(
            "at least one sample is required",
        ));
    }

    let constrained_mean = constrained_mean(posterior, constraint_matrix, constraint_rhs)?;
    let dim = posterior.dimension();
    let mut variances = GmrfVector::zeros(dim);
    for _ in 0..num_samples {
        let draw = posterior.sample_constrained(constraint_matrix, constraint_rhs, rng)?;
        let centered = &draw - &constrained_mean;
        variances += centered.component_mul(&centered);
    }
    Ok(variances / (num_samples as f64))
}

fn constrained_mean(
    posterior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
    constraint_rhs: &GmrfVector,
) -> Result<GmrfVector, gmrf_core::GmrfError> {
    if constraint_matrix.ncols() != posterior.dimension() {
        return Err(gmrf_core::GmrfError::DimensionMismatch(
            "constraint matrix columns must match latent dimension",
        ));
    }
    if constraint_matrix.nrows() != constraint_rhs.len() {
        return Err(gmrf_core::GmrfError::DimensionMismatch(
            "constraint rhs length must match constraint matrix rows",
        ));
    }
    if constraint_matrix.nrows() == 0 {
        return Ok(posterior.mean().clone());
    }

    let covariance_times_constraint_t =
        covariance_times_constraint_t(posterior, constraint_matrix)?;
    let predicted_constraints = dense_matvec(constraint_matrix, posterior.mean());
    let mut lagrange_rhs = constraint_rhs - &predicted_constraints;

    let schur = schur_complement(constraint_matrix, &covariance_times_constraint_t);
    let schur_factor = schur
        .llt(Side::Lower)
        .map_err(|_| gmrf_core::GmrfError::SingularConstraintSystem)?;
    schur_factor.solve_in_place(lagrange_rhs.as_col_mut().as_mat_mut());

    let correction = dense_matvec(&covariance_times_constraint_t, &lagrange_rhs);
    Ok(posterior.mean() + correction)
}

fn covariance_times_constraint_t(
    posterior: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
) -> Result<GmrfDenseMatrix, gmrf_core::GmrfError> {
    let state_dim = posterior.dimension();
    let constraint_dim = constraint_matrix.nrows();
    let mut columns = Vec::with_capacity(constraint_dim);
    for row in 0..constraint_dim {
        let rhs = dense_row_as_vector(constraint_matrix, row);
        let solved = posterior.solve_precision(&rhs)?;
        columns.push(solved);
    }

    Ok(GmrfDenseMatrix::from_fn(
        state_dim,
        constraint_dim,
        |i, j| columns[j][i],
    ))
}

fn dense_row_as_vector(matrix: &GmrfDenseMatrix, row: usize) -> GmrfVector {
    let mut out = GmrfVector::zeros(matrix.ncols());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        out[j] = col.as_slice()[row];
    }
    out
}

fn schur_complement(
    constraint_matrix: &GmrfDenseMatrix,
    covariance_times_constraint_t: &GmrfDenseMatrix,
) -> GmrfDenseMatrix {
    let mut schur = GmrfDenseMatrix::zeros(constraint_matrix.nrows(), constraint_matrix.nrows());
    for i in 0..constraint_matrix.nrows() {
        for j in 0..constraint_matrix.nrows() {
            let mut sum = 0.0;
            for k in 0..constraint_matrix.ncols() {
                sum += constraint_matrix[(i, k)] * covariance_times_constraint_t[(k, j)];
            }
            schur[(i, j)] = sum;
        }
    }
    schur
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::types::{CooMatrix, SparseMatrix};

    fn identity_precision(size: usize) -> SparseMatrix {
        let mut coo = CooMatrix::new(size, size);
        for i in 0..size {
            coo.push(i, i, 1.0);
        }
        SparseMatrix::from(&coo)
    }

    #[test]
    fn constrained_mean_matches_identity_projection_formula() {
        let mean = GmrfVector::from_vec(vec![1.0, -1.0]);
        let precision = identity_precision(2);
        let mut gmrf = Gmrf::from_mean_and_precision(mean, precision).unwrap();

        let constraints = GmrfDenseMatrix::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 2.0 });
        let rhs = GmrfVector::from_vec(vec![0.0]);
        let constrained = constrained_mean(&mut gmrf, &constraints, &rhs).unwrap();

        let expected = GmrfVector::from_vec(vec![1.2, -0.6]);
        assert!((&constrained - &expected).norm() < 1e-12);
        assert!((dense_matvec(&constraints, &constrained) - rhs).norm() < 1e-12);
    }

    #[test]
    fn constrained_mean_with_empty_constraints_returns_original_mean() {
        let mean = GmrfVector::from_vec(vec![0.3, -0.7, 1.1]);
        let precision = identity_precision(3);
        let mut gmrf = Gmrf::from_mean_and_precision(mean.clone(), precision).unwrap();

        let constraints = GmrfDenseMatrix::zeros(0, 3);
        let rhs = GmrfVector::zeros(0);
        let constrained = constrained_mean(&mut gmrf, &constraints, &rhs).unwrap();

        assert!((constrained - mean).norm() < 1e-12);
    }
}
