use common::linalg::nalgebra::{CsrMatrix, Vector as FeecVector};
use ddf::cochain::{cochain_projection, Cochain};
use exterior::field::DiffFormClosure;
use feg_infer::matern_0form::{
    build_matern_precision_0form, feec_csr_to_gmrf, feec_vec_to_gmrf, LaplaceBeltrami0Form,
    MaternConfig, MaternMassInverse,
};
use feg_infer::util::convert_whittle_params_to_matern;
use formoniq::{
    assemble,
    fe::{fe_l2_error, l2_norm},
    io::write_cochain_vtk,
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::laplace_beltrami::{self, LaplaceBeltramiGalmats},
};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::types::Vector as GmrfVector;
use gmrf_core::Gmrf;
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::geometry::coord::CoordRef;
use rand::{rngs, SeedableRng};
use std::collections::HashSet;
use std::f64::consts::PI;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "out/mixed_bc_2_posterior";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let dim = 2;
    let resolutions = [8, 16, 32, 64];

    let exact_solution = DiffFormClosure::scalar(
        |p| 1. + p[0] + p[1] + p[0] * p[1] + (PI * p[0]).sin() * (PI * p[1]).sin(),
        dim,
    );
    let rhs = DiffFormClosure::scalar(
        |p| 2. * PI * PI * (PI * p[1]).sin() * (PI * p[0]).sin(),
        dim,
    );
    let inner_product_weight = InnerProductWeightClosure::new(|_p| 1.0);

    let mut errors = Vec::new();
    let mut hs = Vec::new();

    let tau = 1.;
    let kappa = 10.;
    let alpha = 2.0; // This is the alpha when using precision KT*M*K
    let (_nu, variance, effective_range) = convert_whittle_params_to_matern(alpha, tau, kappa, dim);

    println!("Equivalent Matern parameters:");
    println!("  variance: {variance}");
    println!("  effective range: {effective_range}");

    for &resolution in &resolutions {
        let mesh = CartesianMeshInfo::new_unit_scaled(dim, resolution, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);
        let galmats = LaplaceBeltramiGalmats::compute(&topology, &metric);
        // let stiffness = galmats.stiffness_csr();
        // let mass = galmats.mass_csr();

        let rhs_vec = assemble::assemble_galvec(
            &topology,
            &metric,
            SourceElVec::new_weighted(&rhs, &coords, None, &inner_product_weight),
        );

        let solution_projected = cochain_projection(&exact_solution, &topology, &coords, None);

        let strong_dof_predicate = |p: CoordRef| p[1] == 1.0;
        // let strong_dof_predicate = |p: CoordRef| true;
        let weak_dof_predicate = |p: CoordRef| !strong_dof_predicate(p);

        let dirichlet_dofs = assemble::boundary_simplices_where_barycenter(
            &topology,
            &coords,
            0,
            strong_dof_predicate,
        )
        .into_iter()
        .collect::<HashSet<usize>>();
        let dirichlet_dof_selector = |sidx: usize| dirichlet_dofs.contains(&sidx);
        let dirichlet_boundary_data = |vidx: usize| solution_projected[vidx];

        let neumann_dofs = assemble::boundary_simplices_where_barycenter(
            &topology,
            &coords,
            1,
            weak_dof_predicate,
        );
        let neumann_data = DiffFormClosure::scalar(
            |p| {
                if p[0] == 0.0 {
                    -(1. + p[1] + PI * (PI * p[1]).sin())
                } else if p[0] == 1.0 {
                    1. + p[1] - PI * (PI * p[1]).sin()
                } else if p[1] == 0.0 {
                    -(1. + p[0] + PI * (PI * p[0]).sin())
                } else if p[1] == 1.0 {
                    1. + p[0] - PI * (PI * p[0]).sin()
                } else {
                    0.0
                }
            },
            1,
        );
        let neumann_dof_selector =
            |kidx: manifold::topology::handle::KSimplexIdx| neumann_dofs.contains(&kidx);

        let neumann_rhs = assemble::assemble_boundary_galvec(
            &topology,
            &metric,
            SourceElVec::new(&neumann_data, &coords, None),
            neumann_dof_selector,
        );

        let total_rhs = rhs_vec.clone() + neumann_rhs.clone();

        // let observations = rhs_vec.neg() + neumann_rhs.neg();

        let galsol = laplace_beltrami::solve_laplace_beltrami_source(
            &topology,
            &metric,
            total_rhs,
            dirichlet_boundary_data,
            Some(&dirichlet_dof_selector),
        );

        let ndofs = galsol.len();
        let kappa = 10.;
        let tau = 1.0;
        // let kappa2 = kappa * kappa;
        // let prior_precision = &stiffness + kappa2 * &mass;

        let mut laplace_galmat = galmats.stiffness().clone();
        let mut source_galvec = rhs_vec.clone() + neumann_rhs.clone();

        assemble::enforce_dirichlet_bc_partial(
            &topology,
            dirichlet_boundary_data,
            &mut laplace_galmat,
            &mut source_galvec,
            Some(&dirichlet_dof_selector),
        );

        let mut mass_galmat = galmats.mass().clone();
        let mut dummy_rhs = FeecVector::zeros(source_galvec.len());

        // Set mass matrix rows corresponding to Dirichlet dofs to identity
        assemble::enforce_dirichlet_bc_partial(
            &topology,
            dirichlet_boundary_data, // Irrelevant
            &mut mass_galmat,
            &mut dummy_rhs,
            Some(&dirichlet_dof_selector),
        );

        let laplace_csr = CsrMatrix::from(&laplace_galmat);
        let mass_csr = CsrMatrix::from(&mass_galmat);

        let observations = feec_vec_to_gmrf(&source_galvec.clone());

        let prior_precision = build_matern_precision_0form(
            &LaplaceBeltrami0Form {
                laplacian: laplace_csr.clone(),
                mass: mass_csr,
            },
            MaternConfig {
                kappa,
                tau,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );
        let q_prior = feec_csr_to_gmrf(&prior_precision);
        let prior_mean = GmrfVector::zeros(ndofs);
        let mut prior = Gmrf::from_mean_and_precision(prior_mean, q_prior.clone())?;

        let mut rng = rngs::StdRng::seed_from_u64(5);
        let prior_variances = prior.mc_variances(256, &mut rng)?;

        // let stiffness = galmats.stiffness_csr();
        // let observation_matrix = feec_csr_to_gmrf(&stiffness);
        let observation_matrix = feec_csr_to_gmrf(&laplace_csr);
        let noise_variance = 0.0000000001;
        let (posterior_precision, information) = apply_gaussian_observations(
            &q_prior,
            &observation_matrix,
            &observations,
            None,
            noise_variance,
        );

        let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)?;
        let posterior_mean = gmrf_vec_to_feec(posterior.mean());
        let posterior_mean_cochain = Cochain::new(0, posterior_mean);
        let posterior_variances = posterior.mc_variances(256, &mut rng)?;
        let posterior_var_cochain = Cochain::new(0, gmrf_vec_to_feec(&posterior_variances));
        let prior_mean_cochain = Cochain::new(0, gmrf_vec_to_feec(prior.mean()));
        let prior_var_cochain = Cochain::new(0, gmrf_vec_to_feec(&prior_variances));

        let feec_error = fe_l2_error(&galsol, &exact_solution, &topology, &coords);
        let error_l2 = fe_l2_error(&posterior_mean_cochain, &exact_solution, &topology, &coords);
        let posterior_galsol_diff = posterior_mean_cochain.clone() - galsol.clone();
        let posterior_galsol_l2 = l2_norm(&posterior_galsol_diff, &topology, &metric);
        let h = 1.0 / resolution as f64;
        println!(
            "resolution={resolution} h={h:.4} dofs={ndofs} L2 error={error_l2:.6e} (feec={feec_error:.6e}, post-feec={posterior_galsol_l2:.6e})"
        );
        if let (Some(&prev_err), Some(&prev_h)) = (errors.last(), hs.last()) {
            let rate = f64::ln(prev_err / error_l2) / f64::ln(prev_h / h);
            println!("  observed rate ~ {rate:.2}");
        }
        let (prior_mean_mean, prior_mean_std) = summarize(prior.mean());
        let (prior_var_mean, prior_var_std) = summarize(&prior_variances);
        println!(
            "  prior stats: mean={prior_mean_mean:.6e} std={prior_mean_std:.6e} var_mean={prior_var_mean:.6e} var_std={prior_var_std:.6e}"
        );

        let res_dir = format!("{out_dir}/res_{resolution}");
        fs::create_dir_all(&res_dir)?;
        write_cochain_vtk(
            format!("{res_dir}/solution_feec.vtk"),
            &coords,
            &topology,
            &galsol,
            "solution_feec",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/posterior_mean.vtk"),
            &coords,
            &topology,
            &posterior_mean_cochain,
            "posterior_mean",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/posterior_variance.vtk"),
            &coords,
            &topology,
            &posterior_var_cochain,
            "posterior_variance",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/prior_mean.vtk"),
            &coords,
            &topology,
            &prior_mean_cochain,
            "prior_mean",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/prior_variance.vtk"),
            &coords,
            &topology,
            &prior_var_cochain,
            "prior_variance",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/solution_exact.vtk"),
            &coords,
            &topology,
            &solution_projected,
            "solution_exact",
        )?;
        let solution_difference = posterior_mean_cochain.clone() - solution_projected.clone();
        write_cochain_vtk(
            format!("{res_dir}/solution_difference.vtk"),
            &coords,
            &topology,
            &solution_difference,
            "solution_difference",
        )?;
        let feec_solution_difference = galsol.clone() - solution_projected.clone();
        write_cochain_vtk(
            format!("{res_dir}/feec_solution_difference.vtk"),
            &coords,
            &topology,
            &feec_solution_difference,
            "feec_solution_difference",
        )?;

        errors.push(error_l2);
        hs.push(h);
    }

    if errors.len() > 1 {
        for (idx, (prev, next)) in errors.iter().zip(errors.iter().skip(1)).enumerate() {
            assert!(
                *next <= *prev * 1.05,
                "error increased between resolutions {} and {}: {prev:.3e} -> {next:.3e}",
                resolutions[idx],
                resolutions[idx + 1]
            );
        }
    }

    println!("Wrote posterior mean and exact solution VTK outputs to {out_dir}");
    Ok(())
}

fn gmrf_vec_to_feec(vec: &gmrf_core::types::Vector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

fn summarize(v: &gmrf_core::types::Vector) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}
