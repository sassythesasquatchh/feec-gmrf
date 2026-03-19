use common::linalg::nalgebra::Vector as FeecVector;
use feg_infer::matern_0form::{
    build_laplace_beltrami_0form, build_matern_precision_0form,
    feec_csr_to_gmrf as feec_csr_to_gmrf_0form, feec_vec_to_gmrf as feec_vec_to_gmrf_0form,
    MaternConfig as Matern0Config, MaternMassInverse as Matern0MassInverse,
};
use feg_infer::matern_1form::{
    build_hodge_laplacian_1form, build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf,
    MaternConfig, MaternMassInverse,
};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::Gmrf;
use manifold::gen::cartesian::CartesianMeshInfo;
use rand::rngs::StdRng;
use rand::SeedableRng;

// Smoke test for FEEC -> GMRF integration.
// This exercises: FEEC assembly, conversion to GMRF sparse types, Gaussian conditioning,
// and sampling from the resulting posterior.
#[test]
fn feec_gmrf_pipeline_samples() {
    // 1) Build a tiny mesh and FEEC operators.
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let hodge = build_hodge_laplacian_1form(&topology, &metric);
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

    // 2) Create a synthetic "true" field and its PDE observation.
    let ndofs = hodge.mass_u.nrows();
    let u_true = FeecVector::from_iterator(ndofs, (0..ndofs).map(|i| (i as f64 * 0.11).cos()));
    let rhs = &hodge.laplacian * &u_true;

    // 3) Convert FEEC matrices/vectors into GMRF types and condition on observations.
    let h_gmrf = feec_csr_to_gmrf(&hodge.laplacian);
    let y_gmrf = feec_vec_to_gmrf(&rhs);
    let q_prior_gmrf = feec_csr_to_gmrf(&prior_precision);

    let noise_variance = 1e-3;
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, noise_variance);

    // 4) Build the posterior and draw a sample to verify basic sanity.
    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .expect("posterior should build");

    let mut rng = StdRng::seed_from_u64(42);
    let sample = posterior.sample(&mut rng).expect("sample should succeed");

    // 5) Sanity checks: dimension and finite values.
    assert_eq!(sample.len(), ndofs);
    assert!(sample.iter().all(|v| v.is_finite()));
}

#[test]
fn feec_gmrf_pipeline_samples_for_0forms() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplace = build_laplace_beltrami_0form(&topology, &metric);
    let prior_precision = build_matern_precision_0form(
        &laplace,
        Matern0Config {
            kappa: 2.0,
            tau: 1.0,
            mass_inverse: Matern0MassInverse::RowSumLumped,
        },
    );

    let ndofs = laplace.mass.nrows();
    let u_true = FeecVector::from_iterator(ndofs, (0..ndofs).map(|i| (i as f64 * 0.17).sin()));
    let rhs = &laplace.laplacian * &u_true;

    let h_gmrf = feec_csr_to_gmrf_0form(&laplace.laplacian);
    let y_gmrf = feec_vec_to_gmrf_0form(&rhs);
    let q_prior_gmrf = feec_csr_to_gmrf_0form(&prior_precision);

    let noise_variance = 1e-3;
    let (posterior_precision, information) =
        apply_gaussian_observations(&q_prior_gmrf, &h_gmrf, &y_gmrf, None, noise_variance);

    let mut posterior = Gmrf::from_information_and_precision(information, posterior_precision)
        .expect("posterior should build");

    let mut rng = StdRng::seed_from_u64(7);
    let sample = posterior.sample(&mut rng).expect("sample should succeed");

    assert_eq!(sample.len(), ndofs);
    assert!(sample.iter().all(|v| v.is_finite()));
}
