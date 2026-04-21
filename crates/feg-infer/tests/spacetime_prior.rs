use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::cochain_projection;
use exterior::field::DiffFormClosure;
use feg_core::{BoundaryRegionSpec, BoundarySpec, BoundaryTreatment};
use feg_infer::model::spacetime::SpacetimeLinearObservationBuilder;
use feg_infer::prior::spacetime::{
    build_0form_spacetime_prior, build_1form_spacetime_prior, SpacetimePriorConfig,
};
use feg_infer::{core_triplet_to_feec_csr, lift_vector_with_layout, reduce_vector_with_layout};
use formoniq::assemble;
use formoniq::problems::spacetime_prior::{
    Hodge1MassInverse, Hodge1PriorConfig, ScalarPriorConfig,
};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::Gmrf;
use manifold::gen::cartesian::CartesianMeshInfo;
use manifold::geometry::coord::CoordRef;

const OBS_VARIANCE: f64 = 1e-10;

#[test]
fn mixed_bc_0form_spacetime_posterior_matches_backward_euler_recurrence() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 4, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);

    let hard_dofs =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, |p: CoordRef| {
            p[1] == 1.0
        });
    let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
        "top-dirichlet",
        hard_dofs.clone(),
        vec![0.0; hard_dofs.len()],
        BoundaryTreatment::HardEssential,
    ));

    let prior = build_0form_spacetime_prior(
        &topology,
        &geometry,
        &boundary,
        ScalarPriorConfig {
            kappa: 1.0,
            tau: 1.0,
        },
        &SpacetimePriorConfig {
            times: vec![0.0, 0.1, 0.2],
        },
    )
    .expect("0-form spacetime prior should assemble");

    let mass = core_triplet_to_feec_csr(&prior.slice.mass);
    let drift = core_triplet_to_feec_csr(&prior.slice.drift);
    let step_matrix = add_sparse(&mass, &scale_matrix(&drift, 0.1));
    let reduced_dim = prior.slice.state_dimension();

    let initial_field = DiffFormClosure::scalar(|p| p[1] * (1.0 - p[1]), 2);
    let initial_full = cochain_projection(&initial_field, &topology, &coords, None);
    let initial_reduced = reduce_vector_with_layout(&prior.slice.layout, initial_full.coeffs())
        .expect("layout reduction");

    let step_solver = |rhs: &FeecVector| {
        let precision = feg_infer::matern_0form::feec_csr_to_gmrf(&step_matrix);
        gmrf_core::Solver::default()
            .solve_matrix(&precision, &feg_infer::matern_0form::feec_vec_to_gmrf(rhs))
            .map(|solution| FeecVector::from_vec(solution.as_slice().to_vec()))
            .expect("backward Euler step should solve")
    };

    let deterministic_1 = step_solver(&(&mass * &initial_reduced));
    let deterministic_2 = step_solver(&(&mass * &deterministic_1));

    let mut builder = SpacetimeLinearObservationBuilder::new(3, &prior.slice);
    builder
        .push_slice_block_from_reduced(
            0,
            &identity_matrix(reduced_dim),
            &initial_reduced,
            &FeecVector::zeros(reduced_dim),
            OBS_VARIANCE,
        )
        .expect("initial condition should stack");
    let left = scale_matrix(&mass, -1.0);
    builder
        .push_transition_block_from_reduced(
            0,
            &left,
            &step_matrix,
            &FeecVector::zeros(reduced_dim),
            &FeecVector::zeros(reduced_dim),
            OBS_VARIANCE,
        )
        .expect("first transition should stack");
    builder
        .push_transition_block_from_reduced(
            1,
            &left,
            &step_matrix,
            &FeecVector::zeros(reduced_dim),
            &FeecVector::zeros(reduced_dim),
            OBS_VARIANCE,
        )
        .expect("second transition should stack");
    let observations = builder.finish();

    let (posterior_precision, information) = apply_gaussian_observations(
        &prior.precision.to_sparse(),
        &observations.matrix,
        &observations.observations,
        Some(&observations.bias),
        observations.noise_variance,
    );
    let posterior =
        Gmrf::from_information_and_precision(information, posterior_precision).expect("posterior");
    let mean = posterior.mean().as_slice();

    assert_slice_close(&mean[0..reduced_dim], initial_reduced.as_slice(), 1e-5);
    assert_slice_close(
        &mean[reduced_dim..2 * reduced_dim],
        deterministic_1.as_slice(),
        1e-5,
    );
    assert_slice_close(
        &mean[2 * reduced_dim..3 * reduced_dim],
        deterministic_2.as_slice(),
        1e-5,
    );
}

#[test]
fn mixed_bc_1form_soft_constraints_approach_hard_boundary_values() {
    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let geometry = coords.to_edge_lengths(&topology);
    let state_dofs = topology
        .boundary_subcomplex_simplices(1)
        .into_iter()
        .take(2)
        .map(|simp| simp.kidx)
        .collect::<Vec<_>>();
    let auxiliary_dofs = topology
        .boundary_subcomplex_simplices(0)
        .into_iter()
        .take(2)
        .map(|simp| simp.kidx)
        .collect::<Vec<_>>();

    let hard_boundary = BoundarySpec::default()
        .with_state_region(BoundaryRegionSpec::new(
            "hard-state",
            state_dofs.clone(),
            vec![1.0; state_dofs.len()],
            BoundaryTreatment::HardEssential,
        ))
        .with_auxiliary_region(BoundaryRegionSpec::new(
            "hard-aux",
            auxiliary_dofs.clone(),
            vec![0.0; auxiliary_dofs.len()],
            BoundaryTreatment::HardEssential,
        ));
    let soft_boundary = BoundarySpec::default()
        .with_state_region(BoundaryRegionSpec::new(
            "soft-state",
            state_dofs.clone(),
            vec![1.0; state_dofs.len()],
            BoundaryTreatment::SoftEssential { variance: 1e-8 },
        ))
        .with_auxiliary_region(BoundaryRegionSpec::new(
            "hard-aux",
            auxiliary_dofs,
            vec![0.0; 2],
            BoundaryTreatment::HardEssential,
        ));
    let time = SpacetimePriorConfig {
        times: vec![0.0, 0.1, 0.2],
    };

    let hard_prior = build_1form_spacetime_prior(
        &topology,
        &geometry,
        &hard_boundary,
        Hodge1PriorConfig {
            kappa: 1.0,
            tau: 1.0,
            mass_inverse: Hodge1MassInverse::Nc1ProjectedSparseInverse,
        },
        &time,
    )
    .expect("hard 1-form prior should assemble");
    let soft_prior = build_1form_spacetime_prior(
        &topology,
        &geometry,
        &soft_boundary,
        Hodge1PriorConfig {
            kappa: 1.0,
            tau: 1.0,
            mass_inverse: Hodge1MassInverse::Nc1ProjectedSparseInverse,
        },
        &time,
    )
    .expect("soft 1-form prior should assemble");

    let hard_mass = core_triplet_to_feec_csr(&hard_prior.slice.mass);
    let hard_drift = core_triplet_to_feec_csr(&hard_prior.slice.drift);
    let hard_step = add_sparse(&hard_mass, &scale_matrix(&hard_drift, 0.1));
    let hard_dim = hard_prior.slice.state_dimension();
    let soft_mass = core_triplet_to_feec_csr(&soft_prior.slice.mass);
    let soft_drift = core_triplet_to_feec_csr(&soft_prior.slice.drift);
    let soft_step = add_sparse(&soft_mass, &scale_matrix(&soft_drift, 0.1));
    let soft_dim = soft_prior.slice.state_dimension();

    let hard_mean = posterior_mean_with_zero_dynamics(&hard_prior, &hard_step, hard_dim, false);
    let soft_mean = posterior_mean_with_zero_dynamics(&soft_prior, &soft_step, soft_dim, true);

    let hard_last = FeecVector::from_vec(hard_mean[2 * hard_dim..3 * hard_dim].to_vec());
    let soft_last = FeecVector::from_vec(soft_mean[2 * soft_dim..3 * soft_dim].to_vec());
    let hard_full =
        lift_vector_with_layout(&hard_prior.slice.layout, &hard_last).expect("hard lift");
    let soft_full =
        lift_vector_with_layout(&soft_prior.slice.layout, &soft_last).expect("soft lift");

    for dof in state_dofs {
        assert!(
            (hard_full[dof] - 1.0).abs() < 1e-12,
            "hard boundary dof {dof} should equal the prescribed value"
        );
        assert!(
            (soft_full[dof] - hard_full[dof]).abs() < 5e-3,
            "soft boundary dof {dof} should approach the hard value"
        );
    }
}

fn posterior_mean_with_zero_dynamics(
    prior: &feg_infer::prior::spacetime::SpacetimePrior,
    step_matrix: &FeecCsr,
    state_dim: usize,
    add_soft_constraints: bool,
) -> Vec<f64> {
    let mut builder = SpacetimeLinearObservationBuilder::new(3, &prior.slice);
    let left = scale_matrix(&core_triplet_to_feec_csr(&prior.slice.mass), -1.0);
    builder
        .push_transition_block_from_reduced(
            0,
            &left,
            step_matrix,
            &FeecVector::zeros(state_dim),
            &FeecVector::zeros(state_dim),
            OBS_VARIANCE,
        )
        .expect("first transition should stack");
    builder
        .push_transition_block_from_reduced(
            1,
            &left,
            step_matrix,
            &FeecVector::zeros(state_dim),
            &FeecVector::zeros(state_dim),
            OBS_VARIANCE,
        )
        .expect("second transition should stack");
    if add_soft_constraints {
        builder
            .add_soft_boundary_constraints(3, &prior.slice)
            .expect("soft boundary constraints should stack");
    }
    let observations = builder.finish();
    let (posterior_precision, information) = apply_gaussian_observations(
        &prior.precision.to_sparse(),
        &observations.matrix,
        &observations.observations,
        Some(&observations.bias),
        observations.noise_variance,
    );
    let posterior =
        Gmrf::from_information_and_precision(information, posterior_precision).expect("posterior");
    posterior.mean().as_slice().to_vec()
}

fn identity_matrix(dimension: usize) -> FeecCsr {
    let mut coo = FeecCoo::new(dimension, dimension);
    for i in 0..dimension {
        coo.push(i, i, 1.0);
    }
    FeecCsr::from(&coo)
}

fn scale_matrix(matrix: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value * scale);
    }
    FeecCsr::from(&coo)
}

fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
    let mut coo = FeecCoo::from(lhs);
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

fn assert_slice_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (index, (left, right)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (left - right).abs() <= tol,
            "entry {index} differs: {left} vs {right}"
        );
    }
}
