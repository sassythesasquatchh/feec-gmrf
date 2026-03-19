use feg_infer::diagnostics::{
    build_analytic_torus_harmonic_basis, build_harmonic_orthogonality_constraints,
    compute_matern_1form_torus_residual_study_report, LinearEqualityConstraints,
    Matern1FormTorusResidualStudyReport,
};
use feg_infer::matern_1form::{
    build_hodge_laplacian_1form, feec_csr_to_gmrf, MaternConfig, MaternMassInverse,
};
use gmrf_core::types::Vector as GmrfVector;
use std::collections::HashMap;
use std::path::PathBuf;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 * (1.0 + a.abs().max(b.abs()))
}

fn assert_matrix_is_symmetric_and_finite(mat: &common::linalg::nalgebra::CsrMatrix) {
    let mut entries = HashMap::new();
    for (row, col, value) in mat.triplet_iter() {
        assert!(value.is_finite());
        *entries.entry((row, col)).or_insert(0.0) += *value;
    }

    for ((row, col), value) in &entries {
        let transpose = entries.get(&(*col, *row)).copied().unwrap_or(0.0);
        assert!(approx_eq(*value, transpose));
    }
}

fn assert_vector_is_finite(vec: &common::linalg::nalgebra::Vector) {
    assert!(vec.iter().all(|value| value.is_finite()));
}

fn max_abs_diff(
    lhs: &common::linalg::nalgebra::Vector,
    rhs: &common::linalg::nalgebra::Vector,
) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    (0..lhs.len())
        .map(|i| (lhs[i] - rhs[i]).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_contributions_sum(report: &Matern1FormTorusResidualStudyReport) {
    for batch in &report.batch_contributions {
        let sum = batch.position_even_fourier
            + batch.direction_legendre
            + batch.interaction_even
            + batch.discrete_surrogates
            + batch.unexplained;
        assert!(
            (sum - batch.total_post_length).abs() <= 1e-8 * (1.0 + batch.total_post_length.abs())
        );
    }

    let summary = &report.contribution_summary;
    let mean_sum = summary.position_even_fourier.absolute.mean
        + summary.direction_legendre.absolute.mean
        + summary.interaction_even.absolute.mean
        + summary.discrete_surrogates.absolute.mean
        + summary.unexplained.absolute.mean;
    assert!(
        (mean_sum - summary.total_post_length.mean).abs()
            <= 1e-8 * (1.0 + summary.total_post_length.mean.abs())
    );
}

fn assert_report_is_consistent(report: &Matern1FormTorusResidualStudyReport, ndofs: usize) {
    assert_eq!(report.prior_precision.nrows(), ndofs);
    assert_eq!(report.prior_precision.ncols(), ndofs);
    assert_eq!(report.forcing_scale_diag.len(), ndofs);
    assert_eq!(report.edge_diagnostics.variances.len(), ndofs);
    assert_eq!(report.harmonic_free_edge_diagnostics.variances.len(), ndofs);
    assert_eq!(
        report.harmonic_removed_edge_diagnostics.variances.len(),
        ndofs
    );
    assert_eq!(report.geometry.midpoint_rho.len(), ndofs);
    assert_eq!(
        report
            .field_decomposition
            .log_harmonic_free_variance_per_length2
            .len(),
        ndofs
    );

    assert_vector_is_finite(&report.forcing_scale_diag);
    assert_vector_is_finite(&report.edge_diagnostics.variances);
    assert_vector_is_finite(&report.harmonic_free_edge_diagnostics.variances);
    assert_vector_is_finite(&report.harmonic_removed_edge_diagnostics.variances);
    assert_vector_is_finite(&report.geometry.midpoint_rho);
    assert_vector_is_finite(&report.geometry.midpoint_theta);
    assert_vector_is_finite(&report.geometry.gaussian_curvature);
    assert_vector_is_finite(&report.geometry.toroidal_alignment_sq);
    assert_vector_is_finite(
        &report
            .field_decomposition
            .log_harmonic_free_variance_per_length2,
    );
    assert_vector_is_finite(&report.field_decomposition.position_even_fourier_component);
    assert_vector_is_finite(&report.field_decomposition.direction_legendre_component);
    assert_vector_is_finite(&report.field_decomposition.interaction_even_component);
    assert_vector_is_finite(&report.field_decomposition.discrete_surrogate_component);
    assert_vector_is_finite(&report.field_decomposition.unexplained_residual);

    assert!(report
        .geometry
        .toroidal_alignment_sq
        .iter()
        .all(|value| *value >= 0.0 && *value <= 1.0 + 1e-12));
    assert!(report
        .harmonic_removed_edge_diagnostics
        .variances
        .iter()
        .all(|value| *value >= 0.0));
    assert_contributions_sum(report);
}

#[test]
fn torus_residual_study_reports_are_valid_stable_and_distinct() {
    let mesh_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell.msh");
    let mesh_bytes = std::fs::read(mesh_path).expect("Failed to read torus mesh");
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let harmonic_basis = build_analytic_torus_harmonic_basis(&topology, &coords, &metric)
        .expect("analytic harmonic basis should build");
    let harmonic_constraint_matrix =
        build_harmonic_orthogonality_constraints(&harmonic_basis, &hodge.mass_u)
            .expect("harmonic constraints should build");
    let harmonic_constraint_rhs = GmrfVector::zeros(harmonic_constraint_matrix.nrows());
    let constraints = LinearEqualityConstraints {
        matrix: &harmonic_constraint_matrix,
        rhs: &harmonic_constraint_rhs,
    };
    let ndofs = hodge.mass_u.nrows();

    let row_sum = compute_matern_1form_torus_residual_study_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 20.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
        64,
        13,
        4,
        constraints,
    )
    .expect("row-sum report should build");
    let row_sum_repeat = compute_matern_1form_torus_residual_study_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 20.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
        64,
        13,
        4,
        constraints,
    )
    .expect("repeated row-sum report should build");
    let row_sum_alt_seed = compute_matern_1form_torus_residual_study_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 20.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
        64,
        97,
        4,
        constraints,
    )
    .expect("alternate-seed row-sum report should build");
    let row_sum_more_probes = compute_matern_1form_torus_residual_study_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 20.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::RowSumLumped,
        },
        256,
        13,
        4,
        constraints,
    )
    .expect("higher-probe row-sum report should build");
    let projected = compute_matern_1form_torus_residual_study_report(
        &topology,
        &coords,
        &metric,
        &hodge,
        MaternConfig {
            kappa: 20.0,
            tau: 1.0,
            mass_inverse: MaternMassInverse::Nc1ProjectedSparseInverse,
        },
        64,
        13,
        4,
        constraints,
    )
    .expect("projected report should build");

    assert_report_is_consistent(&row_sum, ndofs);
    assert_report_is_consistent(&row_sum_repeat, ndofs);
    assert_report_is_consistent(&row_sum_alt_seed, ndofs);
    assert_report_is_consistent(&row_sum_more_probes, ndofs);
    assert_report_is_consistent(&projected, ndofs);

    assert_matrix_is_symmetric_and_finite(&row_sum.prior_precision);
    assert_matrix_is_symmetric_and_finite(&projected.prior_precision);
    feec_csr_to_gmrf(&row_sum.prior_precision)
        .cholesky_sqrt_lower()
        .expect("row-sum precision should factorize");
    feec_csr_to_gmrf(&projected.prior_precision)
        .cholesky_sqrt_lower()
        .expect("projected precision should factorize");

    assert!(
        max_abs_diff(
            &row_sum
                .field_decomposition
                .log_harmonic_free_variance_per_length2,
            &projected
                .field_decomposition
                .log_harmonic_free_variance_per_length2,
        ) > 1e-9
    );
    assert!(
        max_abs_diff(
            &row_sum.field_decomposition.unexplained_residual,
            &projected.field_decomposition.unexplained_residual,
        ) > 1e-9
    );

    assert!(approx_eq(
        max_abs_diff(
            &row_sum
                .field_decomposition
                .log_harmonic_free_variance_per_length2,
            &row_sum_repeat
                .field_decomposition
                .log_harmonic_free_variance_per_length2,
        ),
        0.0
    ));
    assert!(approx_eq(
        max_abs_diff(
            &row_sum.field_decomposition.unexplained_residual,
            &row_sum_repeat.field_decomposition.unexplained_residual,
        ),
        0.0
    ));

    let total_diff = (row_sum.contribution_summary.total_post_length.mean
        - row_sum_alt_seed.contribution_summary.total_post_length.mean)
        .abs();
    let total_se = row_sum
        .contribution_summary
        .total_post_length
        .standard_error
        .max(
            row_sum_alt_seed
                .contribution_summary
                .total_post_length
                .standard_error,
        );
    assert!(
        total_diff <= 6.0 * total_se.max(1e-12),
        "alternate seeds should agree within RBMC uncertainty"
    );

    assert!(
        row_sum_more_probes
            .contribution_summary
            .total_post_length
            .standard_error
            < row_sum
                .contribution_summary
                .total_post_length
                .standard_error
    );
}
