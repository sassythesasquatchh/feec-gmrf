use feg_infer::matern_1form::{build_hodge_laplacian_1form, feec_csr_to_gmrf};
use feg_infer::torus_1form_mass_inverse_isolation::{
    compute_torus_1form_mass_inverse_isolation_report, default_torus_shell_coarse_mesh_path,
};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn matern_torus_mass_inverse_isolation() {
    let mesh_path = default_torus_shell_coarse_mesh_path();
    let mesh_bytes = std::fs::read(mesh_path).expect("failed to read coarse torus mesh");
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let hodge = build_hodge_laplacian_1form(&topology, &metric);
    let report = compute_torus_1form_mass_inverse_isolation_report(
        &topology, &coords, &metric, &hodge, 20.0, 1.0,
    )
    .expect("mass-inverse isolation report should build");

    for strategy in [&report.row_sum, &report.projected, &report.exact_consistent] {
        feec_csr_to_gmrf(&strategy.prior_precision)
            .cholesky_sqrt_lower()
            .expect("precision should factorize");
        assert_eq!(strategy.prior_precision.nrows(), report.edge_lengths.len());
        assert_eq!(strategy.prior_precision.ncols(), report.edge_lengths.len());
        assert_eq!(
            strategy.unconstrained_variances.len(),
            report.edge_lengths.len()
        );
        assert_eq!(
            strategy.harmonic_free_variances.len(),
            report.edge_lengths.len()
        );
        assert!(strategy
            .unconstrained_variances
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert!(strategy
            .harmonic_free_variances
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert!(strategy
            .harmonic_removed_variances
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        assert!(strategy
            .log_harmonic_free_variance_per_length2
            .iter()
            .all(|value| value.is_finite()));
        for i in 0..strategy.unconstrained_variances.len() {
            assert!(
                strategy.harmonic_free_variances[i] <= strategy.unconstrained_variances[i] + 1e-10
            );
        }
    }

    let row_distance = report
        .row_sum
        .distance_to_exact
        .expect("row-sum distance should be populated");
    let projected_distance = report
        .projected
        .distance_to_exact
        .expect("projected distance should be populated");

    assert!(row_distance.is_finite() && row_distance > 0.0);
    assert!(projected_distance.is_finite() && projected_distance > 0.0);
    assert!(projected_distance < row_distance);

    assert!(!approx_eq(
        report.row_sum.h_post_len,
        report.exact_consistent.h_post_len
    ));
    assert!(!approx_eq(
        report.projected.h_post_len,
        report.exact_consistent.h_post_len
    ));
}
