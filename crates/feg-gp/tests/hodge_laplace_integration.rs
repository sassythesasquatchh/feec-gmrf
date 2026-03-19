use feg_gp::{SpectralMaternConfig, SpectralMaternGp};
use manifold::gen::cartesian::CartesianMeshInfo;
use std::path::PathBuf;

fn petsc_solver_available() -> bool {
    if let Ok(path) = std::env::var("PETSC_SOLVER_PATH") {
        let candidate = PathBuf::from(path).join("ghiep.out");
        return candidate.exists();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../feec/petsc-solver/ghiep.out")
        .exists()
}

#[test]
fn hodge_laplace_largest_eigenpairs_drive_gp() {
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return;
    }

    let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let config = SpectralMaternConfig {
        k: 3,
        ..Default::default()
    };

    let gp = SpectralMaternGp::from_hodge_laplace(&topology, &metric, 0, config)
        .expect("spectral GP should build");
    let cov = gp.covariance_matrix();

    assert_eq!(cov.nrows(), cov.ncols());
    assert_eq!(cov.nrows(), gp.eigenvectors().nrows());
    assert_eq!(gp.k(), 3);
}
