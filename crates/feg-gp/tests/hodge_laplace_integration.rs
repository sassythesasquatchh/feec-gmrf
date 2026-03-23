use common::linalg::nalgebra::{bilinear_form_sparse, CooMatrix, CsrMatrix, Vector as NaVector};
use ddf::ManifoldComplexExt;
use feg_gp::{
    BoundaryConditionSpec, HodgeBranchKind, HodgeBuildOptions, HodgeCompositionalConfig,
    HodgeCompositionalGp, HodgeDecomposedBasis, ReducedFormLayout,
};
use formoniq::assemble;
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef, io::gmsh::gmsh2coord_complex,
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::{Mutex, MutexGuard, OnceLock},
};

static PETSC_TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn petsc_solver_available() -> bool {
    if let Ok(path) = std::env::var("PETSC_SOLVER_PATH") {
        if !path.is_empty() {
            let candidate = PathBuf::from(path).join("ghiep.out");
            return candidate.exists();
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .map(|ancestor| ancestor.join("feec/petsc-solver/ghiep.out"))
        .any(|candidate| candidate.exists())
}

fn petsc_test_lock() -> MutexGuard<'static, ()> {
    PETSC_TEST_MUTEX
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("PETSc test mutex should not be poisoned")
}

fn selector_matrix(dimension: usize, indices: &[usize]) -> CsrMatrix {
    let mut selector = CooMatrix::new(indices.len(), dimension);
    for (row, &index) in indices.iter().enumerate() {
        selector.push(row, index, 1.0);
    }
    CsrMatrix::from(&selector)
}

fn mat_column_to_vector(matrix: &faer::Mat<f64>, column: usize) -> NaVector {
    NaVector::from_iterator(
        matrix.nrows(),
        (0..matrix.nrows()).map(|row| matrix[(row, column)]),
    )
}

fn max_branch_cross_inner_product(
    mass: &CsrMatrix,
    left: &feg_gp::HodgeBranchBasis,
    right: &feg_gp::HodgeBranchBasis,
) -> f64 {
    let mut maximum: f64 = 0.0;
    for left_col in 0..left.len() {
        let left_vec = mat_column_to_vector(left.eigenvectors(), left_col);
        for right_col in 0..right.len() {
            let right_vec = mat_column_to_vector(right.eigenvectors(), right_col);
            maximum = maximum.max(bilinear_form_sparse(mass, &left_vec, &right_vec).abs());
        }
    }
    maximum
}

fn max_within_branch_orthonormality_error(
    mass: &CsrMatrix,
    branch: &feg_gp::HodgeBranchBasis,
) -> f64 {
    let mut maximum: f64 = 0.0;
    for left_col in 0..branch.len() {
        let left_vec = mat_column_to_vector(branch.eigenvectors(), left_col);
        for right_col in 0..branch.len() {
            let right_vec = mat_column_to_vector(branch.eigenvectors(), right_col);
            let target = if left_col == right_col { 1.0 } else { 0.0 };
            maximum =
                maximum.max((bilinear_form_sparse(mass, &left_vec, &right_vec) - target).abs());
        }
    }
    maximum
}

fn relative_curl_residual(topology: &manifold::topology::complex::Complex, mean: &[f64]) -> f64 {
    let mean = NaVector::from_vec(mean.to_vec());
    let d1 = CsrMatrix::from(&topology.exterior_derivative_operator(1));
    let residual = &d1 * &mean;
    residual.norm() / mean.norm().max(1e-12)
}

fn relative_coclosed_residual(
    topology: &manifold::topology::complex::Complex,
    mass: &CsrMatrix,
    mean: &[f64],
) -> f64 {
    let mean = NaVector::from_vec(mean.to_vec());
    let d0 = CsrMatrix::from(&topology.exterior_derivative_operator(0));
    let weighted_mean = mass * &mean;
    let residual = d0.transpose() * weighted_mean;
    residual.norm() / mean.norm().max(1e-12)
}

fn harmonic_projection_error(
    branch: &feg_gp::HodgeBranchBasis,
    mass: &CsrMatrix,
    mean: &[f64],
) -> f64 {
    let mean = NaVector::from_vec(mean.to_vec());
    let mut reconstructed = NaVector::zeros(mean.len());
    for col in 0..branch.len() {
        let basis_vec = mat_column_to_vector(branch.eigenvectors(), col);
        let coefficient = bilinear_form_sparse(mass, &basis_vec, &mean);
        reconstructed += basis_vec.scale(coefficient);
    }
    let mean_norm = mean.norm().max(1e-12);
    (&reconstructed - mean).norm() / mean_norm
}

#[test]
fn hodge_decomposition_on_torus_respects_branch_biases() {
    let _guard = petsc_test_lock();
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return;
    }

    let mesh_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../meshes/torus_shell_resolution_1.msh");
    let mesh_bytes = fs::read(mesh_path).expect("torus mesh should be readable");
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let mut build_options = HodgeBuildOptions::new(2);
    build_options.exact_mode_count = 12;
    build_options.coexact_mode_count = 12;
    let basis = HodgeDecomposedBasis::build(&topology, &metric, 1, build_options)
        .expect("closed torus Hodge basis should build");

    assert_eq!(
        basis.reduced_layout().full_dimension(),
        basis.reduced_layout().reduced_dimension()
    );
    assert!(basis.branch_basis(HodgeBranchKind::Exact).len() > 0);
    assert!(basis.branch_basis(HodgeBranchKind::Coexact).len() > 0);
    assert_eq!(basis.branch_basis(HodgeBranchKind::Harmonic).len(), 2);

    let mass = basis.reduced_mass();
    assert!(
        max_within_branch_orthonormality_error(mass, basis.branch_basis(HodgeBranchKind::Exact))
            <= 1e-8
    );
    assert!(
        max_within_branch_orthonormality_error(mass, basis.branch_basis(HodgeBranchKind::Coexact))
            <= 1e-8
    );
    assert!(
        max_within_branch_orthonormality_error(mass, basis.branch_basis(HodgeBranchKind::Harmonic))
            <= 1e-8
    );
    assert!(
        max_branch_cross_inner_product(
            mass,
            basis.branch_basis(HodgeBranchKind::Exact),
            basis.branch_basis(HodgeBranchKind::Coexact),
        ) <= 1e-8
    );
    assert!(
        max_branch_cross_inner_product(
            mass,
            basis.branch_basis(HodgeBranchKind::Exact),
            basis.branch_basis(HodgeBranchKind::Harmonic),
        ) <= 1e-8
    );
    assert!(
        max_branch_cross_inner_product(
            mass,
            basis.branch_basis(HodgeBranchKind::Coexact),
            basis.branch_basis(HodgeBranchKind::Harmonic),
        ) <= 1e-8
    );

    let exact_basis = basis.branch_basis(HodgeBranchKind::Exact);
    let coexact_basis = basis.branch_basis(HodgeBranchKind::Coexact);
    let harmonic_basis = basis.branch_basis(HodgeBranchKind::Harmonic);

    let truth = mat_column_to_vector(exact_basis.eigenvectors(), 0).scale(0.8)
        + mat_column_to_vector(coexact_basis.eigenvectors(), 0).scale(-0.6)
        + mat_column_to_vector(harmonic_basis.eigenvectors(), 0).scale(0.75)
        + mat_column_to_vector(harmonic_basis.eigenvectors(), 1).scale(-0.5);

    let ambient_dimension = basis.ambient_dimension();
    let observation_indices = (0..ambient_dimension)
        .step_by((ambient_dimension / 12).max(1))
        .take(12)
        .collect::<Vec<_>>();
    let observation_matrix = selector_matrix(ambient_dimension, &observation_indices);
    let observations = (&observation_matrix * &truth)
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let config = HodgeCompositionalConfig::default();
    let gp = HodgeCompositionalGp::from_hodge_decomposition(basis.clone(), config)
        .expect("compositional GP should build");

    let exact = gp
        .condition_branch_linear_observations(
            HodgeBranchKind::Exact,
            &observation_matrix,
            &observations,
            1e-9,
        )
        .expect("exact branch conditioning should succeed");
    let coexact = gp
        .condition_branch_linear_observations(
            HodgeBranchKind::Coexact,
            &observation_matrix,
            &observations,
            1e-9,
        )
        .expect("coexact branch conditioning should succeed");
    let harmonic = gp
        .condition_branch_linear_observations(
            HodgeBranchKind::Harmonic,
            &observation_matrix,
            &observations,
            1e-9,
        )
        .expect("harmonic branch conditioning should succeed");
    let combined = gp
        .condition_linear_observations(&observation_matrix, &observations, 1e-9)
        .expect("combined conditioning should succeed");

    assert!(relative_curl_residual(&topology, &exact.mean) <= 1e-8);
    assert!(relative_coclosed_residual(&topology, mass, &coexact.mean) <= 1e-6);
    assert!(harmonic_projection_error(harmonic_basis, mass, &harmonic.mean) <= 1e-8);

    let combined_residual = ((&observation_matrix * NaVector::from_vec(combined.mean.clone()))
        - NaVector::from_vec(observations.clone()))
    .norm();
    let exact_residual = ((&observation_matrix * NaVector::from_vec(exact.mean.clone()))
        - NaVector::from_vec(observations.clone()))
    .norm();
    let coexact_residual = ((&observation_matrix * NaVector::from_vec(coexact.mean.clone()))
        - NaVector::from_vec(observations.clone()))
    .norm();
    let harmonic_residual = ((&observation_matrix * NaVector::from_vec(harmonic.mean.clone()))
        - NaVector::from_vec(observations.clone()))
    .norm();

    assert!(combined_residual <= exact_residual + 1e-10);
    assert!(combined_residual <= coexact_residual + 1e-10);
    assert!(combined_residual <= harmonic_residual + 1e-10);
}

#[test]
fn boundary_hodge_build_and_conditioning_use_reduced_layouts() {
    let _guard = petsc_test_lock();
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return;
    }

    let mesh = CartesianMeshInfo::new_unit_scaled(3, 2, 1.0);
    let (topology, coords) = mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let strong_boundary = |point: CoordRef| point[0] == 0.0 || point[1] == 0.0 || point[2] == 0.0;
    let strong_vertices =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 0, strong_boundary)
            .into_iter()
            .collect::<BTreeSet<_>>();
    let strong_edges =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 1, strong_boundary)
            .into_iter()
            .collect::<BTreeSet<_>>();
    let strong_faces =
        assemble::boundary_simplices_where_barycenter(&topology, &coords, 2, strong_boundary)
            .into_iter()
            .collect::<BTreeSet<_>>();

    let boundary = BoundaryConditionSpec::new()
        .with_strong_dofs(0, strong_vertices.iter().copied())
        .with_strong_dofs(1, strong_edges.iter().copied())
        .with_strong_dofs(2, strong_faces.iter().copied());

    let mut build_options = HodgeBuildOptions::new(0);
    build_options.exact_mode_count = 4;
    build_options.coexact_mode_count = 4;
    build_options.boundary = Some(boundary.clone());

    let basis = HodgeDecomposedBasis::build(&topology, &metric, 1, build_options)
        .expect("boundary-aware Hodge basis should build");

    assert!(basis.reduced_layout().reduced_dimension() < basis.reduced_layout().full_dimension());
    assert_eq!(basis.branch_basis(HodgeBranchKind::Harmonic).len(), 0);

    let reduced = NaVector::from_iterator(
        basis.reduced_layout().reduced_dimension(),
        (0..basis.reduced_layout().reduced_dimension()).map(|index| index as f64 + 1.0),
    );
    let lifted = basis
        .reduced_layout()
        .lift_vector(&reduced)
        .expect("lifting a reduced vector should succeed");
    for &edge in &strong_edges {
        assert_eq!(lifted[edge], 0.0);
    }

    let observation_indices = (0..basis.ambient_dimension()).take(4).collect::<Vec<_>>();
    let observation_matrix = selector_matrix(basis.ambient_dimension(), &observation_indices);
    let observations = vec![0.0; observation_indices.len()];
    let gp = HodgeCompositionalGp::from_hodge_decomposition(
        basis.clone(),
        HodgeCompositionalConfig::default(),
    )
    .expect("boundary-aware GP should build");
    let conditioned = gp
        .condition_linear_observations(&observation_matrix, &observations, 1e-6)
        .expect("boundary-aware conditioning should succeed");

    assert_eq!(conditioned.mean.len(), basis.ambient_dimension());
    assert_eq!(conditioned.variance.len(), basis.ambient_dimension());
    assert!(conditioned.mean.iter().all(|value| value.is_finite()));
    assert!(conditioned.variance.iter().all(|value| value.is_finite()));
}

#[test]
fn reduced_form_layout_restricts_and_lifts_consistently() {
    let strong_dofs = [1_usize, 3_usize].into_iter().collect::<BTreeSet<_>>();
    let layout =
        ReducedFormLayout::from_strong_dofs(5, Some(&strong_dofs)).expect("layout should build");

    let full = NaVector::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let reduced = layout
        .reduce_vector(&full)
        .expect("reduction should succeed");
    assert_eq!(reduced.as_slice(), &[10.0, 30.0, 50.0]);

    let lifted = layout
        .lift_vector(&reduced)
        .expect("lifting should succeed");
    assert_eq!(lifted.as_slice(), &[10.0, 0.0, 30.0, 0.0, 50.0]);

    let mut operator = CooMatrix::new(2, 5);
    operator.push(0, 0, 1.0);
    operator.push(0, 1, 2.0);
    operator.push(0, 4, 3.0);
    operator.push(1, 2, -1.0);
    operator.push(1, 3, 4.0);
    let restricted = layout
        .restrict_columns(&CsrMatrix::from(&operator))
        .expect("column restriction should succeed");

    assert_eq!(restricted.nrows(), 2);
    assert_eq!(restricted.ncols(), 3);
    let reduced_result = &restricted * reduced;
    assert_eq!(reduced_result[0], 160.0);
    assert_eq!(reduced_result[1], -30.0);
}
