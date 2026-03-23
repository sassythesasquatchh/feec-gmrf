use common::linalg::nalgebra::{bilinear_form_sparse, CooMatrix, CsrMatrix, Vector as NaVector};
use ddf::{cochain::Cochain, ManifoldComplexExt};
use feg_gp::{
    HodgeBranchKind, HodgeBuildOptions, HodgeCompositionalConfig, HodgeCompositionalGp,
    HodgeDecomposedBasis,
};
use formoniq::io::{write_1form_vector_field_vtk, write_cochain_vtk};
use manifold::io::gmsh::gmsh2coord_complex;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !petsc_solver_available() {
        eprintln!("Skipping: PETSc eigen solver binary not available.");
        return Ok(());
    }

    let total_start = Instant::now();
    let out_dir = PathBuf::from("out/matern_1form_torus_gp");
    let _ = fs::remove_dir_all(&out_dir);
    fs::create_dir_all(&out_dir)?;

    let mesh_bytes = fs::read("meshes/torus_shell_resolution_1.msh")?;
    let (topology, coords) = gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);

    let mut build_options = HodgeBuildOptions::new(2);
    build_options.exact_mode_count = 12;
    build_options.coexact_mode_count = 12;
    let basis = HodgeDecomposedBasis::build(&topology, &metric, 1, build_options)?;

    let truth_reduced = build_truth(&basis)?;
    let observation_indices = observation_indices(basis.ambient_dimension(), 12);
    let observation_matrix = selector_matrix(basis.ambient_dimension(), &observation_indices);
    let observations = (&observation_matrix * &truth_reduced)
        .iter()
        .copied()
        .collect::<Vec<_>>();

    let mut config = HodgeCompositionalConfig::default();
    config.exact.mode_count = 12;
    config.coexact.mode_count = 12;
    config.harmonic.mode_count = 2;
    config.exact.kappa = 4.0;
    config.coexact.kappa = 4.0;
    config.harmonic.kappa = 4.0;
    let gp = HodgeCompositionalGp::from_hodge_decomposition(basis.clone(), config)?;

    let exact = gp.condition_branch_linear_observations(
        HodgeBranchKind::Exact,
        &observation_matrix,
        &observations,
        1e-9,
    )?;
    let coexact = gp.condition_branch_linear_observations(
        HodgeBranchKind::Coexact,
        &observation_matrix,
        &observations,
        1e-9,
    )?;
    let harmonic = gp.condition_branch_linear_observations(
        HodgeBranchKind::Harmonic,
        &observation_matrix,
        &observations,
        1e-9,
    )?;
    let combined = gp.condition_linear_observations(&observation_matrix, &observations, 1e-9)?;

    let truth_full = basis.reduced_layout().lift_vector(&truth_reduced)?;
    write_field(&coords, &topology, &out_dir.join("truth"), &truth_full)?;
    write_branch_outputs(
        &coords,
        &topology,
        &basis,
        &out_dir.join("exact"),
        HodgeBranchKind::Exact,
        &exact.mean,
        &exact.variance,
    )?;
    write_branch_outputs(
        &coords,
        &topology,
        &basis,
        &out_dir.join("coexact"),
        HodgeBranchKind::Coexact,
        &coexact.mean,
        &coexact.variance,
    )?;
    write_branch_outputs(
        &coords,
        &topology,
        &basis,
        &out_dir.join("harmonic"),
        HodgeBranchKind::Harmonic,
        &harmonic.mean,
        &harmonic.variance,
    )?;
    write_branch_outputs(
        &coords,
        &topology,
        &basis,
        &out_dir.join("combined"),
        HodgeBranchKind::Harmonic,
        &combined.mean,
        &combined.variance,
    )?;
    write_observation_mask(
        &coords,
        &topology,
        &out_dir.join("observation_mask.vtk"),
        basis.ambient_dimension(),
        &observation_indices,
    )?;
    write_summary(
        &out_dir.join("summary.txt"),
        &topology,
        &basis,
        &observation_matrix,
        &observations,
        &exact.mean,
        &coexact.mean,
        &harmonic.mean,
        &combined.mean,
    )?;

    println!("Wrote outputs to {}", out_dir.display());
    println!("total runtime: {:.3}s", total_start.elapsed().as_secs_f64());
    Ok(())
}

fn build_truth(basis: &HodgeDecomposedBasis) -> Result<NaVector, String> {
    let exact = basis.branch_basis(HodgeBranchKind::Exact);
    let coexact = basis.branch_basis(HodgeBranchKind::Coexact);
    let harmonic = basis.branch_basis(HodgeBranchKind::Harmonic);
    if exact.len() == 0 || coexact.len() == 0 || harmonic.len() < 2 {
        return Err(
            "torus example requires non-empty exact, coexact, and two harmonic modes".into(),
        );
    }

    Ok(column(exact.eigenvectors(), 0).scale(0.8)
        + column(coexact.eigenvectors(), 0).scale(-0.6)
        + column(harmonic.eigenvectors(), 0).scale(0.75)
        + column(harmonic.eigenvectors(), 1).scale(-0.5))
}

fn selector_matrix(dimension: usize, indices: &[usize]) -> CsrMatrix {
    let mut selector = CooMatrix::new(indices.len(), dimension);
    for (row, &index) in indices.iter().enumerate() {
        selector.push(row, index, 1.0);
    }
    CsrMatrix::from(&selector)
}

fn observation_indices(dimension: usize, count: usize) -> Vec<usize> {
    let step = (dimension / count.max(1)).max(1);
    (0..dimension).step_by(step).take(count).collect()
}

fn column(matrix: &faer::Mat<f64>, index: usize) -> NaVector {
    NaVector::from_iterator(
        matrix.nrows(),
        (0..matrix.nrows()).map(|row| matrix[(row, index)]),
    )
}

fn write_field(
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    path_prefix: &std::path::Path,
    values: &NaVector,
) -> Result<(), Box<dyn std::error::Error>> {
    let cochain = Cochain::new(1, values.clone());
    write_cochain_vtk(
        path_prefix.with_extension("vtk"),
        coords,
        topology,
        &cochain,
        "field",
    )?;
    write_1form_vector_field_vtk(
        path_prefix.with_file_name(format!(
            "{}_vector_field.vtk",
            path_prefix.file_name().unwrap().to_string_lossy()
        )),
        coords,
        topology,
        &cochain,
        "field_vector",
    )?;
    Ok(())
}

fn write_branch_outputs(
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    basis: &HodgeDecomposedBasis,
    branch_dir: &std::path::Path,
    _kind: HodgeBranchKind,
    reduced_mean: &[f64],
    reduced_variance: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(branch_dir)?;
    let lifted_mean = basis
        .reduced_layout()
        .lift_vector(&NaVector::from_vec(reduced_mean.to_vec()))?;
    let lifted_variance = basis
        .reduced_layout()
        .lift_vector(&NaVector::from_vec(reduced_variance.to_vec()))?;
    write_field(
        coords,
        topology,
        &branch_dir.join("posterior_mean"),
        &lifted_mean,
    )?;
    let variance_cochain = Cochain::new(1, lifted_variance);
    write_cochain_vtk(
        branch_dir.join("posterior_variance.vtk"),
        coords,
        topology,
        &variance_cochain,
        "posterior_variance",
    )?;
    Ok(())
}

fn write_observation_mask(
    coords: &manifold::geometry::coord::mesh::MeshCoords,
    topology: &manifold::topology::complex::Complex,
    path: &std::path::Path,
    dimension: usize,
    indices: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mask = NaVector::zeros(dimension);
    for &index in indices {
        mask[index] = 1.0;
    }
    let cochain = Cochain::new(1, mask);
    write_cochain_vtk(path, coords, topology, &cochain, "observation_mask")?;
    Ok(())
}

fn write_summary(
    path: &std::path::Path,
    topology: &manifold::topology::complex::Complex,
    basis: &HodgeDecomposedBasis,
    observation_matrix: &CsrMatrix,
    observations: &[f64],
    exact_mean: &[f64],
    coexact_mean: &[f64],
    harmonic_mean: &[f64],
    combined_mean: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "ambient_dimension={}", basis.ambient_dimension())?;
    writeln!(
        writer,
        "exact_modes={} coexact_modes={} harmonic_modes={}",
        basis.branch_basis(HodgeBranchKind::Exact).len(),
        basis.branch_basis(HodgeBranchKind::Coexact).len(),
        basis.branch_basis(HodgeBranchKind::Harmonic).len(),
    )?;
    writeln!(
        writer,
        "exact_curl_relative={}",
        relative_curl_residual(topology, exact_mean)
    )?;
    writeln!(
        writer,
        "coexact_coclosed_relative={}",
        relative_coclosed_residual(topology, basis.reduced_mass(), coexact_mean)
    )?;
    writeln!(
        writer,
        "harmonic_projection_error={}",
        harmonic_projection_error(
            basis.branch_basis(HodgeBranchKind::Harmonic),
            basis.reduced_mass(),
            harmonic_mean,
        )
    )?;
    writeln!(
        writer,
        "combined_observation_residual={}",
        observation_residual_norm(observation_matrix, observations, combined_mean)
    )?;
    Ok(())
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
    let weighted = mass * &mean;
    let residual = d0.transpose() * weighted;
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
        let basis_vec = column(branch.eigenvectors(), col);
        let coeff = bilinear_form_sparse(mass, &basis_vec, &mean);
        reconstructed += basis_vec.scale(coeff);
    }
    let norm = mean.norm().max(1e-12);
    (&reconstructed - mean).norm() / norm
}

fn observation_residual_norm(
    observation_matrix: &CsrMatrix,
    observations: &[f64],
    mean: &[f64],
) -> f64 {
    let predicted = observation_matrix * NaVector::from_vec(mean.to_vec());
    (predicted - NaVector::from_vec(observations.to_vec())).norm()
}

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
