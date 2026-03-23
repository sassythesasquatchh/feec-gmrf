use crate::{
    condition_low_rank_factors, condition_low_rank_factors_with_covariance,
    diagonal_from_feature_matrix, full_covariance_from_feature_matrix, weighted_feature_matrix,
    ConditionedFullGp, ConditionedGp, GpError, DEFAULT_K,
};
use common::linalg::faer::FaerCholesky;
use common::linalg::nalgebra::{
    bilinear_form_sparse, CooMatrix, CsrMatrix, Matrix as NaMatrix, Vector as NaVector,
};
use common::linalg::petsc::{
    petsc_ghep_reduced_with_which, petsc_ghiep, GhiepReducedSolve, GhiepWhich,
};
use ddf::ManifoldComplexExt;
use exterior::ExteriorGrade;
use faer::Mat;
use formoniq::{
    assemble,
    operators::HodgeMassElmat,
    problems::{hodge_laplace, laplace_beltrami},
};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    panic::{catch_unwind, AssertUnwindSafe},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HodgeBranchKind {
    Exact,
    Coexact,
    Harmonic,
}

impl HodgeBranchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Coexact => "coexact",
            Self::Harmonic => "harmonic",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct BoundaryConditionSpec {
    strong_dofs_by_grade: BTreeMap<ExteriorGrade, BTreeSet<usize>>,
}

impl BoundaryConditionSpec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_strong_dofs(
        mut self,
        grade: ExteriorGrade,
        dofs: impl IntoIterator<Item = usize>,
    ) -> Self {
        self.insert_strong_dofs(grade, dofs);
        self
    }

    pub fn insert_strong_dofs(
        &mut self,
        grade: ExteriorGrade,
        dofs: impl IntoIterator<Item = usize>,
    ) {
        self.strong_dofs_by_grade
            .entry(grade)
            .or_default()
            .extend(dofs);
    }

    pub fn strong_dofs(&self, grade: ExteriorGrade) -> Option<&BTreeSet<usize>> {
        self.strong_dofs_by_grade.get(&grade)
    }

    pub fn contains(&self, grade: ExteriorGrade, dof: usize) -> bool {
        self.strong_dofs(grade)
            .is_some_and(|strong_dofs| strong_dofs.contains(&dof))
    }
}

#[derive(Debug, Clone)]
pub struct ReducedFormLayout {
    full_dimension: usize,
    reduced_dimension: usize,
    kept_dofs: Vec<usize>,
    full_to_reduced: Vec<Option<usize>>,
}

impl ReducedFormLayout {
    pub fn from_strong_dofs(
        full_dimension: usize,
        strong_dofs: Option<&BTreeSet<usize>>,
    ) -> Result<Self, GpError> {
        let mut full_to_reduced = vec![None; full_dimension];
        let mut kept_dofs = Vec::with_capacity(full_dimension);

        if let Some(strong_dofs) = strong_dofs {
            for &index in strong_dofs {
                if index >= full_dimension {
                    return Err(GpError::InvalidStrongDofIndex {
                        index,
                        dimension: full_dimension,
                    });
                }
            }
        }

        for full_index in 0..full_dimension {
            if strong_dofs.is_some_and(|strong_dofs| strong_dofs.contains(&full_index)) {
                continue;
            }
            full_to_reduced[full_index] = Some(kept_dofs.len());
            kept_dofs.push(full_index);
        }

        Ok(Self {
            full_dimension,
            reduced_dimension: kept_dofs.len(),
            kept_dofs,
            full_to_reduced,
        })
    }

    pub fn full_dimension(&self) -> usize {
        self.full_dimension
    }

    pub fn reduced_dimension(&self) -> usize {
        self.reduced_dimension
    }

    pub fn kept_dofs(&self) -> &[usize] {
        &self.kept_dofs
    }

    pub fn reduce_vector(&self, full: &NaVector) -> Result<NaVector, GpError> {
        if full.len() != self.full_dimension {
            return Err(GpError::FullDimensionMismatch {
                expected: self.full_dimension,
                got: full.len(),
            });
        }
        Ok(NaVector::from_iterator(
            self.reduced_dimension,
            self.kept_dofs.iter().map(|&index| full[index]),
        ))
    }

    pub fn lift_vector(&self, reduced: &NaVector) -> Result<NaVector, GpError> {
        if reduced.len() != self.reduced_dimension {
            return Err(GpError::ReducedDimensionMismatch {
                expected: self.reduced_dimension,
                got: reduced.len(),
            });
        }

        let mut full = NaVector::zeros(self.full_dimension);
        for (reduced_index, &full_index) in self.kept_dofs.iter().enumerate() {
            full[full_index] = reduced[reduced_index];
        }
        Ok(full)
    }

    pub fn restrict_rows(&self, operator: &CsrMatrix) -> Result<CsrMatrix, GpError> {
        let identity = Self::identity(operator.ncols());
        Self::restrict_operator(self, &identity, operator)
    }

    pub fn restrict_columns(&self, operator: &CsrMatrix) -> Result<CsrMatrix, GpError> {
        let identity = Self::identity(operator.nrows());
        Self::restrict_operator(&identity, self, operator)
    }

    pub fn restrict_operator(
        row_layout: &Self,
        col_layout: &Self,
        operator: &CsrMatrix,
    ) -> Result<CsrMatrix, GpError> {
        if operator.nrows() != row_layout.full_dimension {
            return Err(GpError::FullDimensionMismatch {
                expected: row_layout.full_dimension,
                got: operator.nrows(),
            });
        }
        if operator.ncols() != col_layout.full_dimension {
            return Err(GpError::FullDimensionMismatch {
                expected: col_layout.full_dimension,
                got: operator.ncols(),
            });
        }

        let mut reduced =
            CooMatrix::new(row_layout.reduced_dimension, col_layout.reduced_dimension);
        for (row, col, value) in operator.triplet_iter() {
            let Some(reduced_row) = row_layout.full_to_reduced[row] else {
                continue;
            };
            let Some(reduced_col) = col_layout.full_to_reduced[col] else {
                continue;
            };
            reduced.push(reduced_row, reduced_col, *value);
        }
        Ok(CsrMatrix::from(&reduced))
    }

    fn identity(dimension: usize) -> Self {
        Self {
            full_dimension: dimension,
            reduced_dimension: dimension,
            kept_dofs: (0..dimension).collect(),
            full_to_reduced: (0..dimension).map(Some).collect(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HodgeEigensolverOptions {
    pub mass_solve: GhiepReducedSolve,
    pub oversampling: usize,
}

impl Default for HodgeEigensolverOptions {
    fn default() -> Self {
        Self {
            mass_solve: GhiepReducedSolve::Direct,
            oversampling: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeBuildOptions {
    pub harmonic_dim: usize,
    pub exact_mode_count: usize,
    pub coexact_mode_count: usize,
    pub eigenvalue_zero_tolerance: f64,
    pub eigensolver: HodgeEigensolverOptions,
    pub boundary: Option<BoundaryConditionSpec>,
}

impl HodgeBuildOptions {
    pub fn new(harmonic_dim: usize) -> Self {
        Self {
            harmonic_dim,
            exact_mode_count: DEFAULT_K,
            coexact_mode_count: DEFAULT_K,
            eigenvalue_zero_tolerance: 1e-10,
            eigensolver: HodgeEigensolverOptions::default(),
            boundary: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeBranchBasis {
    eigenvalues: Vec<f64>,
    eigenvectors: Mat<f64>,
}

impl HodgeBranchBasis {
    pub fn empty(ambient_dimension: usize) -> Self {
        Self {
            eigenvalues: Vec::new(),
            eigenvectors: Mat::zeros(ambient_dimension, 0),
        }
    }

    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    pub fn ambient_dimension(&self) -> usize {
        self.eigenvectors.nrows()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &Mat<f64> {
        &self.eigenvectors
    }

    fn truncate_smallest(&self, mode_count: usize) -> Self {
        let keep = mode_count.min(self.len());
        let mut truncated = Mat::zeros(self.ambient_dimension(), keep);
        for col in 0..keep {
            for row in 0..self.ambient_dimension() {
                truncated[(row, col)] = self.eigenvectors[(row, col)];
            }
        }
        Self {
            eigenvalues: self.eigenvalues[..keep].to_vec(),
            eigenvectors: truncated,
        }
    }

    fn feature_matrix(&self, alpha: f64, config: HodgeBranchConfig) -> Result<Mat<f64>, GpError> {
        validate_alpha(alpha)?;
        validate_branch_config(config)?;
        if config.mode_count == 0 || self.is_empty() {
            return Ok(Mat::zeros(self.ambient_dimension(), 0));
        }

        let truncated = self.truncate_smallest(config.mode_count);
        let mut weights = Vec::with_capacity(truncated.len());
        for &eigenvalue in &truncated.eigenvalues {
            weights.push(branch_matern_weight(eigenvalue, alpha, config)?);
        }
        Ok(weighted_feature_matrix(&truncated.eigenvectors, &weights))
    }
}

#[derive(Debug, Clone)]
pub struct HodgeDecomposedBasis {
    grade: ExteriorGrade,
    reduced_layout: ReducedFormLayout,
    reduced_mass: CsrMatrix,
    exact: HodgeBranchBasis,
    coexact: HodgeBranchBasis,
    harmonic: HodgeBranchBasis,
    boundary: Option<BoundaryConditionSpec>,
}

impl HodgeDecomposedBasis {
    pub fn build(
        topology: &Complex,
        geometry: &MeshLengths,
        grade: ExteriorGrade,
        options: HodgeBuildOptions,
    ) -> Result<Self, GpError> {
        let boundary = options.boundary.clone();
        let reduced_layout = ReducedFormLayout::from_strong_dofs(
            topology.nsimplices(grade),
            strong_dofs(boundary.as_ref(), grade),
        )?;

        if options.harmonic_dim > reduced_layout.reduced_dimension() {
            return Err(GpError::InvalidHarmonicDimension {
                requested: options.harmonic_dim,
                reduced_dimension: reduced_layout.reduced_dimension(),
            });
        }

        let reduced_mass = build_reduced_mass(topology, geometry, grade, &reduced_layout)?;
        let exact = build_exact_branch(
            topology,
            geometry,
            grade,
            &reduced_layout,
            &reduced_mass,
            boundary.as_ref(),
            &options,
        )?;
        let coexact = build_coexact_branch(
            topology,
            geometry,
            grade,
            &reduced_layout,
            &reduced_mass,
            boundary.as_ref(),
            &options,
        )?;
        let harmonic = build_harmonic_branch(
            topology,
            geometry,
            grade,
            &reduced_layout,
            &reduced_mass,
            boundary.as_ref(),
            &options,
        )?;

        Ok(Self {
            grade,
            reduced_layout,
            reduced_mass,
            exact,
            coexact,
            harmonic,
            boundary,
        })
    }

    pub fn grade(&self) -> ExteriorGrade {
        self.grade
    }

    pub fn reduced_layout(&self) -> &ReducedFormLayout {
        &self.reduced_layout
    }

    pub fn reduced_mass(&self) -> &CsrMatrix {
        &self.reduced_mass
    }

    pub fn ambient_dimension(&self) -> usize {
        self.reduced_layout.reduced_dimension()
    }

    pub fn boundary(&self) -> Option<&BoundaryConditionSpec> {
        self.boundary.as_ref()
    }

    pub fn branch_basis(&self, kind: HodgeBranchKind) -> &HodgeBranchBasis {
        match kind {
            HodgeBranchKind::Exact => &self.exact,
            HodgeBranchKind::Coexact => &self.coexact,
            HodgeBranchKind::Harmonic => &self.harmonic,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HodgeBranchConfig {
    pub kappa: f64,
    pub tau: f64,
    pub mode_count: usize,
}

impl Default for HodgeBranchConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            tau: 1.0,
            mode_count: DEFAULT_K,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeCompositionalConfig {
    pub alpha: f64,
    pub exact: HodgeBranchConfig,
    pub coexact: HodgeBranchConfig,
    pub harmonic: HodgeBranchConfig,
}

impl Default for HodgeCompositionalConfig {
    fn default() -> Self {
        Self {
            alpha: 2.0,
            exact: HodgeBranchConfig::default(),
            coexact: HodgeBranchConfig::default(),
            harmonic: HodgeBranchConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HodgeCompositionalGp {
    basis: HodgeDecomposedBasis,
    config: HodgeCompositionalConfig,
    exact_features: Mat<f64>,
    coexact_features: Mat<f64>,
    harmonic_features: Mat<f64>,
    combined_features: Mat<f64>,
}

impl HodgeCompositionalGp {
    pub fn from_hodge_decomposition(
        basis: HodgeDecomposedBasis,
        config: HodgeCompositionalConfig,
    ) -> Result<Self, GpError> {
        validate_alpha(config.alpha)?;
        let exact_features = basis.exact.feature_matrix(config.alpha, config.exact)?;
        let coexact_features = basis.coexact.feature_matrix(config.alpha, config.coexact)?;
        let harmonic_features = basis
            .harmonic
            .feature_matrix(config.alpha, config.harmonic)?;
        let combined_features =
            concatenate_feature_matrices(&[&exact_features, &coexact_features, &harmonic_features]);

        Ok(Self {
            basis,
            config,
            exact_features,
            coexact_features,
            harmonic_features,
            combined_features,
        })
    }

    pub fn basis(&self) -> &HodgeDecomposedBasis {
        &self.basis
    }

    pub fn config(&self) -> &HodgeCompositionalConfig {
        &self.config
    }

    pub fn ambient_dimension(&self) -> usize {
        self.combined_features.nrows()
    }

    pub fn latent_dimension(&self) -> usize {
        self.combined_features.ncols()
    }

    pub fn combined_prior_variance(&self) -> Vec<f64> {
        diagonal_from_feature_matrix(&self.combined_features)
    }

    pub fn branch_prior_variance(&self, kind: HodgeBranchKind) -> Vec<f64> {
        diagonal_from_feature_matrix(self.branch_features(kind))
    }

    pub fn combined_covariance_matrix(&self) -> Mat<f64> {
        full_covariance_from_feature_matrix(&self.combined_features)
    }

    pub fn branch_covariance_matrix(&self, kind: HodgeBranchKind) -> Mat<f64> {
        full_covariance_from_feature_matrix(self.branch_features(kind))
    }

    pub fn condition_linear_observations(
        &self,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedGp, GpError> {
        condition_low_rank_factors(
            &self.combined_features,
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    pub fn condition_linear_observations_with_covariance(
        &self,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedFullGp, GpError> {
        condition_low_rank_factors_with_covariance(
            &self.combined_features,
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    pub fn condition_branch_linear_observations(
        &self,
        kind: HodgeBranchKind,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedGp, GpError> {
        condition_low_rank_factors(
            self.branch_features(kind),
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    pub fn condition_branch_linear_observations_with_covariance(
        &self,
        kind: HodgeBranchKind,
        observation_matrix: &CsrMatrix,
        observation_values: &[f64],
        noise_variance: f64,
    ) -> Result<ConditionedFullGp, GpError> {
        condition_low_rank_factors_with_covariance(
            self.branch_features(kind),
            observation_matrix,
            observation_values,
            noise_variance,
        )
    }

    pub fn sample_from_standard_normal(&self, z: &[f64]) -> Result<Vec<f64>, GpError> {
        if z.len() != self.latent_dimension() {
            return Err(GpError::SampleDimensionMismatch {
                expected: self.latent_dimension(),
                got: z.len(),
            });
        }

        let mut sample = vec![0.0; self.ambient_dimension()];
        for (col, &latent_value) in z.iter().enumerate() {
            for (row, output) in sample.iter_mut().enumerate() {
                *output += self.combined_features[(row, col)] * latent_value;
            }
        }
        Ok(sample)
    }

    fn branch_features(&self, kind: HodgeBranchKind) -> &Mat<f64> {
        match kind {
            HodgeBranchKind::Exact => &self.exact_features,
            HodgeBranchKind::Coexact => &self.coexact_features,
            HodgeBranchKind::Harmonic => &self.harmonic_features,
        }
    }
}

pub fn estimate_harmonic_dim(
    topology: &Complex,
    grade: ExteriorGrade,
    boundary: Option<&BoundaryConditionSpec>,
) -> usize {
    match boundary {
        None => topology.homology_dim(grade),
        Some(boundary) => {
            let strong_k_minus_one = |index| grade > 0 && boundary.contains(grade - 1, index);
            let strong_k = |index| boundary.contains(grade, index);
            let strong_k_plus_one = |index| boundary.contains(grade + 1, index);
            topology.relative_homology_dim(
                grade,
                &strong_k_minus_one,
                &strong_k,
                &strong_k_plus_one,
            )
        }
    }
}

fn strong_dofs<'a>(
    boundary: Option<&'a BoundaryConditionSpec>,
    grade: ExteriorGrade,
) -> Option<&'a BTreeSet<usize>> {
    boundary.and_then(|boundary| boundary.strong_dofs(grade))
}

fn build_reduced_mass(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    layout: &ReducedFormLayout,
) -> Result<CsrMatrix, GpError> {
    let mass = CsrMatrix::from(&formoniq::assemble::assemble_galmat(
        topology,
        geometry,
        HodgeMassElmat::new(topology.dim(), grade),
    ));
    ReducedFormLayout::restrict_operator(layout, layout, &mass)
}

fn build_exact_branch(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    ambient_layout: &ReducedFormLayout,
    ambient_mass: &CsrMatrix,
    boundary: Option<&BoundaryConditionSpec>,
    options: &HodgeBuildOptions,
) -> Result<HodgeBranchBasis, GpError> {
    if grade == 0 || options.exact_mode_count == 0 {
        return Ok(HodgeBranchBasis::empty(ambient_layout.reduced_dimension()));
    }

    let source_grade = grade - 1;
    let source_layout = ReducedFormLayout::from_strong_dofs(
        topology.nsimplices(source_grade),
        strong_dofs(boundary, source_grade),
    )?;
    if source_layout.reduced_dimension() == 0 {
        return Ok(HodgeBranchBasis::empty(ambient_layout.reduced_dimension()));
    }

    let full_transform = CsrMatrix::from(&topology.exterior_derivative_operator(source_grade));
    let reduced_transform =
        ReducedFormLayout::restrict_operator(ambient_layout, &source_layout, &full_transform)?;

    let candidates = collect_transformed_modes(
        source_layout.reduced_dimension(),
        options.exact_mode_count,
        options,
        |requested| {
            solve_source_eigenpairs(
                topology,
                geometry,
                source_grade,
                requested,
                boundary,
                options.eigensolver.mass_solve,
            )
        },
        |eigenvectors| {
            let eigenvectors = normalize_mode_matrix(&source_layout, eigenvectors.clone())?;
            sparse_matrix_times_dense(&reduced_transform, &eigenvectors)
        },
    )?;

    Ok(orthonormalize_branch(
        ambient_layout.reduced_dimension(),
        candidates,
        ambient_mass,
        options.eigenvalue_zero_tolerance,
    ))
}

fn build_coexact_branch(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    ambient_layout: &ReducedFormLayout,
    ambient_mass: &CsrMatrix,
    boundary: Option<&BoundaryConditionSpec>,
    options: &HodgeBuildOptions,
) -> Result<HodgeBranchBasis, GpError> {
    if grade >= topology.dim() || options.coexact_mode_count == 0 {
        return Ok(HodgeBranchBasis::empty(ambient_layout.reduced_dimension()));
    }

    let source_grade = grade + 1;
    let source_layout = ReducedFormLayout::from_strong_dofs(
        topology.nsimplices(source_grade),
        strong_dofs(boundary, source_grade),
    )?;
    if source_layout.reduced_dimension() == 0 || ambient_layout.reduced_dimension() == 0 {
        return Ok(HodgeBranchBasis::empty(ambient_layout.reduced_dimension()));
    }

    let codifferential = build_reduced_codifferential_transform(
        topology,
        geometry,
        grade,
        ambient_layout,
        &source_layout,
        boundary,
        ambient_mass,
    )?;

    let candidates = collect_transformed_modes(
        source_layout.reduced_dimension(),
        options.coexact_mode_count,
        options,
        |requested| {
            solve_source_eigenpairs(
                topology,
                geometry,
                source_grade,
                requested,
                boundary,
                options.eigensolver.mass_solve,
            )
        },
        |eigenvectors| {
            let eigenvectors = normalize_mode_matrix(&source_layout, eigenvectors.clone())?;
            sparse_matrix_times_dense(&codifferential, &eigenvectors)
        },
    )?;

    Ok(orthonormalize_branch(
        ambient_layout.reduced_dimension(),
        candidates,
        ambient_mass,
        options.eigenvalue_zero_tolerance,
    ))
}

fn build_harmonic_branch(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    ambient_layout: &ReducedFormLayout,
    ambient_mass: &CsrMatrix,
    boundary: Option<&BoundaryConditionSpec>,
    options: &HodgeBuildOptions,
) -> Result<HodgeBranchBasis, GpError> {
    if options.harmonic_dim == 0 {
        return Ok(HodgeBranchBasis::empty(ambient_layout.reduced_dimension()));
    }

    let harmonics = catch_unwind(AssertUnwindSafe(|| {
        let galmats = hodge_laplace::MixedGalmats::compute(topology, geometry, grade);
        match boundary {
            None => hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
                topology,
                &galmats,
                grade,
                options.harmonic_dim,
                None,
                None,
            ),
            Some(boundary) => {
                let strong_k_minus_one = |index| grade > 0 && boundary.contains(grade - 1, index);
                let strong_k = |index| boundary.contains(grade, index);
                hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
                    topology,
                    &galmats,
                    grade,
                    options.harmonic_dim,
                    Some(&strong_k_minus_one),
                    Some(&strong_k),
                )
            }
        }
    }))
    .map_err(|_| GpError::InvalidHarmonicDimension {
        requested: options.harmonic_dim,
        reduced_dimension: ambient_layout.reduced_dimension(),
    })?;

    let harmonics = normalize_mode_matrix(ambient_layout, harmonics)?;
    let columns = matrix_columns(&harmonics);
    let candidates = columns
        .into_iter()
        .map(|column| (0.0, column))
        .collect::<Vec<_>>();
    Ok(orthonormalize_branch(
        ambient_layout.reduced_dimension(),
        candidates,
        ambient_mass,
        options.eigenvalue_zero_tolerance,
    ))
}

fn build_reduced_codifferential_transform(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    ambient_layout: &ReducedFormLayout,
    source_layout: &ReducedFormLayout,
    boundary: Option<&BoundaryConditionSpec>,
    ambient_mass: &CsrMatrix,
) -> Result<CsrMatrix, GpError> {
    let galmats = hodge_laplace::MixedGalmats::compute(topology, geometry, grade + 1);
    let full_codif = CsrMatrix::from(galmats.codif_u());
    let reduced_codif =
        ReducedFormLayout::restrict_operator(ambient_layout, source_layout, &full_codif)?;

    if ambient_layout.reduced_dimension() == 0 {
        return Ok(CsrMatrix::from(&CooMatrix::new(
            0,
            source_layout.reduced_dimension(),
        )));
    }

    let solver = FaerCholesky::new(ambient_mass.clone());
    let mut reduced = CooMatrix::new(
        ambient_layout.reduced_dimension(),
        source_layout.reduced_dimension(),
    );
    for source_column in 0..source_layout.reduced_dimension() {
        let rhs = sparse_column_to_vector(
            &reduced_codif,
            source_column,
            ambient_layout.reduced_dimension(),
        );
        let solution = solver.solve(&rhs);
        for (row, value) in solution.iter().enumerate() {
            if value.abs() > 1e-12 {
                reduced.push(row, source_column, *value);
            }
        }
    }
    let _ = boundary;
    Ok(CsrMatrix::from(&reduced))
}

fn collect_transformed_modes<S, T>(
    source_dimension: usize,
    desired_mode_count: usize,
    options: &HodgeBuildOptions,
    mut solve_source: S,
    mut transform_modes: T,
) -> Result<Vec<(f64, NaVector)>, GpError>
where
    S: FnMut(usize) -> Result<(NaVector, NaMatrix), GpError>,
    T: FnMut(&NaMatrix) -> Result<NaMatrix, GpError>,
{
    if desired_mode_count == 0 || source_dimension == 0 {
        return Ok(Vec::new());
    }

    let mut requested = desired_mode_count
        .saturating_add(options.eigensolver.oversampling)
        .min(source_dimension)
        .max(1);

    let mut candidates = Vec::new();
    loop {
        let (eigenvalues, source_modes) = solve_source(requested)?;
        let transformed_modes = transform_modes(&source_modes)?;
        candidates.clear();
        for column in 0..transformed_modes.ncols() {
            let eigenvalue = eigenvalues[column];
            if !eigenvalue.is_finite() || eigenvalue <= options.eigenvalue_zero_tolerance {
                continue;
            }
            candidates.push((eigenvalue, transformed_modes.column(column).into_owned()));
        }

        if candidates.len() >= desired_mode_count || requested == source_dimension {
            break;
        }

        let next_requested = requested.saturating_mul(2).min(source_dimension);
        if next_requested == requested {
            break;
        }
        requested = next_requested;
    }

    Ok(candidates)
}

fn solve_source_eigenpairs(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    requested: usize,
    boundary: Option<&BoundaryConditionSpec>,
    mass_solve: GhiepReducedSolve,
) -> Result<(NaVector, NaMatrix), GpError> {
    if requested == 0 {
        return Ok((
            NaVector::zeros(0),
            NaMatrix::zeros(topology.nsimplices(grade), 0),
        ));
    }

    if grade == 0 {
        return solve_reduced_zero_form_eigenpairs(
            topology,
            geometry,
            requested,
            strong_dofs(boundary, 0),
        );
    }

    solve_reduced_hodge_eigenpairs(topology, geometry, grade, requested, boundary, mass_solve)
}

fn solve_reduced_zero_form_eigenpairs(
    topology: &Complex,
    geometry: &MeshLengths,
    requested: usize,
    strong_dofs: Option<&BTreeSet<usize>>,
) -> Result<(NaVector, NaMatrix), GpError> {
    let galmats = laplace_beltrami::LaplaceBeltramiGalmats::compute(topology, geometry);
    match strong_dofs {
        None => Ok(laplace_beltrami::solve_laplace_beltrami_evp_as_matrix(
            topology, geometry, requested,
        )),
        Some(strong_dofs) => {
            let mut stiffness = galmats.stiffness().clone();
            let mut mass = galmats.mass().clone();
            let strong_dofs = strong_dofs.iter().copied().collect::<HashSet<_>>();
            assemble::drop_dofs_galmat(&strong_dofs, &mut stiffness);
            assemble::drop_dofs_galmat(&strong_dofs, &mut mass);
            Ok(petsc_ghiep(
                &CsrMatrix::from(&stiffness),
                &CsrMatrix::from(&mass),
                requested,
            ))
        }
    }
}

fn solve_reduced_hodge_eigenpairs(
    topology: &Complex,
    geometry: &MeshLengths,
    grade: ExteriorGrade,
    requested: usize,
    boundary: Option<&BoundaryConditionSpec>,
    mass_solve: GhiepReducedSolve,
) -> Result<(NaVector, NaMatrix), GpError> {
    let galmats = hodge_laplace::MixedGalmats::compute(topology, geometry, grade);
    let mut mass_sigma = galmats.mass_sigma().clone();
    let mut dif_sigma = galmats.dif_sigma().clone();
    let mut codif_u = galmats.codif_u().clone();
    let mut codifdif_u = galmats.codifdif_u().clone();
    let mut mass_u = galmats.mass_u().clone();

    if let Some(boundary) = boundary {
        let strong_k_minus_one = (0..mass_sigma.nrows())
            .filter(|&index| grade > 0 && boundary.contains(grade - 1, index))
            .collect::<HashSet<_>>();
        let strong_k = (0..mass_u.nrows())
            .filter(|&index| boundary.contains(grade, index))
            .collect::<HashSet<_>>();

        assemble::drop_dofs_galmat(&strong_k_minus_one, &mut mass_sigma);
        assemble::drop_dofs_rectangular_galmat(&strong_k, &strong_k_minus_one, &mut dif_sigma);
        assemble::drop_dofs_rectangular_galmat(&strong_k_minus_one, &strong_k, &mut codif_u);
        assemble::drop_dofs_galmat(&strong_k, &mut codifdif_u);
        assemble::drop_dofs_galmat(&strong_k, &mut mass_u);
    }

    if codifdif_u.nrows() == 0 && mass_u.nrows() > 0 {
        codifdif_u = CooMatrix::zeros(mass_u.nrows(), mass_u.nrows());
    }

    if mass_sigma.nrows() == 0 {
        let (eigenvalues, modes) = petsc_ghiep(
            &CsrMatrix::from(&codifdif_u),
            &CsrMatrix::from(&mass_u),
            requested,
        );
        return Ok((eigenvalues, modes));
    }

    let (eigenvalues, _sigma_modes, modes) = petsc_ghep_reduced_with_which(
        &CsrMatrix::from(&codifdif_u),
        &CsrMatrix::from(&dif_sigma),
        &CsrMatrix::from(&codif_u),
        &CsrMatrix::from(&mass_sigma),
        &CsrMatrix::from(&mass_u),
        requested,
        GhiepWhich::Smallest,
        mass_solve,
    );
    Ok((eigenvalues, modes))
}

fn orthonormalize_branch(
    ambient_dimension: usize,
    candidates: Vec<(f64, NaVector)>,
    ambient_mass: &CsrMatrix,
    tolerance: f64,
) -> HodgeBranchBasis {
    let mut eigenvalues = Vec::new();
    let mut columns = Vec::new();

    for (eigenvalue, mut column) in candidates {
        for previous in &columns {
            let coefficient = bilinear_form_sparse(ambient_mass, previous, &column);
            column -= previous * coefficient;
        }

        let norm_sq = bilinear_form_sparse(ambient_mass, &column, &column);
        if !norm_sq.is_finite() || norm_sq <= tolerance {
            continue;
        }

        column /= norm_sq.sqrt();
        eigenvalues.push(eigenvalue);
        columns.push(column);
    }

    let eigenvectors = if columns.is_empty() {
        Mat::zeros(ambient_dimension, 0)
    } else {
        na_matrix_to_faer(&NaMatrix::from_columns(&columns))
    };

    HodgeBranchBasis {
        eigenvalues,
        eigenvectors,
    }
}

fn sparse_matrix_times_dense(lhs: &CsrMatrix, rhs: &NaMatrix) -> Result<NaMatrix, GpError> {
    if lhs.ncols() != rhs.nrows() {
        return Err(GpError::ObservationMatrixDimensionMismatch {
            expected: rhs.nrows(),
            got: lhs.ncols(),
        });
    }

    let mut product = NaMatrix::zeros(lhs.nrows(), rhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        for dense_col in 0..rhs.ncols() {
            product[(row, dense_col)] += *value * rhs[(col, dense_col)];
        }
    }
    Ok(product)
}

fn sparse_column_to_vector(matrix: &CsrMatrix, column: usize, nrows: usize) -> NaVector {
    let mut rhs = NaVector::zeros(nrows);
    for (row, col, value) in matrix.triplet_iter() {
        if col == column {
            rhs[row] += *value;
        }
    }
    rhs
}

fn matrix_columns(matrix: &NaMatrix) -> Vec<NaVector> {
    matrix
        .column_iter()
        .map(|column| column.into_owned())
        .collect()
}

fn normalize_mode_matrix(layout: &ReducedFormLayout, modes: NaMatrix) -> Result<NaMatrix, GpError> {
    if modes.nrows() == layout.reduced_dimension() {
        return Ok(modes);
    }
    if modes.nrows() == layout.full_dimension() {
        let mut reduced = NaMatrix::zeros(layout.reduced_dimension(), modes.ncols());
        for col in 0..modes.ncols() {
            let full_column = modes.column(col).into_owned();
            let reduced_column = layout.reduce_vector(&full_column)?;
            for row in 0..layout.reduced_dimension() {
                reduced[(row, col)] = reduced_column[row];
            }
        }
        return Ok(reduced);
    }
    Err(GpError::ObservationMatrixDimensionMismatch {
        expected: layout.reduced_dimension(),
        got: modes.nrows(),
    })
}

fn concatenate_feature_matrices(matrices: &[&Mat<f64>]) -> Mat<f64> {
    let ambient_dimension = matrices.first().map_or(0, |matrix| matrix.nrows());
    let total_columns = matrices.iter().map(|matrix| matrix.ncols()).sum();
    let mut concatenated = Mat::zeros(ambient_dimension, total_columns);

    let mut column_offset = 0;
    for matrix in matrices {
        for col in 0..matrix.ncols() {
            for row in 0..ambient_dimension {
                concatenated[(row, column_offset + col)] = matrix[(row, col)];
            }
        }
        column_offset += matrix.ncols();
    }

    concatenated
}

fn na_matrix_to_faer(matrix: &NaMatrix) -> Mat<f64> {
    let mut out = Mat::zeros(matrix.nrows(), matrix.ncols());
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            out[(row, col)] = matrix[(row, col)];
        }
    }
    out
}

fn branch_matern_weight(
    eigenvalue: f64,
    alpha: f64,
    config: HodgeBranchConfig,
) -> Result<f64, GpError> {
    if !eigenvalue.is_finite() {
        return Err(GpError::NonFiniteEigenvalue { value: eigenvalue });
    }
    let shifted = config.kappa * config.kappa + eigenvalue;
    if !shifted.is_finite() || shifted <= 0.0 {
        return Err(GpError::InvalidEigenvalue { value: eigenvalue });
    }
    Ok((1.0 / config.tau) * (1.0 / config.tau) * shifted.powf(-alpha))
}

fn validate_alpha(alpha: f64) -> Result<(), GpError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(GpError::InvalidAlpha);
    }
    Ok(())
}

fn validate_branch_config(config: HodgeBranchConfig) -> Result<(), GpError> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(GpError::InvalidKappa);
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(GpError::InvalidTau);
    }
    Ok(())
}
