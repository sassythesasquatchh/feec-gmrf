//! Shared contracts for FEEC/GMRF spacetime integration.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryTreatment {
    HardEssential,
    SoftEssential { variance: f64 },
    Natural,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRegionSpec {
    pub name: String,
    pub dofs: Vec<usize>,
    pub values: Vec<f64>,
    pub treatment: BoundaryTreatment,
}

impl BoundaryRegionSpec {
    pub fn new(
        name: impl Into<String>,
        dofs: Vec<usize>,
        values: Vec<f64>,
        treatment: BoundaryTreatment,
    ) -> Self {
        assert_eq!(
            dofs.len(),
            values.len(),
            "boundary dof and value counts must match"
        );
        Self {
            name: name.into(),
            dofs,
            values,
            treatment,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundarySpec {
    pub state_regions: Vec<BoundaryRegionSpec>,
    pub auxiliary_regions: Vec<BoundaryRegionSpec>,
}

impl BoundarySpec {
    pub fn with_state_region(mut self, region: BoundaryRegionSpec) -> Self {
        self.state_regions.push(region);
        self
    }

    pub fn with_auxiliary_region(mut self, region: BoundaryRegionSpec) -> Self {
        self.auxiliary_regions.push(region);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseTriplet {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTripletMatrix {
    nrows: usize,
    ncols: usize,
    triplets: Vec<SparseTriplet>,
}

impl SparseTripletMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            triplets: Vec::new(),
        }
    }

    pub fn from_triplets(
        nrows: usize,
        ncols: usize,
        triplets: impl IntoIterator<Item = SparseTriplet>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            triplets: triplets.into_iter().collect(),
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn nnz(&self) -> usize {
        self.triplets.len()
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.triplets.push(SparseTriplet { row, col, value });
    }

    pub fn triplet_iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.triplets.iter().map(|t| (t.row, t.col, t.value))
    }

    pub fn transpose(&self) -> Self {
        Self::from_triplets(
            self.ncols,
            self.nrows,
            self.triplets.iter().map(|entry| SparseTriplet {
                row: entry.col,
                col: entry.row,
                value: entry.value,
            }),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedDof {
    pub index: usize,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateLayout {
    pub full_dimension: usize,
    pub active_dofs: Vec<usize>,
    pub fixed_dofs: Vec<FixedDof>,
}

impl StateLayout {
    pub fn new(full_dimension: usize, active_dofs: Vec<usize>, fixed_dofs: Vec<FixedDof>) -> Self {
        let active_set = active_dofs.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(
            active_set.len(),
            active_dofs.len(),
            "active dofs must be unique"
        );
        for dof in &active_dofs {
            assert!(
                *dof < full_dimension,
                "active dof index {dof} must be less than full dimension {full_dimension}"
            );
        }
        for fixed in &fixed_dofs {
            assert!(
                fixed.index < full_dimension,
                "fixed dof index {} must be less than full dimension {}",
                fixed.index,
                full_dimension
            );
            assert!(
                !active_set.contains(&fixed.index),
                "fixed dof {} must not also be active",
                fixed.index
            );
        }
        Self {
            full_dimension,
            active_dofs,
            fixed_dofs,
        }
    }

    pub fn identity(dimension: usize) -> Self {
        Self {
            full_dimension: dimension,
            active_dofs: (0..dimension).collect(),
            fixed_dofs: Vec::new(),
        }
    }

    pub fn reduced_dimension(&self) -> usize {
        self.active_dofs.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SoftBoundaryConstraint {
    pub operator: SparseTripletMatrix,
    pub target: Vec<f64>,
    pub variance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepresentationPreference {
    Auto,
    ForceCollapsed,
    ForceLatent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianPriorSpec {
    pub mean: Vec<f64>,
    pub precision: SparseTripletMatrix,
}

impl GaussianPriorSpec {
    pub fn dimension(&self) -> usize {
        self.precision.nrows()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.precision.nrows() != self.precision.ncols() {
            return Err("Gaussian prior precision must be square".to_string());
        }
        if self.mean.len() != self.precision.nrows() {
            return Err(format!(
                "Gaussian prior mean length {} must match precision dimension {}",
                self.mean.len(),
                self.precision.nrows()
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearUncertainInputSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub prior: GaussianPriorSpec,
    pub preference: RepresentationPreference,
    pub collapsed_precision: Option<SparseTripletMatrix>,
}

impl LinearUncertainInputSpec {
    pub fn validate(&self, residual_dimension: usize) -> Result<(), String> {
        self.prior.validate()?;
        if self.operator.nrows() != residual_dimension {
            return Err(format!(
                "uncertain input `{}` operator row count {} must match residual dimension {}",
                self.name,
                self.operator.nrows(),
                residual_dimension
            ));
        }
        if self.operator.ncols() != self.prior.dimension() {
            return Err(format!(
                "uncertain input `{}` operator column count {} must match prior dimension {}",
                self.name,
                self.operator.ncols(),
                self.prior.dimension()
            ));
        }
        if let Some(collapsed) = &self.collapsed_precision {
            if collapsed.nrows() != residual_dimension || collapsed.ncols() != residual_dimension {
                return Err(format!(
                    "uncertain input `{}` collapsed precision must be {}x{}, got {}x{}",
                    self.name,
                    residual_dimension,
                    residual_dimension,
                    collapsed.nrows(),
                    collapsed.ncols()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianMeasurementSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub variance: f64,
}

impl LinearGaussianMeasurementSpec {
    pub fn validate(&self, state_dimension: usize) -> Result<(), String> {
        if self.operator.ncols() != state_dimension {
            return Err(format!(
                "measurement `{}` operator column count {} must match state dimension {}",
                self.name,
                self.operator.ncols(),
                state_dimension
            ));
        }
        if self.operator.nrows() != self.observations.len() {
            return Err(format!(
                "measurement `{}` observation length {} must match operator row count {}",
                self.name,
                self.observations.len(),
                self.operator.nrows()
            ));
        }
        if self.operator.nrows() != self.bias.len() {
            return Err(format!(
                "measurement `{}` bias length {} must match operator row count {}",
                self.name,
                self.bias.len(),
                self.operator.nrows()
            ));
        }
        if !self.variance.is_finite() || self.variance <= 0.0 {
            return Err(format!(
                "measurement `{}` variance must be finite and positive",
                self.name
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionWeightedGaussianMeasurementSpec {
    pub name: String,
    pub operator: SparseTripletMatrix,
    pub observations: Vec<f64>,
    pub bias: Vec<f64>,
    pub precision: SparseTripletMatrix,
}

impl PrecisionWeightedGaussianMeasurementSpec {
    pub fn validate(&self, state_dimension: usize) -> Result<(), String> {
        if self.operator.ncols() != state_dimension {
            return Err(format!(
                "precision-weighted measurement `{}` operator column count {} must match state dimension {}",
                self.name,
                self.operator.ncols(),
                state_dimension
            ));
        }
        if self.operator.nrows() != self.observations.len() {
            return Err(format!(
                "precision-weighted measurement `{}` observation length {} must match operator row count {}",
                self.name,
                self.observations.len(),
                self.operator.nrows()
            ));
        }
        if self.operator.nrows() != self.bias.len() {
            return Err(format!(
                "precision-weighted measurement `{}` bias length {} must match operator row count {}",
                self.name,
                self.bias.len(),
                self.operator.nrows()
            ));
        }
        if self.precision.nrows() != self.operator.nrows()
            || self.precision.ncols() != self.operator.nrows()
        {
            return Err(format!(
                "precision-weighted measurement `{}` precision must be {}x{}, got {}x{}",
                self.name,
                self.operator.nrows(),
                self.operator.nrows(),
                self.precision.nrows(),
                self.precision.ncols()
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpatialPriorSlice {
    pub mass: SparseTripletMatrix,
    pub drift: SparseTripletMatrix,
    pub initial_precision: SparseTripletMatrix,
    pub driving_noise_precision: SparseTripletMatrix,
    pub layout: StateLayout,
    pub soft_boundary_constraints: Vec<SoftBoundaryConstraint>,
}

impl SpatialPriorSlice {
    pub fn state_dimension(&self) -> usize {
        self.mass.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_triplet_matrix_transposes() {
        let mut matrix = SparseTripletMatrix::new(2, 3);
        matrix.push(0, 1, 2.0);
        matrix.push(1, 2, -3.0);

        let transpose = matrix.transpose();
        let triplets = transpose.triplet_iter().collect::<Vec<_>>();

        assert_eq!(transpose.nrows(), 3);
        assert_eq!(transpose.ncols(), 2);
        assert_eq!(triplets, vec![(1, 0, 2.0), (2, 1, -3.0)]);
    }

    #[test]
    fn state_layout_identity_keeps_all_dofs() {
        let layout = StateLayout::identity(4);
        assert_eq!(layout.reduced_dimension(), 4);
        assert!(layout.fixed_dofs.is_empty());
    }

    #[test]
    fn gaussian_prior_spec_rejects_mismatched_mean_length() {
        let spec = GaussianPriorSpec {
            mean: vec![1.0],
            precision: SparseTripletMatrix::new(2, 2),
        };
        assert!(spec.validate().is_err());
    }
}
