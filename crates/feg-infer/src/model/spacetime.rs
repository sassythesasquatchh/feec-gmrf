use crate::sparse::{core_triplet_to_feec_csr, feec_csr_to_gmrf};
use common::linalg::nalgebra::{CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Vector as FeecVector};
use feg_core::{SoftBoundaryConstraint, SpatialPriorSlice, StateLayout};
use gmrf_core::{StackedObservationSystem, TimeStackedObservationBuilder};

#[derive(Debug, Clone)]
pub struct SpacetimeLinearObservationBuilder {
    inner: TimeStackedObservationBuilder,
    layout: StateLayout,
}

impl SpacetimeLinearObservationBuilder {
    pub fn new(slice_count: usize, slice: &SpatialPriorSlice) -> Self {
        Self {
            inner: TimeStackedObservationBuilder::new(slice_count, slice.state_dimension()),
            layout: slice.layout.clone(),
        }
    }

    pub fn push_slice_block_from_full(
        &mut self,
        slice_index: usize,
        full_block: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        let (reduced, reduced_bias) = restrict_columns_and_fold_fixed(full_block, bias, &self.layout)?;
        self.inner
            .push_slice_block(
                slice_index,
                &feec_csr_to_gmrf(&reduced),
                observations.as_slice(),
                reduced_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_slice_block_from_reduced(
        &mut self,
        slice_index: usize,
        reduced_block: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        self.inner
            .push_slice_block(
                slice_index,
                &feec_csr_to_gmrf(reduced_block),
                observations.as_slice(),
                bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_transition_block_from_full(
        &mut self,
        left_slice: usize,
        left_full: &FeecCsr,
        right_full: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        let (left_reduced, left_bias) = restrict_columns_and_fold_fixed(left_full, bias, &self.layout)?;
        let (right_reduced, final_bias) =
            restrict_columns_and_fold_fixed(right_full, &left_bias, &self.layout)?;
        self.inner
            .push_transition_block(
                left_slice,
                &feec_csr_to_gmrf(&left_reduced),
                &feec_csr_to_gmrf(&right_reduced),
                observations.as_slice(),
                final_bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn push_transition_block_from_reduced(
        &mut self,
        left_slice: usize,
        left_reduced: &FeecCsr,
        right_reduced: &FeecCsr,
        observations: &FeecVector,
        bias: &FeecVector,
        variance: f64,
    ) -> Result<(), String> {
        self.inner
            .push_transition_block(
                left_slice,
                &feec_csr_to_gmrf(left_reduced),
                &feec_csr_to_gmrf(right_reduced),
                observations.as_slice(),
                bias.as_slice(),
                variance,
            )
            .map_err(|err| err.to_string())
    }

    pub fn add_soft_boundary_constraints(&mut self, slice_count: usize, slice: &SpatialPriorSlice) -> Result<(), String> {
        for time_index in 0..slice_count {
            for constraint in &slice.soft_boundary_constraints {
                self.push_soft_constraint(time_index, constraint)?;
            }
        }
        Ok(())
    }

    pub fn finish(self) -> StackedObservationSystem {
        self.inner.finish()
    }

    fn push_soft_constraint(
        &mut self,
        slice_index: usize,
        constraint: &SoftBoundaryConstraint,
    ) -> Result<(), String> {
        let block = core_triplet_to_feec_csr(&constraint.operator);
        let observations = FeecVector::from_vec(constraint.target.clone());
        let bias = FeecVector::zeros(observations.len());
        self.push_slice_block_from_reduced(
            slice_index,
            &block,
            &observations,
            &bias,
            constraint.variance,
        )
    }
}

fn restrict_columns_and_fold_fixed(
    full_block: &FeecCsr,
    bias: &FeecVector,
    layout: &StateLayout,
) -> Result<(FeecCsr, FeecVector), String> {
    if full_block.ncols() != layout.full_dimension {
        return Err(format!(
            "full observation block column count {} does not match layout dimension {}",
            full_block.ncols(),
            layout.full_dimension
        ));
    }
    if full_block.nrows() != bias.len() {
        return Err(format!(
            "bias length {} does not match observation row count {}",
            bias.len(),
            full_block.nrows()
        ));
    }

    let mut reduced = FeecCoo::new(full_block.nrows(), layout.reduced_dimension());
    let mut reduced_bias = bias.clone();
    let reduced_map = reduced_index_map(layout);
    for (row, col, value) in full_block.triplet_iter() {
        if let Some(reduced_col) = reduced_map[col] {
            reduced.push(row, reduced_col, *value);
        } else if let Some(fixed) = layout.fixed_dofs.iter().find(|entry| entry.index == col) {
            reduced_bias[row] += *value * fixed.value;
        }
    }
    Ok((FeecCsr::from(&reduced), reduced_bias))
}

fn reduced_index_map(layout: &StateLayout) -> Vec<Option<usize>> {
    let mut map = vec![None; layout.full_dimension];
    for (reduced, full) in layout.active_dofs.iter().copied().enumerate() {
        map[full] = Some(reduced);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use feg_core::{FixedDof, SparseTripletMatrix, StateLayout};

    #[test]
    fn hard_fixed_columns_are_folded_into_bias() {
        let mut block = FeecCoo::new(1, 3);
        block.push(0, 0, 2.0);
        block.push(0, 1, -1.0);
        block.push(0, 2, 3.0);
        let block = FeecCsr::from(&block);
        let bias = FeecVector::from_vec(vec![0.5]);
        let layout = StateLayout::new(
            3,
            vec![0, 2],
            vec![FixedDof { index: 1, value: 4.0 }],
        );

        let (reduced, folded_bias) =
            restrict_columns_and_fold_fixed(&block, &bias, &layout).unwrap();

        let mut rows = reduced.triplet_iter().map(|(r, c, v)| (r, c, *v)).collect::<Vec<_>>();
        rows.sort_by_key(|(r, c, _)| (*r, *c));
        assert_eq!(rows, vec![(0, 0, 2.0), (0, 1, 3.0)]);
        assert!((folded_bias[0] + 3.5).abs() < 1e-12);
    }

    #[test]
    fn soft_constraints_are_added_for_each_time_slice() {
        let slice = SpatialPriorSlice {
            mass: SparseTripletMatrix::new(2, 2),
            drift: SparseTripletMatrix::new(2, 2),
            initial_precision: SparseTripletMatrix::new(2, 2),
            driving_noise_precision: SparseTripletMatrix::new(2, 2),
            layout: StateLayout::identity(2),
            soft_boundary_constraints: vec![SoftBoundaryConstraint {
                operator: {
                    let mut mat = SparseTripletMatrix::new(1, 2);
                    mat.push(0, 1, 1.0);
                    mat
                },
                target: vec![2.0],
                variance: 0.25,
            }],
        };

        let mut builder = SpacetimeLinearObservationBuilder::new(3, &slice);
        builder.add_soft_boundary_constraints(3, &slice).unwrap();
        let system = builder.finish();
        assert_eq!(system.matrix.nrows(), 3);
        assert_eq!(system.matrix.ncols(), 6);
        assert_eq!(system.observations.as_slice(), &[4.0, 4.0, 4.0]);
    }
}
