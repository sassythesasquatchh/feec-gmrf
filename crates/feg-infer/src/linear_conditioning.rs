use common::linalg::nalgebra::Matrix as FeecMatrix;
use faer::linalg::solvers::Solve;
use faer::Side;
use gmrf_core::observation::{apply_gaussian_observations, ht_weighted_observations};
use gmrf_core::types::{
    DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix, Vector as GmrfVector,
};
use gmrf_core::{Gmrf, GmrfError, SparseRowOperator, TransformedVarianceDecomposition};
use rand::Rng;
use rand::SeedableRng;
use std::collections::BTreeMap;
use std::error::Error;

const EPS: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedVarianceMode {
    Exact,
    Rbmc,
}

#[derive(Debug, Clone)]
pub struct DerivedOperator {
    pub operator: SparseRowOperator,
    pub variance_mode: DerivedVarianceMode,
}

pub type DerivedOperatorSet = BTreeMap<String, DerivedOperator>;

#[derive(Debug, Clone)]
pub struct HarmonicSubspace {
    pub basis: FeecMatrix,
    pub constraints: GmrfDenseMatrix,
    pub projector: Option<SparseRowOperator>,
}

#[derive(Debug, Clone, Copy)]
pub struct RbmcConfig {
    pub num_probes: usize,
    pub batch_count: usize,
    pub rng_seed: u64,
}

#[derive(Debug, Clone)]
pub struct LinearGaussianConditioningProblem {
    pub prior_precision: GmrfSparseMatrix,
    pub observation_operator: GmrfSparseMatrix,
    pub observations: GmrfVector,
    pub noise_variance: f64,
    pub harmonic_subspace: Option<HarmonicSubspace>,
    pub derived_operators: DerivedOperatorSet,
    pub rbmc: RbmcConfig,
}

#[derive(Debug, Clone)]
pub struct LinearGaussianConditioningResult {
    pub observations: GmrfVector,
    pub posterior_precision: GmrfSparseMatrix,
    pub information: GmrfVector,
    pub posterior_mean: GmrfVector,
    pub constrained_posterior_mean: Option<GmrfVector>,
    pub posterior_observations: GmrfVector,
    pub observation_residual: GmrfVector,
    pub prior_latent_variance: TransformedVarianceDecomposition,
    pub posterior_latent_variance: TransformedVarianceDecomposition,
    pub derived_prior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
    pub derived_posterior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedLinearGaussianConditioningProblem {
    observation_operator: GmrfSparseMatrix,
    noise_variance: f64,
    harmonic_subspace: Option<HarmonicSubspace>,
    posterior_precision: GmrfSparseMatrix,
    prior_latent_variance: TransformedVarianceDecomposition,
    posterior_latent_variance: TransformedVarianceDecomposition,
    derived_prior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
    derived_posterior_variances: BTreeMap<String, TransformedVarianceDecomposition>,
}

impl LinearGaussianConditioningProblem {
    pub fn solve(&self) -> Result<LinearGaussianConditioningResult, Box<dyn Error>> {
        let prepared = self.prepare()?;
        prepared.solve_with_observations(&self.observations)
    }

    pub(crate) fn prepare(
        &self,
    ) -> Result<PreparedLinearGaussianConditioningProblem, Box<dyn Error>> {
        self.validate()?;

        let zero_observations = GmrfVector::zeros(self.observation_operator.nrows());
        let (posterior_precision, _) = apply_gaussian_observations(
            &self.prior_precision,
            &self.observation_operator,
            &zero_observations,
            None,
            self.noise_variance,
        );

        let harmonic_constraints = self.harmonic_subspace.as_ref().map_or_else(
            || GmrfDenseMatrix::zeros(0, self.prior_precision.nrows()),
            |subspace| subspace.constraints.clone(),
        );

        let prior_factor = self.prior_precision.cholesky_sqrt_lower()?;
        let mut prior = Gmrf::from_mean_and_precision(
            GmrfVector::zeros(self.prior_precision.nrows()),
            self.prior_precision.clone(),
        )?
        .with_precision_sqrt(prior_factor);
        let prior_latent = prior.exact_constrained_variance_decomposition(&harmonic_constraints)?;
        let prior_latent_variance = TransformedVarianceDecomposition {
            unconstrained_diag: prior_latent.unconstrained_diag,
            constrained_diag: prior_latent.constrained_diag,
            removed_diag: prior_latent.removed_diag,
        };

        let posterior_factor = posterior_precision.cholesky_sqrt_lower()?;
        let mut posterior = Gmrf::from_mean_and_precision(
            GmrfVector::zeros(posterior_precision.nrows()),
            posterior_precision.clone(),
        )?
        .with_precision_sqrt(posterior_factor);
        let posterior_latent =
            posterior.exact_constrained_variance_decomposition(&harmonic_constraints)?;
        let posterior_latent_variance = TransformedVarianceDecomposition {
            unconstrained_diag: posterior_latent.unconstrained_diag,
            constrained_diag: posterior_latent.constrained_diag,
            removed_diag: posterior_latent.removed_diag,
        };

        let mut derived_prior_variances = BTreeMap::new();
        let mut derived_posterior_variances = BTreeMap::new();
        for (name, derived) in &self.derived_operators {
            let prior_decomposition = match derived.variance_mode {
                DerivedVarianceMode::Exact => exact_or_rbmc_decomposition(
                    &mut prior,
                    &self.prior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.rbmc,
                )?,
                DerivedVarianceMode::Rbmc => estimate_rbmc_decomposition(
                    &self.prior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.rbmc,
                )?,
            };
            let posterior_decomposition = match derived.variance_mode {
                DerivedVarianceMode::Exact => exact_or_rbmc_decomposition(
                    &mut posterior,
                    &posterior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.rbmc,
                )?,
                DerivedVarianceMode::Rbmc => estimate_rbmc_decomposition(
                    &posterior_precision,
                    &derived.operator,
                    &harmonic_constraints,
                    self.rbmc,
                )?,
            };

            derived_prior_variances.insert(name.clone(), prior_decomposition);
            derived_posterior_variances.insert(name.clone(), posterior_decomposition);
        }

        Ok(PreparedLinearGaussianConditioningProblem {
            observation_operator: self.observation_operator.clone(),
            noise_variance: self.noise_variance,
            harmonic_subspace: self.harmonic_subspace.clone(),
            posterior_precision,
            prior_latent_variance,
            posterior_latent_variance,
            derived_prior_variances,
            derived_posterior_variances,
        })
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
        let state_dim = self.prior_precision.nrows();
        if self.prior_precision.ncols() != state_dim {
            return Err("prior precision must be square".into());
        }
        if self.observation_operator.ncols() != state_dim {
            return Err("observation operator column count must match latent dimension".into());
        }
        if self.observations.len() != self.observation_operator.nrows() {
            return Err("observations length must match observation operator rows".into());
        }
        if !self.noise_variance.is_finite() || self.noise_variance <= 0.0 {
            return Err("noise_variance must be finite and positive".into());
        }
        if self.rbmc.num_probes == 0 {
            return Err("rbmc.num_probes must be >= 1".into());
        }
        if self.rbmc.batch_count == 0 {
            return Err("rbmc.batch_count must be >= 1".into());
        }
        if let Some(subspace) = &self.harmonic_subspace {
            if subspace.constraints.ncols() != state_dim {
                return Err("harmonic constraint columns must match latent dimension".into());
            }
            if let Some(projector) = &subspace.projector {
                if projector.ncols != state_dim {
                    return Err(
                        "harmonic projector column count must match latent dimension".into(),
                    );
                }
            }
        }
        for derived in self.derived_operators.values() {
            if derived.operator.ncols != state_dim {
                return Err("derived operator column count must match latent dimension".into());
            }
        }
        Ok(())
    }
}

impl PreparedLinearGaussianConditioningProblem {
    pub(crate) fn solve_with_observations(
        &self,
        observations: &GmrfVector,
    ) -> Result<LinearGaussianConditioningResult, Box<dyn Error>> {
        if observations.len() != self.observation_operator.nrows() {
            return Err("observations length must match observation operator rows".into());
        }

        let information = ht_weighted_observations(
            &self.observation_operator,
            observations,
            1.0 / self.noise_variance,
        );
        let posterior_precision = self.posterior_precision.clone();
        let mut posterior =
            Gmrf::from_information_and_precision(information.clone(), posterior_precision.clone())?;
        let posterior_mean = posterior.mean().clone();
        let constrained_posterior_mean = self
            .harmonic_subspace
            .as_ref()
            .map(|subspace| {
                posterior.constrained_mean(
                    &subspace.constraints,
                    &GmrfVector::zeros(subspace.constraints.nrows()),
                )
            })
            .transpose()?;

        let posterior_observations = &self.observation_operator * &posterior_mean;
        let observation_residual = &posterior_observations - observations;

        Ok(LinearGaussianConditioningResult {
            observations: observations.clone(),
            posterior_precision,
            information,
            posterior_mean,
            constrained_posterior_mean,
            posterior_observations,
            observation_residual,
            prior_latent_variance: self.prior_latent_variance.clone(),
            posterior_latent_variance: self.posterior_latent_variance.clone(),
            derived_prior_variances: self.derived_prior_variances.clone(),
            derived_posterior_variances: self.derived_posterior_variances.clone(),
        })
    }
}

fn estimate_rbmc_decomposition(
    precision: &GmrfSparseMatrix,
    operator: &SparseRowOperator,
    constraints: &GmrfDenseMatrix,
    config: RbmcConfig,
) -> Result<TransformedVarianceDecomposition, Box<dyn Error>> {
    let batch_sizes = rbmc_batch_sizes(config.num_probes, config.batch_count);
    let mut batches = Vec::with_capacity(batch_sizes.len());
    let removed_diag = transformed_constraint_correction_diag(precision, operator, constraints)?;

    for (batch_index, batch_size) in batch_sizes.iter().copied().enumerate() {
        let batch_seed = config.rng_seed.wrapping_add(
            0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((batch_index as u64).wrapping_add(1)),
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(batch_seed);
        let factor = precision.cholesky_sqrt_lower()?;
        let mut gmrf =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?
                .with_precision_sqrt(factor);
        let raw_unconstrained =
            rbmc_unconstrained_variances_batch(&mut gmrf, operator, batch_size, &mut rng)?;
        let stabilized_unconstrained = stabilize_positive_variances(&raw_unconstrained);
        let constrained = stabilize_constrained_variances(&stabilized_unconstrained, &removed_diag);
        batches.push(TransformedVarianceDecomposition {
            unconstrained_diag: stabilized_unconstrained,
            constrained_diag: constrained,
            removed_diag: removed_diag.clone(),
        });
    }

    weighted_average_decompositions(&batches, &batch_sizes)
        .map_err(|err| -> Box<dyn Error> { err.into() })
}

fn exact_or_rbmc_decomposition(
    gmrf: &mut Gmrf,
    precision: &GmrfSparseMatrix,
    operator: &SparseRowOperator,
    constraints: &GmrfDenseMatrix,
    rbmc: RbmcConfig,
) -> Result<TransformedVarianceDecomposition, Box<dyn Error>> {
    match gmrf.exact_transformed_variance_decomposition(operator, constraints) {
        Ok(exact) => Ok(exact),
        Err(GmrfError::NumericalInstability(
            "removed transformed marginal variance exceeded unconstrained variance",
        )) => estimate_rbmc_decomposition(precision, operator, constraints, rbmc),
        Err(err) => Err(err.into()),
    }
}

fn rbmc_unconstrained_variances_batch(
    gmrf: &mut Gmrf,
    operator: &SparseRowOperator,
    num_samples: usize,
    rng: &mut rand::rngs::StdRng,
) -> Result<GmrfVector, GmrfError> {
    if num_samples == 0 {
        return Err(GmrfError::DimensionMismatch(
            "at least one RBMC probe is required",
        ));
    }

    let output_dim = operator.nrows();
    let mut variances = GmrfVector::zeros(output_dim);
    for _ in 0..num_samples {
        let probe = GmrfVector::from_fn(output_dim, |_| rng.sample(rand_distr::StandardNormal));
        let rhs = operator.apply_transpose(&probe)?;
        let solved = gmrf.solve_precision(&rhs)?;
        let projected = operator.apply(&solved)?;
        variances += projected.component_mul(&probe);
    }

    Ok(variances / (num_samples as f64))
}

fn transformed_constraint_correction_diag(
    precision: &GmrfSparseMatrix,
    operator: &SparseRowOperator,
    constraints: &GmrfDenseMatrix,
) -> Result<GmrfVector, Box<dyn Error>> {
    if constraints.nrows() == 0 {
        return Ok(GmrfVector::zeros(operator.nrows()));
    }

    let factor = precision.cholesky_sqrt_lower()?;
    let mut gmrf =
        Gmrf::from_mean_and_precision(GmrfVector::zeros(precision.nrows()), precision.clone())?
            .with_precision_sqrt(factor);
    let covariance_times_constraint_t = covariance_times_constraint_t(&mut gmrf, constraints)?;
    let schur = schur_complement(constraints, &covariance_times_constraint_t);
    let schur_inverse = invert_spd_dense(&schur)?;

    Ok(GmrfVector::from_iterator(
        operator.nrows(),
        operator.rows.iter().map(|row| {
            let mut g = GmrfVector::zeros(schur_inverse.nrows());
            for (constraint_idx, column) in covariance_times_constraint_t
                .as_ref()
                .col_iter()
                .enumerate()
            {
                let column = column
                    .try_as_col_major()
                    .expect("dense matrix is column-major");
                g[constraint_idx] = row
                    .iter()
                    .map(|(state_idx, value)| *value * column.as_slice()[*state_idx])
                    .sum::<f64>();
            }
            quadratic_form_dense(&schur_inverse, &g)
        }),
    ))
}

fn covariance_times_constraint_t(
    gmrf: &mut Gmrf,
    constraint_matrix: &GmrfDenseMatrix,
) -> Result<GmrfDenseMatrix, GmrfError> {
    let state_dim = gmrf.dimension();
    let constraint_dim = constraint_matrix.nrows();
    let mut columns = Vec::with_capacity(constraint_dim);
    for row in 0..constraint_dim {
        let rhs = dense_row_as_vector(constraint_matrix, row);
        let solved = gmrf.solve_precision(&rhs)?;
        columns.push(solved);
    }

    Ok(GmrfDenseMatrix::from_fn(
        state_dim,
        constraint_dim,
        |i, j| columns[j][i],
    ))
}

fn dense_row_as_vector(matrix: &GmrfDenseMatrix, row: usize) -> GmrfVector {
    let mut out = GmrfVector::zeros(matrix.ncols());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        out[j] = col.as_slice()[row];
    }
    out
}

fn schur_complement(
    constraint_matrix: &GmrfDenseMatrix,
    covariance_times_constraint_t: &GmrfDenseMatrix,
) -> GmrfDenseMatrix {
    let mut schur = GmrfDenseMatrix::zeros(constraint_matrix.nrows(), constraint_matrix.nrows());
    for i in 0..constraint_matrix.nrows() {
        for j in 0..constraint_matrix.nrows() {
            let mut sum = 0.0;
            for k in 0..constraint_matrix.ncols() {
                sum += constraint_matrix[(i, k)] * covariance_times_constraint_t[(k, j)];
            }
            schur[(i, j)] = sum;
        }
    }
    schur
}

fn invert_spd_dense(matrix: &GmrfDenseMatrix) -> Result<GmrfDenseMatrix, GmrfError> {
    let factor = matrix
        .clone()
        .llt(Side::Lower)
        .map_err(|_| GmrfError::SingularConstraintSystem)?;
    let dim = matrix.nrows();
    let mut columns = Vec::with_capacity(dim);
    for j in 0..dim {
        let mut rhs = GmrfVector::zeros(dim);
        rhs[j] = 1.0;
        factor.solve_in_place(rhs.as_col_mut().as_mat_mut());
        columns.push(rhs);
    }
    Ok(GmrfDenseMatrix::from_fn(dim, dim, |i, j| columns[j][i]))
}

fn dense_matvec(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> GmrfVector {
    let mut out = GmrfVector::zeros(matrix.nrows());
    for (j, col) in matrix.as_ref().col_iter().enumerate() {
        let xj = vector[j];
        if xj == 0.0 {
            continue;
        }
        let col = col
            .try_as_col_major()
            .expect("dense matrix is column-major");
        for (i, value) in col.as_slice().iter().enumerate() {
            out[i] += *value * xj;
        }
    }
    out
}

fn quadratic_form_dense(matrix: &GmrfDenseMatrix, vector: &GmrfVector) -> f64 {
    let applied = dense_matvec(matrix, vector);
    vector.dot(&applied)
}

fn rbmc_batch_sizes(num_probes: usize, batch_count: usize) -> Vec<usize> {
    let batches = batch_count.min(num_probes);
    let base = num_probes / batches;
    let remainder = num_probes % batches;
    (0..batches)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn stabilize_decomposition(
    decomposition: &TransformedVarianceDecomposition,
) -> TransformedVarianceDecomposition {
    let unconstrained = stabilize_positive_variances(&decomposition.unconstrained_diag);
    let removed = stabilize_positive_variances(&decomposition.removed_diag);
    let constrained = stabilize_constrained_variances(&unconstrained, &removed);

    TransformedVarianceDecomposition {
        unconstrained_diag: unconstrained,
        constrained_diag: constrained,
        removed_diag: removed,
    }
}

fn stabilize_positive_variances(variances: &GmrfVector) -> GmrfVector {
    let positive_sum = variances
        .iter()
        .copied()
        .filter(|value| *value > EPS)
        .sum::<f64>();
    let positive_count = variances.iter().filter(|value| **value > EPS).count();
    let positive_mean = if positive_count > 0 {
        positive_sum / positive_count as f64
    } else {
        1.0
    };
    let floor = positive_mean.abs().max(1.0) * 1e-12;

    GmrfVector::from_iterator(
        variances.len(),
        variances.iter().copied().map(|value| value.max(floor)),
    )
}

fn stabilize_constrained_variances(unconstrained: &GmrfVector, removed: &GmrfVector) -> GmrfVector {
    let positive_sum = unconstrained
        .iter()
        .copied()
        .filter(|value| *value > EPS)
        .sum::<f64>();
    let positive_count = unconstrained.iter().filter(|value| **value > EPS).count();
    let positive_mean = if positive_count > 0 {
        positive_sum / positive_count as f64
    } else {
        1.0
    };
    let floor = positive_mean.abs().max(1.0) * 1e-12;

    GmrfVector::from_iterator(
        unconstrained.len(),
        (0..unconstrained.len()).map(|i| (unconstrained[i] - removed[i]).max(floor)),
    )
}

fn weighted_average_decompositions(
    values: &[TransformedVarianceDecomposition],
    weights: &[usize],
) -> Result<TransformedVarianceDecomposition, &'static str> {
    if values.is_empty() || values.len() != weights.len() {
        return Err("RBMC batches and weights must be non-empty and aligned");
    }
    let dim = values[0].unconstrained_diag.len();
    if values.iter().any(|value| {
        value.unconstrained_diag.len() != dim
            || value.constrained_diag.len() != dim
            || value.removed_diag.len() != dim
    }) {
        return Err("RBMC decomposition batch dimensions must match");
    }
    let total_weight = weights.iter().sum::<usize>() as f64;

    let average = |extract: fn(&TransformedVarianceDecomposition) -> &GmrfVector| {
        let mut out = GmrfVector::zeros(dim);
        for (value, weight) in values.iter().zip(weights.iter().copied()) {
            let scale = weight as f64 / total_weight;
            let component = extract(value);
            for i in 0..dim {
                out[i] += scale * component[i];
            }
        }
        out
    };

    Ok(TransformedVarianceDecomposition {
        unconstrained_diag: average(|value| &value.unconstrained_diag),
        constrained_diag: average(|value| &value.constrained_diag),
        removed_diag: average(|value| &value.removed_diag),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gmrf_core::observation::ht_weighted_observations;
    use gmrf_core::types::CooMatrix;

    fn assert_sparse_matrix_eq(left: &GmrfSparseMatrix, right: &GmrfSparseMatrix) {
        assert_eq!(left.nrows(), right.nrows());
        assert_eq!(left.ncols(), right.ncols());

        let mut left_entries = left
            .triplet_iter()
            .map(|(row, col, value)| (row, col, *value))
            .collect::<Vec<_>>();
        let mut right_entries = right
            .triplet_iter()
            .map(|(row, col, value)| (row, col, *value))
            .collect::<Vec<_>>();
        left_entries.sort_by(|a, b| a.partial_cmp(b).expect("finite sparse entries"));
        right_entries.sort_by(|a, b| a.partial_cmp(b).expect("finite sparse entries"));
        assert_eq!(left_entries, right_entries);
    }

    fn assert_decomposition_eq(
        left: &TransformedVarianceDecomposition,
        right: &TransformedVarianceDecomposition,
    ) {
        assert_eq!(left.unconstrained_diag, right.unconstrained_diag);
        assert_eq!(left.constrained_diag, right.constrained_diag);
        assert_eq!(left.removed_diag, right.removed_diag);
    }

    fn test_precision() -> GmrfSparseMatrix {
        let mut coo = CooMatrix::new(3, 3);
        coo.push(0, 0, 5.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 1.0);
        coo.push(1, 1, 4.0);
        coo.push(1, 2, 0.5);
        coo.push(2, 1, 0.5);
        coo.push(2, 2, 3.0);
        GmrfSparseMatrix::from(&coo)
    }

    fn test_observation_operator() -> GmrfSparseMatrix {
        let mut coo = CooMatrix::new(2, 3);
        coo.push(0, 0, 1.0);
        coo.push(1, 2, 1.0);
        GmrfSparseMatrix::from(&coo)
    }

    fn test_problem() -> LinearGaussianConditioningProblem {
        let mut derived_operators = DerivedOperatorSet::new();
        derived_operators.insert(
            "identity".to_string(),
            DerivedOperator {
                operator: SparseRowOperator::identity(3),
                variance_mode: DerivedVarianceMode::Exact,
            },
        );
        derived_operators.insert(
            "sum".to_string(),
            DerivedOperator {
                operator: SparseRowOperator::new(3, vec![vec![(0, 1.0), (1, -0.5)]])
                    .expect("valid sparse row operator"),
                variance_mode: DerivedVarianceMode::Exact,
            },
        );

        LinearGaussianConditioningProblem {
            prior_precision: test_precision(),
            observation_operator: test_observation_operator(),
            observations: GmrfVector::from_vec(vec![0.75, -0.25]),
            noise_variance: 0.2,
            harmonic_subspace: Some(HarmonicSubspace {
                basis: FeecMatrix::zeros(3, 0),
                constraints: GmrfDenseMatrix::from_fn(1, 3, |_, j| if j == 0 { 1.0 } else { 0.0 }),
                projector: None,
            }),
            derived_operators,
            rbmc: RbmcConfig {
                num_probes: 24,
                batch_count: 4,
                rng_seed: 7,
            },
        }
    }

    #[test]
    fn solve_matches_direct_gaussian_update_and_constraint_projection() {
        let problem = test_problem();
        let result = problem.solve().expect("conditioning should succeed");

        let information = ht_weighted_observations(
            &problem.observation_operator,
            &problem.observations,
            1.0 / problem.noise_variance,
        );
        let (posterior_precision, _) = apply_gaussian_observations(
            &problem.prior_precision,
            &problem.observation_operator,
            &problem.observations,
            None,
            problem.noise_variance,
        );
        let mut posterior =
            Gmrf::from_information_and_precision(information.clone(), posterior_precision.clone())
                .expect("posterior should build");

        assert_eq!(result.information, information);
        assert_sparse_matrix_eq(&result.posterior_precision, &posterior_precision);
        assert_eq!(result.posterior_mean, posterior.mean().clone());

        let constrained = posterior
            .constrained_mean(
                &problem
                    .harmonic_subspace
                    .as_ref()
                    .expect("harmonic subspace")
                    .constraints,
                &GmrfVector::zeros(1),
            )
            .expect("constrained mean should succeed");
        assert_eq!(
            result.constrained_posterior_mean,
            Some(constrained),
            "shared conditioning core should reuse gmrf-core constrained_mean",
        );
    }

    #[test]
    fn prepare_reuses_precision_and_variances_across_observation_vectors() {
        let problem = test_problem();
        let prepared = problem.prepare().expect("problem should prepare");

        let first = prepared
            .solve_with_observations(&GmrfVector::from_vec(vec![0.75, -0.25]))
            .expect("first solve should succeed");
        let second = prepared
            .solve_with_observations(&GmrfVector::from_vec(vec![-0.25, 0.5]))
            .expect("second solve should succeed");

        assert_sparse_matrix_eq(&first.posterior_precision, &second.posterior_precision);
        assert_eq!(
            first.prior_latent_variance.unconstrained_diag,
            second.prior_latent_variance.unconstrained_diag
        );
        assert_eq!(
            first.posterior_latent_variance.constrained_diag,
            second.posterior_latent_variance.constrained_diag
        );
        assert_eq!(
            first.derived_prior_variances.len(),
            second.derived_prior_variances.len()
        );
        for (name, first_decomposition) in &first.derived_prior_variances {
            assert_decomposition_eq(first_decomposition, &second.derived_prior_variances[name]);
        }
        assert_eq!(
            first.derived_posterior_variances.len(),
            second.derived_posterior_variances.len()
        );
        for (name, first_decomposition) in &first.derived_posterior_variances {
            assert_decomposition_eq(
                first_decomposition,
                &second.derived_posterior_variances[name],
            );
        }
        assert_ne!(first.posterior_mean, second.posterior_mean);
    }

    #[test]
    fn exact_derived_variances_match_direct_gmrf_computation() {
        let problem = test_problem();
        let result = problem.solve().expect("conditioning should succeed");
        let constraints = &problem
            .harmonic_subspace
            .as_ref()
            .expect("harmonic subspace")
            .constraints;

        let mut prior =
            Gmrf::from_mean_and_precision(GmrfVector::zeros(3), problem.prior_precision.clone())
                .expect("prior should build");
        let expected_prior = prior
            .exact_transformed_variance_decomposition(
                &problem.derived_operators["sum"].operator,
                constraints,
            )
            .expect("prior transformed variances should succeed");

        let mut posterior = Gmrf::from_information_and_precision(
            result.information.clone(),
            result.posterior_precision.clone(),
        )
        .expect("posterior should build");
        let expected_posterior = posterior
            .exact_transformed_variance_decomposition(
                &problem.derived_operators["sum"].operator,
                constraints,
            )
            .expect("posterior transformed variances should succeed");

        assert_decomposition_eq(&result.derived_prior_variances["sum"], &expected_prior);
        assert_decomposition_eq(
            &result.derived_posterior_variances["sum"],
            &expected_posterior,
        );
    }
}
