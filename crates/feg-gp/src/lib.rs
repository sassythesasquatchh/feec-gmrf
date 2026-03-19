//! Spectral Gaussian process utilities based on Hodge Laplacian eigenfunctions.

use common::linalg::nalgebra::{
    quadratic_form_sparse, CsrMatrix, Matrix as NaMatrix, Vector as NaVector,
};
use common::linalg::petsc::{GhiepReducedSolve, GhiepWhich};
use exterior::ExteriorGrade;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::Solve;
use faer::{Accum, Idx, Mat, Par, Side, Unbind};
use formoniq::assemble;
use formoniq::operators::HodgeMassElmat;
use formoniq::problems::{hodge_laplace, laplace_beltrami};
use libm::tgamma;
use manifold::geometry::metric::mesh::MeshLengths;
use manifold::topology::complex::Complex;
use std::cmp::Ordering;
use thiserror::Error;

pub const DEFAULT_K: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct SpectralMaternConfig {
    pub kappa: f64,
    pub alpha: f64,
    pub tau: f64,
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EuclideanMaternConfig {
    pub kappa: f64,
    pub nu: f64,
    pub variance: f64,
}

impl Default for SpectralMaternConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: DEFAULT_K,
        }
    }
}

impl Default for EuclideanMaternConfig {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            nu: 1.5,
            variance: 1.0,
        }
    }
}

#[derive(Debug, Error)]
pub enum GpError {
    #[error("k must be positive")]
    InvalidK,
    #[error("kappa must be positive")]
    InvalidKappa,
    #[error("nu must be positive")]
    InvalidNu,
    #[error("tau must be positive")]
    InvalidTau,
    #[error("alpha must be positive")]
    InvalidAlpha,
    #[error("variance must be non-negative")]
    InvalidVariance,
    #[error("points slice is empty")]
    EmptyPoints,
    #[error("points must have at least one dimension")]
    EmptyPointDimension,
    #[error("point dimension mismatch: expected {expected}, got {got}")]
    PointDimensionMismatch { expected: usize, got: usize },
    #[error("distance must be finite and non-negative")]
    InvalidDistance,
    #[error("unsupported nu {nu}; supported values are 0.5, 1.0, 1.5, 2.5")]
    UnsupportedNu { nu: f64 },
    #[error("eigenvalues length {values} does not match eigenvector columns {cols}")]
    EigenDimensionMismatch { values: usize, cols: usize },
    #[error("eigenvectors must have at least one row")]
    EmptyEigenvectors,
    #[error("non-finite eigenvalue {value}")]
    NonFiniteEigenvalue { value: f64 },
    #[error("invalid eigenvalue {value}: kappa^2 + lambda must be positive")]
    InvalidEigenvalue { value: f64 },
    #[error("invalid eigenvector mass norm {value}")]
    InvalidEigenvectorNorm { value: f64 },
    #[error("covariance must be square, got {rows}x{cols}")]
    InvalidCovarianceShape { rows: usize, cols: usize },
    #[error("observation indices length {indices} does not match values length {values}")]
    ObservationCountMismatch { indices: usize, values: usize },
    #[error("observation index {index} out of bounds for size {n}")]
    ObservationIndexOutOfBounds { index: usize, n: usize },
    #[error("observation index {index} is duplicated")]
    ObservationIndexRepeated { index: usize },
    #[error("noise variance must be finite and non-negative")]
    InvalidNoiseVariance,
    #[error("covariance is not positive definite (cholesky failed)")]
    CholeskyFailed,
    #[error("standard normal length {got} does not match basis size {expected}")]
    SampleDimensionMismatch { expected: usize, got: usize },
}

#[derive(Debug, Clone)]
pub struct EigenBasis {
    eigenvalues: Vec<f64>,
    eigenvectors: Mat<f64>,
}

impl EigenBasis {
    pub fn new(eigenvalues: Vec<f64>, eigenvectors: Mat<f64>) -> Result<Self, GpError> {
        if eigenvalues.is_empty() {
            return Err(GpError::InvalidK);
        }
        if eigenvectors.nrows() == 0 {
            return Err(GpError::EmptyEigenvectors);
        }
        if eigenvalues.len() != eigenvectors.ncols() {
            return Err(GpError::EigenDimensionMismatch {
                values: eigenvalues.len(),
                cols: eigenvectors.ncols(),
            });
        }
        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    pub fn from_hodge_laplace_eigenpairs(
        eigenvalues: NaVector,
        eigenvectors: NaMatrix,
    ) -> Result<Self, GpError> {
        let values = eigenvalues.iter().copied().collect::<Vec<_>>();
        let vectors = nalgebra_matrix_to_faer(&eigenvectors);
        Self::new(values, vectors)
    }

    pub fn nrows(&self) -> usize {
        self.eigenvectors.nrows()
    }

    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn eigenvectors(&self) -> &Mat<f64> {
        &self.eigenvectors
    }

    pub fn truncate_largest(&self, k: usize) -> Result<Self, GpError> {
        if k == 0 {
            return Err(GpError::InvalidK);
        }
        let keep = k.min(self.eigenvalues.len());
        let mut indices: Vec<usize> = (0..self.eigenvalues.len()).collect();
        indices.sort_by(|&a, &b| {
            let left = self.eigenvalues[a];
            let right = self.eigenvalues[b];
            right.partial_cmp(&left).unwrap_or(Ordering::Equal)
        });

        let mut values = Vec::with_capacity(keep);
        let mut vectors = Mat::zeros(self.eigenvectors.nrows(), keep);

        for (new_col, &old_col) in indices.iter().take(keep).enumerate() {
            let val = self.eigenvalues[old_col];
            if !val.is_finite() {
                return Err(GpError::NonFiniteEigenvalue { value: val });
            }
            values.push(val);
            for row in 0..self.eigenvectors.nrows() {
                vectors[(row, new_col)] = self.eigenvectors[(row, old_col)];
            }
        }

        Ok(Self {
            eigenvalues: values,
            eigenvectors: vectors,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SpectralMaternGp {
    basis: EigenBasis,
    config: SpectralMaternConfig,
    weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConditionedGp {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConditionedFullGp {
    pub mean: Vec<f64>,
    pub covariance: Mat<f64>,
}

impl SpectralMaternGp {
    pub fn from_eigenbasis(
        basis: EigenBasis,
        config: SpectralMaternConfig,
    ) -> Result<Self, GpError> {
        validate_config(config)?;
        let basis = basis.truncate_largest(config.k)?;
        let mut weights = Vec::with_capacity(basis.len());
        for &lambda in basis.eigenvalues() {
            weights.push(matern_weight(lambda, config)?);
        }
        Ok(Self {
            basis,
            config,
            weights,
        })
    }

    pub fn from_hodge_laplace(
        topology: &Complex,
        geometry: &MeshLengths,
        grade: ExteriorGrade,
        config: SpectralMaternConfig,
    ) -> Result<Self, GpError> {
        validate_config(config)?;

        let (eigenvals, mut eigenvecs) = if grade == 0 {
            let (eigenvalues, eigen_us) = laplace_beltrami::solve_laplace_beltrami_evp_as_matrix(
                &topology, &geometry, config.k,
            );
            (eigenvalues, eigen_us)
        } else {
            let (eigenvals, _eigen_sigmas, eigen_us) =
                hodge_laplace::solve_hodge_laplace_evp_config(
                    &topology,
                    &geometry,
                    grade,
                    config.k,
                    GhiepWhich::Smallest,
                    GhiepReducedSolve::Direct,
                );
            (eigenvals, eigen_us)
        };
        // TODO put inside feec library
        let mass_galmat = assemble::assemble_galmat(
            topology,
            geometry,
            HodgeMassElmat::new(topology.dim(), grade),
        );
        let mass = CsrMatrix::from(&mass_galmat);
        mass_normalize_eigenvectors(&mass, &mut eigenvecs)?;

        let basis = EigenBasis::from_hodge_laplace_eigenpairs(eigenvals, eigenvecs)?;
        Self::from_eigenbasis(basis, config)
    }

    pub fn k(&self) -> usize {
        self.basis.len()
    }

    pub fn eigenvalues(&self) -> &[f64] {
        self.basis.eigenvalues()
    }

    pub fn eigenvectors(&self) -> &Mat<f64> {
        self.basis.eigenvectors()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn covariance_matrix(&self) -> Mat<f64> {
        let n = self.basis.eigenvectors.nrows();
        let k = self.basis.eigenvectors.ncols();
        let mut scaled = self.basis.eigenvectors.clone();
        for col in 0..k {
            let weight = self.weights[col];
            if weight == 1.0 {
                continue;
            }
            for row in 0..n {
                scaled[(row, col)] *= weight;
            }
        }

        let mut cov = Mat::zeros(n, n);
        matmul(
            &mut cov,
            Accum::Replace,
            scaled.as_ref(),
            self.basis.eigenvectors.as_ref().transpose(),
            1.0,
            Par::Seq,
        );
        cov
    }

    pub fn covariance_entry(&self, i: usize, j: usize) -> f64 {
        let mut acc = 0.0;
        for (col, &weight) in self.weights.iter().enumerate() {
            acc += weight * self.basis.eigenvectors[(i, col)] * self.basis.eigenvectors[(j, col)];
        }
        acc
    }

    pub fn config(&self) -> SpectralMaternConfig {
        self.config
    }

    pub fn sample_from_standard_normal(&self, z: &[f64]) -> Result<Vec<f64>, GpError> {
        let k = self.weights.len();
        if z.len() != k {
            return Err(GpError::SampleDimensionMismatch {
                expected: k,
                got: z.len(),
            });
        }
        let n = self.basis.eigenvectors.nrows();
        let mut sample = vec![0.0; n];
        for (col, &weight) in self.weights.iter().enumerate() {
            let scale = weight.max(0.0).sqrt();
            if scale == 0.0 {
                continue;
            }
            let zc = z[col];
            for row in 0..n {
                sample[row] += scale * self.basis.eigenvectors[(row, col)] * zc;
            }
        }
        Ok(sample)
    }
}

pub fn matern_covariance_euclidean(
    distance: f64,
    config: EuclideanMaternConfig,
) -> Result<f64, GpError> {
    validate_euclidean_config(config)?;
    if !distance.is_finite() || distance < 0.0 {
        return Err(GpError::InvalidDistance);
    }
    if distance == 0.0 {
        return Ok(config.variance);
    }
    let x = config.kappa * distance;
    if let Some(val) = half_integer_kernel(x, config.variance, config.nu) {
        return Ok(val);
    }
    if (config.nu - 1.0).abs() <= 1e-12 {
        let prefactor = 2.0_f64.powf(1.0 - config.nu) / tgamma(config.nu);
        let value = config.variance * prefactor * x.powf(config.nu) * bessel_k1(x);
        return Ok(value);
    }
    Err(GpError::UnsupportedNu { nu: config.nu })
}

pub fn matern_covariance_matrix_euclidean(
    points: &[Vec<f64>],
    config: EuclideanMaternConfig,
) -> Result<Mat<f64>, GpError> {
    validate_euclidean_config(config)?;
    if points.is_empty() {
        return Err(GpError::EmptyPoints);
    }
    let dim = points[0].len();
    if dim == 0 {
        return Err(GpError::EmptyPointDimension);
    }
    if !supports_matern_nu(config.nu) {
        return Err(GpError::UnsupportedNu { nu: config.nu });
    }
    if points[0].iter().any(|v| !v.is_finite()) {
        return Err(GpError::InvalidDistance);
    }
    for point in points.iter().skip(1) {
        if point.len() != dim {
            return Err(GpError::PointDimensionMismatch {
                expected: dim,
                got: point.len(),
            });
        }
        if point.iter().any(|v| !v.is_finite()) {
            return Err(GpError::InvalidDistance);
        }
    }

    let n = points.len();
    let mut cov = Mat::zeros(n, n);
    for i in 0..n {
        cov[(i, i)] = config.variance;
        for j in (i + 1)..n {
            let dist = euclidean_distance(&points[i], &points[j])?;
            let x = config.kappa * dist;
            let value = if let Some(val) = half_integer_kernel(x, config.variance, config.nu) {
                val
            } else if (config.nu - 1.0).abs() <= 1e-12 {
                let prefactor = 2.0_f64.powf(1.0 - config.nu) / tgamma(config.nu);
                config.variance * prefactor * x.powf(config.nu) * bessel_k1(x)
            } else {
                return Err(GpError::UnsupportedNu { nu: config.nu });
            };
            cov[(i, j)] = value;
            cov[(j, i)] = value;
        }
    }
    Ok(cov)
}

pub fn condition_full_covariance(
    cov: &Mat<f64>,
    obs_indices: &[usize],
    obs_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedGp, GpError> {
    if cov.nrows() != cov.ncols() {
        return Err(GpError::InvalidCovarianceShape {
            rows: cov.nrows(),
            cols: cov.ncols(),
        });
    }
    if obs_indices.len() != obs_values.len() {
        return Err(GpError::ObservationCountMismatch {
            indices: obs_indices.len(),
            values: obs_values.len(),
        });
    }
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(GpError::InvalidNoiseVariance);
    }

    let n = cov.nrows();
    let m = obs_indices.len();
    let mut seen = vec![false; n];
    for &idx in obs_indices {
        if idx >= n {
            return Err(GpError::ObservationIndexOutOfBounds { index: idx, n });
        }
        if seen[idx] {
            return Err(GpError::ObservationIndexRepeated { index: idx });
        }
        seen[idx] = true;
    }

    let observations = Mat::from_fn(m, 1, |i, _| obs_values[i.unbound()]);

    let mut k_oo = Mat::from_fn(m, m, |i, j| {
        let row = obs_indices[i.unbound()];
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });
    for i in 0..m {
        let ii = idx(i);
        k_oo[(ii, ii)] += noise_variance;
    }

    let chol = k_oo.llt(Side::Lower).map_err(|_| GpError::CholeskyFailed)?;
    let mut alpha = observations.clone();
    chol.solve_in_place(alpha.as_mut());

    let k_star_o = Mat::from_fn(n, m, |i, j| {
        let row = i.unbound();
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });

    let mut posterior_mean = Mat::zeros(n, 1);
    matmul(
        &mut posterior_mean,
        Accum::Replace,
        k_star_o.as_ref(),
        alpha.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut w = k_star_o.transpose().to_owned();
    chol.solve_in_place(w.as_mut());

    let mut posterior_var = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..m {
            acc += k_star_o[(idx(i), idx(j))] * w[(idx(j), idx(i))];
        }
        let prior_var = cov[(idx(i), idx(i))];
        let var = (prior_var - acc).max(0.0);
        posterior_var.push(var);
    }

    Ok(ConditionedGp {
        mean: mat_col_to_vec(&posterior_mean),
        variance: posterior_var,
    })
}

pub fn condition_full_covariance_with_covariance(
    cov: &Mat<f64>,
    obs_indices: &[usize],
    obs_values: &[f64],
    noise_variance: f64,
) -> Result<ConditionedFullGp, GpError> {
    if cov.nrows() != cov.ncols() {
        return Err(GpError::InvalidCovarianceShape {
            rows: cov.nrows(),
            cols: cov.ncols(),
        });
    }
    if obs_indices.len() != obs_values.len() {
        return Err(GpError::ObservationCountMismatch {
            indices: obs_indices.len(),
            values: obs_values.len(),
        });
    }
    if !noise_variance.is_finite() || noise_variance < 0.0 {
        return Err(GpError::InvalidNoiseVariance);
    }

    let n = cov.nrows();
    let m = obs_indices.len();
    let mut seen = vec![false; n];
    for &idx in obs_indices {
        if idx >= n {
            return Err(GpError::ObservationIndexOutOfBounds { index: idx, n });
        }
        if seen[idx] {
            return Err(GpError::ObservationIndexRepeated { index: idx });
        }
        seen[idx] = true;
    }

    let observations = Mat::from_fn(m, 1, |i, _| obs_values[i.unbound()]);

    let mut k_oo = Mat::from_fn(m, m, |i, j| {
        let row = obs_indices[i.unbound()];
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });
    for i in 0..m {
        let ii = idx(i);
        k_oo[(ii, ii)] += noise_variance;
    }

    let chol = k_oo.llt(Side::Lower).map_err(|_| GpError::CholeskyFailed)?;
    let mut alpha = observations.clone();
    chol.solve_in_place(alpha.as_mut());

    let k_star_o = Mat::from_fn(n, m, |i, j| {
        let row = i.unbound();
        let col = obs_indices[j.unbound()];
        cov[(idx(row), idx(col))]
    });

    let mut posterior_mean = Mat::zeros(n, 1);
    matmul(
        &mut posterior_mean,
        Accum::Replace,
        k_star_o.as_ref(),
        alpha.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut w = k_star_o.transpose().to_owned();
    chol.solve_in_place(w.as_mut());

    let mut correction = Mat::zeros(n, n);
    matmul(
        &mut correction,
        Accum::Replace,
        k_star_o.as_ref(),
        w.as_ref(),
        1.0,
        Par::Seq,
    );

    let mut posterior_cov = cov.clone();
    for i in 0..n {
        for j in 0..n {
            posterior_cov[(idx(i), idx(j))] -= correction[(idx(i), idx(j))];
        }
    }

    Ok(ConditionedFullGp {
        mean: mat_col_to_vec(&posterior_mean),
        covariance: posterior_cov,
    })
}

fn validate_config(config: SpectralMaternConfig) -> Result<(), GpError> {
    if config.k == 0 {
        return Err(GpError::InvalidK);
    }
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(GpError::InvalidKappa);
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(GpError::InvalidAlpha);
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(GpError::InvalidTau);
    }
    if !config.alpha.is_finite() || config.alpha <= 0.0 {
        return Err(GpError::InvalidAlpha);
    }
    if !config.tau.is_finite() || config.tau <= 0.0 {
        return Err(GpError::InvalidTau);
    }
    Ok(())
}

fn validate_euclidean_config(config: EuclideanMaternConfig) -> Result<(), GpError> {
    if !config.kappa.is_finite() || config.kappa <= 0.0 {
        return Err(GpError::InvalidKappa);
    }
    if !config.nu.is_finite() || config.nu <= 0.0 {
        return Err(GpError::InvalidNu);
    }
    if !config.variance.is_finite() || config.variance < 0.0 {
        return Err(GpError::InvalidVariance);
    }
    Ok(())
}

fn matern_weight(lambda: f64, config: SpectralMaternConfig) -> Result<f64, GpError> {
    if !lambda.is_finite() {
        return Err(GpError::NonFiniteEigenvalue { value: lambda });
    }
    let base = config.kappa * config.kappa + lambda;
    if !base.is_finite() || base <= 0.0 {
        return Err(GpError::InvalidEigenvalue { value: lambda });
    }
    let inverse_tau = 1. / config.tau;
    Ok(inverse_tau * inverse_tau * base.powf(-config.alpha))
}

fn half_integer_kernel(x: f64, variance: f64, nu: f64) -> Option<f64> {
    let tol = 1e-12;
    let exp_term = (-x).exp();
    if (nu - 0.5).abs() <= tol {
        return Some(variance * exp_term);
    }
    if (nu - 1.5).abs() <= tol {
        return Some(variance * (1.0 + x) * exp_term);
    }
    if (nu - 2.5).abs() <= tol {
        return Some(variance * (1.0 + x + x * x / 3.0) * exp_term);
    }
    None
}

fn supports_matern_nu(nu: f64) -> bool {
    (nu - 0.5).abs() <= 1e-12
        || (nu - 1.0).abs() <= 1e-12
        || (nu - 1.5).abs() <= 1e-12
        || (nu - 2.5).abs() <= 1e-12
}

fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let mut out = if ax <= 3.75 {
        let t = ax / 3.75;
        let t2 = t * t;
        ax * (0.5
            + t2 * (0.87890594
                + t2 * (0.51498869
                    + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (-0.03988024
                    + t * (-0.00362018
                        + t * (0.00163801
                            + t * (-0.01031555
                                + t * (0.02282967
                                    + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))))
    };
    if x < 0.0 {
        out = -out;
    }
    out
}

fn bessel_k1(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x <= 2.0 {
        let t = x * x / 4.0;
        (x / 2.0).ln() * bessel_i1(x)
            + (1.0 / x)
                * (1.0
                    + t * (0.15443144
                        + t * (-0.67278579
                            + t * (-0.18156897
                                + t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))))
    } else {
        let t = 2.0 / x;
        (-(x)).exp() / x.sqrt()
            * (1.25331414
                + t * (0.23498619
                    + t * (-0.03655620
                        + t * (0.01504268
                            + t * (-0.00780353 + t * (0.00325614 - t * 0.00068245))))))
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64, GpError> {
    if a.len() != b.len() {
        return Err(GpError::PointDimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Err(GpError::EmptyPointDimension);
    }
    let mut sum = 0.0;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let diff = av - bv;
        sum += diff * diff;
    }
    Ok(sum.sqrt())
}

fn nalgebra_matrix_to_faer(matrix: &NaMatrix) -> Mat<f64> {
    let mut out = Mat::zeros(matrix.nrows(), matrix.ncols());
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            out[(i, j)] = matrix[(i, j)];
        }
    }
    out
}

fn mass_normalize_eigenvectors(
    mass: &CsrMatrix,
    eigenvectors: &mut NaMatrix,
) -> Result<(), GpError> {
    for col in 0..eigenvectors.ncols() {
        let vec = eigenvectors.column(col).into_owned();
        let norm2 = quadratic_form_sparse(mass, &vec);
        if !norm2.is_finite() || norm2 <= 0.0 {
            return Err(GpError::InvalidEigenvectorNorm { value: norm2 });
        }
        let scale = 1.0 / norm2.sqrt();
        for row in 0..eigenvectors.nrows() {
            eigenvectors[(row, col)] *= scale;
        }
    }
    Ok(())
}

#[inline]
fn idx(i: usize) -> Idx<usize> {
    unsafe { Idx::<usize>::new_unbound(i) }
}

fn mat_col_to_vec(col: &Mat<f64>) -> Vec<f64> {
    let mut out = Vec::with_capacity(col.nrows());
    for i in 0..col.nrows() {
        out.push(col[(idx(i), idx(0))]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64, tol: f64) {
        let diff = (left - right).abs();
        assert!(diff <= tol, "expected {left} to be within {tol} of {right}");
    }

    #[test]
    fn covariance_matches_weighted_identity_basis() {
        let mut eigenvectors = Mat::zeros(2, 2);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        let eigenvalues = vec![4.0, 9.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();
        let cov = gp.covariance_matrix();

        let w0 = matern_weight(4.0, config).unwrap();
        let w1 = matern_weight(9.0, config).unwrap();
        assert_close(cov[(0, 0)], w0, 1e-12);
        assert_close(cov[(1, 1)], w1, 1e-12);
        assert_close(cov[(0, 1)], 0.0, 1e-12);
        assert_close(cov[(1, 0)], 0.0, 1e-12);
    }

    #[test]
    fn truncation_keeps_largest_eigenvalues() {
        let mut eigenvectors = Mat::zeros(3, 3);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        eigenvectors[(2, 2)] = 1.0;
        let eigenvalues = vec![1.0, 10.0, 5.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        assert_eq!(gp.k(), 2);
        assert!(gp.eigenvalues()[0] >= gp.eigenvalues()[1]);
        assert_close(gp.eigenvalues()[0], 10.0, 1e-12);
        assert_close(gp.eigenvalues()[1], 5.0, 1e-12);
    }

    #[test]
    fn conditioning_matches_small_covariance() {
        let mut cov = Mat::zeros(2, 2);
        cov[(idx(0), idx(0))] = 1.0;
        cov[(idx(1), idx(1))] = 1.0;
        cov[(idx(0), idx(1))] = 0.5;
        cov[(idx(1), idx(0))] = 0.5;

        let obs_indices = vec![0];
        let obs_values = vec![1.0];
        let conditioned = condition_full_covariance(&cov, &obs_indices, &obs_values, 0.0).unwrap();

        assert_close(conditioned.mean[0], 1.0, 1e-12);
        assert_close(conditioned.mean[1], 0.5, 1e-12);
        assert_close(conditioned.variance[0], 0.0, 1e-12);
        assert_close(conditioned.variance[1], 0.75, 1e-12);
    }

    #[test]
    fn sample_from_standard_normal_matches_identity_basis() {
        let mut eigenvectors = Mat::zeros(2, 2);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 1)] = 1.0;
        let eigenvalues = vec![3.0, 1.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 2,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        let z = vec![2.0, -4.0];
        let sample = gp.sample_from_standard_normal(&z).unwrap();

        let w0 = matern_weight(3.0, config).unwrap();
        let w1 = matern_weight(1.0, config).unwrap();
        assert_close(sample[0], w0.sqrt() * z[0], 1e-12);
        assert_close(sample[1], w1.sqrt() * z[1], 1e-12);
    }

    #[test]
    fn sample_from_standard_normal_rejects_mismatch() {
        let mut eigenvectors = Mat::zeros(2, 1);
        eigenvectors[(0, 0)] = 1.0;
        eigenvectors[(1, 0)] = 1.0;
        let eigenvalues = vec![1.0];

        let basis = EigenBasis::new(eigenvalues, eigenvectors).unwrap();
        let config = SpectralMaternConfig {
            kappa: 1.0,
            alpha: 2.0,
            tau: 1.0,
            k: 1,
        };
        let gp = SpectralMaternGp::from_eigenbasis(basis, config).unwrap();

        let err = gp.sample_from_standard_normal(&[0.0, 1.0]).unwrap_err();
        match err {
            GpError::SampleDimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn euclidean_matern_half_integer_matches_closed_form() {
        let config = EuclideanMaternConfig {
            kappa: 1.3,
            nu: 0.5,
            variance: 2.5,
        };
        let r = 0.7;
        let expected = config.variance * (-config.kappa * r).exp();
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-12);

        let config = EuclideanMaternConfig {
            kappa: 0.8,
            nu: 1.5,
            variance: 1.2,
        };
        let r = 1.1;
        let x = config.kappa * r;
        let expected = config.variance * (1.0 + x) * (-x).exp();
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-12);
    }

    #[test]
    fn euclidean_covariance_matrix_is_symmetric() {
        let points = vec![vec![0.0], vec![1.0], vec![2.0]];
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 2.5,
            variance: 0.7,
        };
        let cov = matern_covariance_matrix_euclidean(&points, config).unwrap();
        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);
        assert_close(cov[(0, 0)], config.variance, 1e-12);
        assert_close(cov[(1, 1)], config.variance, 1e-12);
        assert_close(cov[(2, 2)], config.variance, 1e-12);
        assert_close(cov[(0, 1)], cov[(1, 0)], 1e-12);
        assert_close(cov[(0, 2)], cov[(2, 0)], 1e-12);
        assert_close(cov[(1, 2)], cov[(2, 1)], 1e-12);
    }

    #[test]
    fn conditioning_full_covariance_produces_full_posterior() {
        let mut cov = Mat::zeros(2, 2);
        cov[(idx(0), idx(0))] = 1.0;
        cov[(idx(1), idx(1))] = 1.0;
        cov[(idx(0), idx(1))] = 0.5;
        cov[(idx(1), idx(0))] = 0.5;

        let obs_indices = vec![0];
        let obs_values = vec![1.0];
        let conditioned =
            condition_full_covariance_with_covariance(&cov, &obs_indices, &obs_values, 0.0)
                .unwrap();

        assert_close(conditioned.mean[0], 1.0, 1e-12);
        assert_close(conditioned.mean[1], 0.5, 1e-12);
        assert_close(conditioned.covariance[(idx(0), idx(0))], 0.0, 1e-12);
        assert_close(conditioned.covariance[(idx(1), idx(1))], 0.75, 1e-12);
    }

    #[test]
    fn euclidean_matern_rejects_unsupported_nu() {
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 0.7,
            variance: 1.0,
        };
        let err = matern_covariance_euclidean(0.3, config).unwrap_err();
        match err {
            GpError::UnsupportedNu { nu } => assert_close(nu, 0.7, 1e-12),
            _ => panic!("unexpected error: {err:?}"),
        }
    }

    #[test]
    fn euclidean_matern_nu_one_uses_bessel_k1() {
        let config = EuclideanMaternConfig {
            kappa: 1.0,
            nu: 1.0,
            variance: 2.0,
        };
        let r = 1.0;
        let expected = 2.0 * 1.0 * bessel_k1(1.0);
        let got = matern_covariance_euclidean(r, config).unwrap();
        assert_close(got, expected, 1e-6);
    }

    #[test]
    fn mass_normalization_scales_columns_to_unit_mass() {
        use common::linalg::nalgebra::{CooMatrix, Vector as NaVector};

        let mass =
            CooMatrix::try_from_triplets(2, 2, vec![0, 1], vec![0, 1], vec![2.0, 8.0]).unwrap();
        let mass = CsrMatrix::from(&mass);

        let mut eigenvectors = NaMatrix::from_column_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        mass_normalize_eigenvectors(&mass, &mut eigenvectors).unwrap();

        for col in 0..eigenvectors.ncols() {
            let vec = NaVector::from_column_slice(eigenvectors.column(col).as_slice());
            let norm2 = quadratic_form_sparse(&mass, &vec);
            assert_close(norm2, 1.0, 1e-12);
        }
    }
}
