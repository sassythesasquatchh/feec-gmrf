use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use gmrf_core::types::{CooMatrix as GmrfCoo, SparseMatrix as GmrfSparse, Vector as GmrfVector};

pub fn feec_csr_to_gmrf(mat: &FeecCsr) -> GmrfSparse {
    let mut coo = GmrfCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        coo.push(row, col, *value);
    }
    GmrfSparse::from(&coo)
}

pub fn feec_vec_to_gmrf(vec: &FeecVector) -> GmrfVector {
    GmrfVector::from_vec(vec.iter().copied().collect())
}

pub fn feec_csr_to_dense(mat: &FeecCsr) -> FeecMatrix {
    let mut dense = FeecMatrix::zeros(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        dense[(row, col)] += *value;
    }
    dense
}

pub fn dense_to_feec_csr(mat: &FeecMatrix, drop_tolerance: f64) -> FeecCsr {
    let tol = drop_tolerance.abs();
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let value = mat[(i, j)];
            if value.abs() > tol {
                coo.push(i, j, value);
            }
        }
    }
    FeecCsr::from(&coo)
}

pub(crate) fn lumped_diag(mat: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    diag
}

pub(crate) fn invert_diag(diag: &[f64]) -> Vec<f64> {
    let eps = 1e-12;
    diag.iter()
        .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
        .collect()
}

pub(crate) fn diag_matrix(diag: &[f64]) -> FeecCsr {
    let mut coo = FeecCoo::new(diag.len(), diag.len());
    for (i, value) in diag.iter().copied().enumerate() {
        if value != 0.0 {
            coo.push(i, i, value);
        }
    }
    FeecCsr::from(&coo)
}

pub(crate) fn scale_matrix(mat: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        let scaled = *value * scale;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

pub(crate) fn add_sparse(a: &FeecCsr, b: &FeecCsr) -> FeecCsr {
    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());

    let mut coo = FeecCoo::new(a.nrows(), a.ncols());
    for (row, col, value) in a.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in b.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn feec_csr_dense_roundtrip_preserves_entries() {
        let mut coo = FeecCoo::new(3, 3);
        coo.push(0, 0, 2.0);
        coo.push(0, 2, -1.5);
        coo.push(1, 1, 3.0);
        coo.push(2, 0, 0.25);
        let csr = FeecCsr::from(&coo);

        let dense = feec_csr_to_dense(&csr);
        let roundtrip = dense_to_feec_csr(&dense, 0.0);

        assert_eq!(csr.nrows(), roundtrip.nrows());
        assert_eq!(csr.ncols(), roundtrip.ncols());
        let dense_roundtrip = feec_csr_to_dense(&roundtrip);
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                assert!(approx_eq(dense[(i, j)], dense_roundtrip[(i, j)]));
            }
        }
    }

    #[test]
    fn dense_to_feec_csr_drops_small_entries() {
        let dense = FeecMatrix::from_row_slice(2, 2, &[1.0, 1e-16, -1e-18, 2.0]);
        let csr = dense_to_feec_csr(&dense, 1e-14);
        let roundtrip = feec_csr_to_dense(&csr);

        assert!(approx_eq(roundtrip[(0, 0)], 1.0));
        assert!(approx_eq(roundtrip[(1, 1)], 2.0));
        assert!(approx_eq(roundtrip[(0, 1)], 0.0));
        assert!(approx_eq(roundtrip[(1, 0)], 0.0));
    }
}
