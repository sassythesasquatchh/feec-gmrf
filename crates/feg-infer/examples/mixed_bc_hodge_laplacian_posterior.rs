use common::linalg::nalgebra::{
    CooMatrix as FeecCoo, CsrMatrix as FeecCsr, Matrix as FeecMatrix, Vector as FeecVector,
};
use ddf::cochain::{cochain_projection, partial_cochain_projection, Cochain};
use exterior::{field::DiffFormClosure, ExteriorElement};
use feg_infer::matern_1form::{
    build_matern_precision_1form, feec_csr_to_gmrf, feec_vec_to_gmrf, HodgeLaplacian1Form,
    MaternConfig, MaternMassInverse,
};
use formoniq::{
    assemble::{self, assemble_boundary_integral_term, assemble_galvec},
    fe::{fe_l2_error, l2_norm},
    io::{write_1form_vector_field_vtk, write_cochain_vtk},
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::hodge_laplace,
};
use gmrf_core::observation::apply_gaussian_observations;
use gmrf_core::Gmrf;
use manifold::{
    gen::cartesian::CartesianMeshInfo, geometry::coord::CoordRef, topology::handle::KSimplexIdx,
};
use std::{
    collections::HashSet,
    f64::consts::PI,
    fs::{self, File},
    io::{BufWriter, Write},
};

#[derive(Debug, Clone)]
struct ConvergenceRow {
    resolution: usize,
    h: f64,
    free_u_dofs: usize,
    deterministic_error_l2: f64,
    posterior_error_l2: f64,
    posterior_minus_deterministic_l2: f64,
    deterministic_rate: Option<f64>,
    posterior_rate: Option<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "out/mixed_bc_hodge_laplacian_posterior";
    let _ = fs::remove_dir_all(out_dir);
    fs::create_dir_all(out_dir)?;

    let resolutions = [1, 2, 3, 4];
    let grade = 1;
    let homology_dim = 0;

    // Matern prior hyperparameters for the reduced 1-form system.
    let kappa = 10.0;
    let tau = 1.0;
    let noise_variance = 1e-10;

    let solution_exact = DiffFormClosure::one_form(
        |p| {
            FeecVector::from_column_slice(&[
                (PI * p[0]).sin() + p[0] * p[0] + p[1] + p[2],
                (PI * p[1]).cos() + p[0] + p[2] * p[2],
                (PI * p[2]).sin() + p[0] * p[0] + p[1] + p[2] * p[2],
            ])
        },
        3,
    );

    let sigma_exact = DiffFormClosure::scalar(
        |p| PI * (-(PI * p[0]).cos() + (PI * p[1]).sin() - (PI * p[2]).cos()) - 2.0 * (p[0] + p[2]),
        3,
    );

    let laplacian_exact = DiffFormClosure::one_form(
        |p| {
            FeecVector::from_column_slice(&[
                (PI * PI) * (PI * p[0]).sin() - 2.0,
                (PI * PI) * (PI * p[1]).cos() - 2.0,
                (PI * PI) * (PI * p[2]).sin() - 4.0,
            ])
        },
        3,
    );

    let solution_neumann_exact = DiffFormClosure::new(
        Box::new(|p| {
            ExteriorElement::new(
                FeecVector::from_column_slice(&[1.0 - 2.0 * p[2], 1.0 - 2.0 * p[0], 0.0]),
                3,
                1,
            )
        }),
        3,
        1,
    );

    let sigma_neumann_exact = DiffFormClosure::new(
        Box::new(|p| {
            ExteriorElement::new(
                FeecVector::from_column_slice(&[
                    -((PI * p[2]).sin() + p[0] * p[0] + p[1] + p[2] * p[2]),
                    (PI * p[1]).cos() + p[0] + p[2] * p[2],
                    -((PI * p[0]).sin() + p[0] * p[0] + p[1] + p[2]),
                ]),
                3,
                2,
            )
        }),
        3,
        2,
    );

    let unit_weight = InnerProductWeightClosure::new(|_p| 1.0);

    let mut rows = Vec::with_capacity(resolutions.len());

    for &resolution in &resolutions {
        let resolution = (2 as u32).pow(resolution as u32) as usize;
        let mesh = CartesianMeshInfo::new_unit_scaled(3, resolution, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let metric = coords.to_edge_lengths(&topology);

        let strong_dof_predicate = |p: CoordRef| p[0] == 0.0 || p[1] == 0.0 || p[2] == 0.0;
        let weak_dof_predicate = |p: CoordRef| !strong_dof_predicate(p);

        let strong_k_dofs = assemble::boundary_simplices_where_barycenter(
            &topology,
            &coords,
            1,
            strong_dof_predicate,
        )
        .into_iter()
        .collect::<HashSet<usize>>();

        let strong_k_minus_one_dofs = assemble::boundary_simplices_where_barycenter(
            &topology,
            &coords,
            0,
            strong_dof_predicate,
        )
        .into_iter()
        .collect::<HashSet<usize>>();

        let strong_k_dof_predicate = |sidx: KSimplexIdx| strong_k_dofs.contains(&sidx);
        let strong_k_minus_one_dof_predicate =
            |sidx: KSimplexIdx| strong_k_minus_one_dofs.contains(&sidx);

        let weak_face_dofs = assemble::boundary_simplices_where_barycenter(
            &topology,
            &coords,
            2,
            weak_dof_predicate,
        )
        .into_iter()
        .collect::<HashSet<usize>>();
        let weak_face_dof_predicate = |sidx: KSimplexIdx| weak_face_dofs.contains(&sidx);

        let solution_essential_data_map = partial_cochain_projection(
            &solution_exact,
            &topology,
            &coords,
            &strong_k_dof_predicate,
            None,
        );
        let solution_essential_data = |kidx: KSimplexIdx| solution_essential_data_map[&kidx];

        let sigma_essential_data_map = partial_cochain_projection(
            &sigma_exact,
            &topology,
            &coords,
            &strong_k_minus_one_dof_predicate,
            None,
        );
        let sigma_essential_data = |kidx: KSimplexIdx| sigma_essential_data_map[&kidx];

        let mut strong_u_dof_coeffs: Vec<(usize, f64)> = solution_essential_data_map
            .iter()
            .map(|(kidx, value)| (*kidx, *value))
            .collect();
        strong_u_dof_coeffs.sort_unstable_by_key(|(idx, _)| *idx);

        let sigma_neumann_galvec = assemble_boundary_integral_term(
            &topology,
            &coords,
            0,
            &sigma_neumann_exact,
            None,
            &weak_face_dof_predicate,
        );

        let solution_neumann_galvec = assemble_boundary_integral_term(
            &topology,
            &coords,
            1,
            &solution_neumann_exact,
            None,
            &weak_face_dof_predicate,
        );

        let source_galvec = assemble_galvec(
            &topology,
            &metric,
            SourceElVec::new(&laplacian_exact, &coords, None),
        );

        let galmats = hodge_laplace::MixedGalmats::compute_weighted(
            &topology,
            &metric,
            grade,
            &coords,
            None,
            &unit_weight,
        );

        let (_sigma_galsol, deterministic_u, _harmonics) =
            hodge_laplace::solve_hodge_laplace_source_with_galmats_and_boundary_conditions(
                &topology,
                &galmats,
                Some(sigma_neumann_galvec.clone()),
                source_galvec.clone() + solution_neumann_galvec.clone(),
                grade,
                homology_dim,
                &strong_k_dof_predicate,
                &solution_essential_data,
                &strong_k_minus_one_dof_predicate,
                &sigma_essential_data,
            );

        let (reduced_laplacian, reduced_rhs, reduced_mass_u) =
            build_reduced_schur_system_with_mixed_bc(
                &galmats,
                source_galvec.clone() + solution_neumann_galvec,
                sigma_neumann_galvec,
                &strong_k_dof_predicate,
                &solution_essential_data,
                &strong_k_minus_one_dof_predicate,
                &sigma_essential_data,
            );

        let prior_precision = build_matern_precision_1form(
            &topology,
            &metric,
            &HodgeLaplacian1Form {
                mass_u: reduced_mass_u,
                laplacian: reduced_laplacian.clone(),
            },
            MaternConfig {
                kappa,
                tau,
                mass_inverse: MaternMassInverse::RowSumLumped,
            },
        );

        let q_prior = feec_csr_to_gmrf(&prior_precision);
        let observation_matrix = feec_csr_to_gmrf(&reduced_laplacian);
        let observations = feec_vec_to_gmrf(&reduced_rhs);

        let (posterior_precision, information) = apply_gaussian_observations(
            &q_prior,
            &observation_matrix,
            &observations,
            None,
            noise_variance,
        );

        let posterior = Gmrf::from_information_and_precision(information, posterior_precision)?;

        let mut posterior_mean_full = gmrf_vec_to_feec(posterior.mean());
        assemble::reintroduce_non_homogenous_dofs_galsols(
            &strong_u_dof_coeffs,
            &mut posterior_mean_full,
        );
        let posterior_mean = Cochain::new(grade, posterior_mean_full);

        let exact_projected = cochain_projection(&solution_exact, &topology, &coords, None);

        let deterministic_error_l2 =
            fe_l2_error(&deterministic_u, &solution_exact, &topology, &coords);
        let posterior_error_l2 = fe_l2_error(&posterior_mean, &solution_exact, &topology, &coords);
        let posterior_minus_deterministic_l2 = l2_norm(
            &(posterior_mean.clone() - deterministic_u.clone()),
            &topology,
            &metric,
        );

        let h = 1.0 / resolution as f64;
        let deterministic_rate = rows.last().map(|prev: &ConvergenceRow| {
            f64::ln(prev.deterministic_error_l2 / deterministic_error_l2) / f64::ln(prev.h / h)
        });
        let posterior_rate = rows.last().map(|prev: &ConvergenceRow| {
            f64::ln(prev.posterior_error_l2 / posterior_error_l2) / f64::ln(prev.h / h)
        });

        println!(
            "res={resolution:>2} h={h:.4} free_u={} det_L2={deterministic_error_l2:.6e} post_L2={posterior_error_l2:.6e} post-det_L2={posterior_minus_deterministic_l2:.6e}",
            reduced_laplacian.nrows(),
        );
        if let Some(rate) = deterministic_rate {
            println!("  deterministic observed rate ~ {rate:.2}");
        }
        if let Some(rate) = posterior_rate {
            println!("  posterior observed rate ~ {rate:.2}");
        }

        let res_dir = format!("{out_dir}/res_{resolution}");
        fs::create_dir_all(&res_dir)?;
        write_cochain_vtk(
            format!("{res_dir}/deterministic_u.vtk"),
            &coords,
            &topology,
            &deterministic_u,
            "deterministic_u",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/posterior_mean.vtk"),
            &coords,
            &topology,
            &posterior_mean,
            "posterior_mean",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/solution_exact.vtk"),
            &coords,
            &topology,
            &exact_projected,
            "solution_exact",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/posterior_minus_exact.vtk"),
            &coords,
            &topology,
            &(posterior_mean.clone() - exact_projected.clone()),
            "posterior_minus_exact",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/deterministic_minus_exact.vtk"),
            &coords,
            &topology,
            &(deterministic_u.clone() - exact_projected.clone()),
            "deterministic_minus_exact",
        )?;
        write_cochain_vtk(
            format!("{res_dir}/posterior_minus_deterministic.vtk"),
            &coords,
            &topology,
            &(posterior_mean.clone() - deterministic_u.clone()),
            "posterior_minus_deterministic",
        )?;
        write_1form_vector_field_vtk(
            format!("{res_dir}/posterior_mean_vector_field.vtk"),
            &coords,
            &topology,
            &posterior_mean,
            "posterior_mean_vector_field",
        )?;
        write_1form_vector_field_vtk(
            format!("{res_dir}/deterministic_u_vector_field.vtk"),
            &coords,
            &topology,
            &deterministic_u,
            "deterministic_u_vector_field",
        )?;
        write_1form_vector_field_vtk(
            format!("{res_dir}/exact_u_vector_field.vtk"),
            &coords,
            &topology,
            &exact_projected,
            "exact_u_vector_field",
        )?;

        rows.push(ConvergenceRow {
            resolution,
            h,
            free_u_dofs: reduced_laplacian.nrows(),
            deterministic_error_l2,
            posterior_error_l2,
            posterior_minus_deterministic_l2,
            deterministic_rate,
            posterior_rate,
        });
    }

    write_convergence_csv(&format!("{out_dir}/convergence.csv"), &rows)?;
    println!("Wrote convergence results to {out_dir}/convergence.csv");
    println!("Wrote per-resolution fields to {out_dir}/res_* directories");

    Ok(())
}

fn build_reduced_schur_system_with_mixed_bc(
    galmats: &hodge_laplace::MixedGalmats,
    mut u_rhs: FeecVector,
    mut sigma_rhs: FeecVector,
    k_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_strong_bc_data: &dyn Fn(KSimplexIdx) -> f64,
    k_minus_one_strong_bc_predicate: &dyn Fn(KSimplexIdx) -> bool,
    k_minus_one_strong_bc_data: &dyn Fn(KSimplexIdx) -> f64,
) -> (FeecCsr, FeecVector, FeecCsr) {
    let reduced_u_len = galmats.free_u_len(k_strong_bc_predicate);
    let harmonics = FeecMatrix::zeros(reduced_u_len, 0);

    let (reduced_mixed, reduced_rhs) = galmats
        .mixed_hodge_laplacian_with_strong_bc_via_elimination(
            k_minus_one_strong_bc_predicate,
            k_minus_one_strong_bc_data,
            k_strong_bc_predicate,
            k_strong_bc_data,
            &mut sigma_rhs,
            &mut u_rhs,
            &harmonics,
        );

    let reduced_sigma_len = galmats.free_sigma_len(k_minus_one_strong_bc_predicate);
    let (mass_sigma, a12, a21, k_matrix) =
        split_reduced_mixed_blocks(&reduced_mixed, reduced_sigma_len, reduced_u_len);

    let c_matrix = scale_matrix(&a12, -1.0);
    let mass_sigma_inv_lumped = invert_diag(&lumped_diag(&mass_sigma));
    let mass_inv_c = scale_rows(&c_matrix, &mass_sigma_inv_lumped);

    let schur = add_sparse(&k_matrix, &(&a21 * &mass_inv_c));

    let mut rhs_sigma = FeecVector::zeros(reduced_sigma_len);
    for i in 0..reduced_sigma_len {
        rhs_sigma[i] = reduced_rhs[i];
    }

    let mut rhs_u = FeecVector::zeros(reduced_u_len);
    for i in 0..reduced_u_len {
        rhs_u[i] = reduced_rhs[reduced_sigma_len + i];
    }

    let scaled_rhs_sigma = FeecVector::from_iterator(
        reduced_sigma_len,
        rhs_sigma
            .iter()
            .enumerate()
            .map(|(i, value)| mass_sigma_inv_lumped[i] * *value),
    );
    let rhs_u_schur = rhs_u - &a21 * scaled_rhs_sigma;

    let mut mass_u = galmats.mass_u().clone();
    let strongly_enforced_u_dofs = (0..mass_u.nrows())
        .filter(|&i| k_strong_bc_predicate(i))
        .collect::<HashSet<_>>();
    assemble::drop_dofs_galmat(&strongly_enforced_u_dofs, &mut mass_u);

    (schur, rhs_u_schur, FeecCsr::from(&mass_u))
}

fn split_reduced_mixed_blocks(
    matrix: &FeecCsr,
    reduced_sigma_len: usize,
    reduced_u_len: usize,
) -> (FeecCsr, FeecCsr, FeecCsr, FeecCsr) {
    let mut mass_sigma = FeecCoo::new(reduced_sigma_len, reduced_sigma_len);
    let mut a12 = FeecCoo::new(reduced_sigma_len, reduced_u_len);
    let mut a21 = FeecCoo::new(reduced_u_len, reduced_sigma_len);
    let mut k_matrix = FeecCoo::new(reduced_u_len, reduced_u_len);

    let u_offset = reduced_sigma_len;
    let total = reduced_sigma_len + reduced_u_len;

    for (row, col, value) in matrix.triplet_iter() {
        if row >= total || col >= total {
            continue;
        }
        if row < reduced_sigma_len {
            if col < reduced_sigma_len {
                mass_sigma.push(row, col, *value);
            } else {
                a12.push(row, col - u_offset, *value);
            }
        } else if col < reduced_sigma_len {
            a21.push(row - u_offset, col, *value);
        } else {
            k_matrix.push(row - u_offset, col - u_offset, *value);
        }
    }

    (
        FeecCsr::from(&mass_sigma),
        FeecCsr::from(&a12),
        FeecCsr::from(&a21),
        FeecCsr::from(&k_matrix),
    )
}

fn lumped_diag(mat: &FeecCsr) -> Vec<f64> {
    let mut diag = vec![0.0; mat.nrows()];
    for (row, _col, value) in mat.triplet_iter() {
        diag[row] += *value;
    }
    diag
}

fn invert_diag(diag: &[f64]) -> Vec<f64> {
    let eps = 1e-12;
    diag.iter()
        .map(|v| if v.abs() < eps { 0.0 } else { 1.0 / v })
        .collect()
}

fn scale_rows(mat: &FeecCsr, row_scales: &[f64]) -> FeecCsr {
    assert_eq!(mat.nrows(), row_scales.len());
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        let scaled = *value * row_scales[row];
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn scale_matrix(mat: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = FeecCoo::new(mat.nrows(), mat.ncols());
    for (row, col, value) in mat.triplet_iter() {
        let scaled = *value * scale;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn add_sparse(a: &FeecCsr, b: &FeecCsr) -> FeecCsr {
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

fn gmrf_vec_to_feec(vec: &gmrf_core::types::Vector) -> FeecVector {
    FeecVector::from_vec(vec.iter().copied().collect())
}

fn write_convergence_csv(path: &str, rows: &[ConvergenceRow]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "resolution,h,free_u_dofs,deterministic_error_l2,posterior_error_l2,posterior_minus_deterministic_l2,deterministic_rate,posterior_rate"
    )?;

    for row in rows {
        let det_rate = row
            .deterministic_rate
            .map(|v| format!("{v:.12e}"))
            .unwrap_or_else(|| "".to_string());
        let post_rate = row
            .posterior_rate
            .map(|v| format!("{v:.12e}"))
            .unwrap_or_else(|| "".to_string());

        writeln!(
            writer,
            "{},{:.12e},{},{:.12e},{:.12e},{:.12e},{},{}",
            row.resolution,
            row.h,
            row.free_u_dofs,
            row.deterministic_error_l2,
            row.posterior_error_l2,
            row.posterior_minus_deterministic_l2,
            det_rate,
            post_rate,
        )?;
    }

    Ok(())
}
