use common::linalg::nalgebra::{CsrMatrix as FeecCsr, Vector as FeecVector};
use ddf::cochain::Cochain;
use ddf::whitney::lsf::WhitneyLsf;
use ddf::ManifoldComplexExt;
use exterior::field::{DiffFormClosure, ExteriorField};
use feg_core::{
    BoundaryRegionSpec, BoundarySpec, BoundaryTreatment, GaussianPriorSpec,
    LinearGaussianMeasurementSpec, LinearUncertainInputSpec, RepresentationPreference,
    SparseTriplet, SparseTripletMatrix,
};
use feg_infer::{
    core_triplet_to_feec_csr,
    diagnostics::build_harmonic_orthogonality_constraints,
    linear_pde::{
        build_linear_pde_joint_posterior_with_config, solve_linear_pde_uq_with_config,
        LinearPdeDerivedQuantitySpec, LinearPdeUqProblem, LinearPdeUqResult,
        LinearPdeUqSolverConfig, LinearPdeVarianceConfig, LinearPdeVarianceMode,
    },
    matern_1form::feec_csr_to_gmrf,
    vtk::write_1cochain_vtk_fields,
};
use formoniq::{
    assemble::{
        self, assemble_boundary_integral_term, assemble_galvec,
        assemble_whitney_projected_sparse_inverse_galmat_weighted,
    },
    io::{
        sample_2form_cell_vectors, write_1form_vector_field_vtk, write_2form_vector_field_vtk,
        write_cochain_vtk, write_top_cell_vtk_fields,
    },
    operators::{InnerProductWeightClosure, SourceElVec},
    problems::{
        hodge_laplace::{self, MixedGalmats},
        linear_uq::{
            build_reduced_hodge_laplace_1form_system_with_galmats,
            reduce_reduced_hodge_laplace_1form_rhs_with_galmats,
        },
    },
};
use gmrf_core::types::{
    CooMatrix as GmrfCoo, DenseMatrix as GmrfDenseMatrix, SparseMatrix as GmrfSparseMatrix,
    Vector as GmrfVector,
};
use gmrf_core::{apply_gaussian_observations, LinearObservationStackBuilder, SparseRowOperator};
use manifold::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            simplex::{barycenter_local, SimplexCoords, SimplexHandleExt},
            CoordRef,
        },
        metric::mesh::MeshLengths,
    },
    topology::{
        complex::Complex,
        handle::{KSimplexIdx, SimplexIdx},
    },
};
use rand::SeedableRng;
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    env,
    error::Error,
    f64::consts::PI,
    fs,
    path::Path,
    time::Instant,
};

const OUT_DIR: &str = "out/examples/magnetostatic_linear_pde_uq";
const MESH_PATH: &str = "meshes/toroidal_inductor.msh";
const SOURCE_MODE_COUNT: usize = 4;
const METRIC_EPS: f64 = 1e-12;
const MAGNETIC_FIELD_DERIVED_NAME: &str = "magnetic_field";
const MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME: &str = "magnetic_field_vector_x";
const MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME: &str = "magnetic_field_vector_y";
const MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME: &str = "magnetic_field_vector_z";

#[derive(Debug, Clone, Copy)]
struct ToroidalInductorGeometry {
    major_radius: f64,
    core_minor_radius: f64,
    coil_minor_radius: f64,
    box_half_length: f64,
    target_air_cell_size: f64,
}

#[derive(Debug, Clone)]
struct StageOutcome {
    name: String,
    result: LinearPdeUqResult,
    relative_l2_error: f64,
    relative_b_l2_error: f64,
    posterior_mean_norm: f64,
    truth_norm: f64,
    pde_residual_norm: f64,
    sensor_predictions: Vec<f64>,
    sensor_rmse: f64,
    harmonic_residual_norm: f64,
}

#[derive(Debug, Clone)]
struct UqSubsetIndices {
    all_edges: Vec<usize>,
    coil_edges: Vec<usize>,
    outer_boundary_edges: Vec<usize>,
    core_boundary_edges: Vec<usize>,
    sensor_edges: Vec<usize>,
    background_edges: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
struct StageUncertaintySummary {
    count: usize,
    absolute_error_mean: f64,
    prior_variance_mean: f64,
    posterior_variance_mean: f64,
    variance_reduction_mean: f64,
    variance_ratio_mean: f64,
    posterior_std_mean: f64,
    normalized_abs_error_mean: f64,
    truth_within_1sigma_fraction: f64,
    truth_within_2sigma_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct StageTransitionSummary {
    count: usize,
    baseline_posterior_variance_mean: f64,
    comparison_posterior_variance_mean: f64,
    posterior_variance_delta_mean: f64,
    posterior_variance_mean_ratio: f64,
    posterior_variance_pointwise_ratio_mean: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct MagneticFieldPosteriorVarianceDebugRow {
    face_index: usize,
    prior_variance: f64,
    reported_posterior_variance: f64,
    exact_posterior_variance: f64,
    mc_posterior_variance: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct MagneticFieldPosteriorVarianceDebugReport {
    stage_name: String,
    variance_mode: LinearPdeVarianceMode,
    num_samples: usize,
    face_count: usize,
    rows: Vec<MagneticFieldPosteriorVarianceDebugRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestedStage {
    Stage1,
    Stage2,
    Stage3,
    Stage4,
}

#[derive(Debug, Clone)]
struct ExampleConfig {
    mesh_path: String,
    pde_variance: f64,
    solver: LinearPdeUqSolverConfig,
    debug_stage3: bool,
    debug_b_posterior_mc_samples: Option<usize>,
    debug_b_face_count: usize,
    skip_output_writing: bool,
    skip_source_equivalence: bool,
    only_stage: Option<RequestedStage>,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            mesh_path: MESH_PATH.to_string(),
            pde_variance: 1e-4,
            solver: LinearPdeUqSolverConfig {
                variance: LinearPdeVarianceConfig {
                    mode: LinearPdeVarianceMode::Exact,
                    num_rbmc_probes: 32,
                    rbmc_batch_count: 4,
                    rng_seed: 17,
                },
                stabilize_precision: false,
                log_diagnostics: true,
            },
            debug_stage3: false,
            debug_b_posterior_mc_samples: None,
            debug_b_face_count: 8,
            skip_output_writing: false,
            skip_source_equivalence: false,
            only_stage: None,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    if config.debug_b_posterior_mc_samples.is_some() && config.only_stage.is_none() {
        return Err(
            invalid_input(
                "--debug-b-posterior-mc requires --only-stage so the diagnostic targets a single posterior",
            )
            .into(),
        );
    }
    let _ = fs::remove_dir_all(OUT_DIR);
    fs::create_dir_all(OUT_DIR)?;

    let mesh_bytes = fs::read(&config.mesh_path)?;
    let (topology, coords) = manifold::io::gmsh::gmsh2coord_complex(&mesh_bytes);
    let metric = coords.to_edge_lengths(&topology);
    let geom = ToroidalInductorGeometry {
        major_radius: 2.0,
        core_minor_radius: 0.60,
        coil_minor_radius: 0.85,
        box_half_length: 6.0,
        target_air_cell_size: 0.25,
    };

    let mu_0 = 4e-7 * PI;
    let mu_0_inverse = 1.0 / mu_0;
    let inverse_permeability = InnerProductWeightClosure::new(move |_| mu_0_inverse);

    let outer_state_dofs = sorted_boundary_dofs(&topology, &coords, 1, |point| {
        outer_boundary_predicate(point, geom)
    });
    let outer_aux_dofs = sorted_boundary_dofs(&topology, &coords, 0, |point| {
        outer_boundary_predicate(point, geom)
    });
    let hard_boundary = BoundarySpec::default()
        .with_state_region(BoundaryRegionSpec::new(
            "outer_state_hard",
            outer_state_dofs.clone(),
            vec![0.0; outer_state_dofs.len()],
            BoundaryTreatment::HardEssential,
        ))
        .with_auxiliary_region(BoundaryRegionSpec::new(
            "outer_aux_hard",
            outer_aux_dofs.clone(),
            vec![0.0; outer_aux_dofs.len()],
            BoundaryTreatment::HardEssential,
        ));
    let soft_boundary = BoundarySpec::default()
        .with_state_region(BoundaryRegionSpec::new(
            "outer_state_soft",
            outer_state_dofs.clone(),
            vec![0.0; outer_state_dofs.len()],
            BoundaryTreatment::SoftEssential { variance: 1e-3 },
        ))
        .with_auxiliary_region(BoundaryRegionSpec::new(
            "outer_aux_hard",
            outer_aux_dofs.clone(),
            vec![0.0; outer_aux_dofs.len()],
            BoundaryTreatment::HardEssential,
        ));

    let nominal_source_full = assemble_weighted_source(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &coil_mode(geom, mu_0, None),
    );

    let galmats =
        MixedGalmats::compute_weighted(&topology, &metric, 1, &coords, None, &inverse_permeability);
    let state_mass_inverse =
        FeecCsr::from(&assemble_whitney_projected_sparse_inverse_galmat_weighted(
            &topology,
            &metric,
            &coords,
            None,
            &inverse_permeability,
        ));
    let hard_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &hard_boundary,
        &state_mass_inverse,
    )?;
    let soft_system = build_reduced_hodge_laplace_1form_system_with_galmats(
        &galmats,
        &soft_boundary,
        &state_mass_inverse,
    )?;

    let hard_source_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        &galmats,
        &hard_boundary,
        &FeecVector::zeros(galmats.sigma_len()),
        &nominal_source_full,
    )?;
    let truth_a = solve_full_feec_deterministic_reference(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &nominal_source_full,
        geom,
    );
    write_truth_outputs(&topology, &coords, &truth_a, OUT_DIR)?;
    let soft_source_rhs = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        &galmats,
        &soft_boundary,
        &FeecVector::zeros(galmats.sigma_len()),
        &nominal_source_full,
    )?;

    let hard_source_operator = build_source_mode_operator(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &galmats,
        &hard_boundary,
        geom,
        mu_0,
    )?;
    let soft_source_operator = build_source_mode_operator(
        &topology,
        &metric,
        &coords,
        &inverse_permeability,
        &galmats,
        &soft_boundary,
        geom,
        mu_0,
    )?;
    let soft_neumann_operator =
        build_core_neumann_operator(&topology, &coords, &galmats, &soft_boundary, geom)?;

    let hard_prior = build_whittle_prior(&hard_system, config.solver.stabilize_precision);
    let soft_prior = build_whittle_prior(&soft_system, config.solver.stabilize_precision);

    let harmonic_measurements =
        build_weighted_harmonic_measurements(&topology, &galmats, &soft_system, &outer_aux_dofs)?;
    let flux_measurements =
        build_flux_loop_measurements(&topology, &coords, truth_a.coeffs.clone(), geom)?;
    let derived_quantities = build_magnetic_field_derived_quantities(&topology, &coords)?;
    let uq_subsets = build_uq_subset_indices(
        &topology,
        &coords,
        &outer_state_dofs,
        &flux_measurements,
        geom,
    );

    if config.debug_stage3 {
        run_stage3_debug_cases(
            &config.solver,
            &soft_prior,
            &soft_system,
            &soft_source_operator,
            &soft_neumann_operator,
            &harmonic_measurements,
        )?;
        return Ok(());
    }

    if let Some(requested_stage) = config.only_stage {
        let (stage, problem) = match requested_stage {
            RequestedStage::Stage1 => {
                let mut system = hard_system.clone();
                subtract_from_bias(&mut system.residual_bias, &hard_source_rhs)?;
                let problem = LinearPdeUqProblem {
                    state_prior: hard_prior.clone(),
                    system,
                    uncertain_inputs: Vec::new(),
                    physical_measurements: Vec::new(),
                    derived_quantities: derived_quantities.clone(),
                    pde_variance: Some(config.pde_variance),
                };
                let stage = run_stage(
                    "stage1_prior_pde_only",
                    &config.solver,
                    problem.clone(),
                    &topology,
                    &truth_a.coeffs,
                    &flux_measurements,
                    &[],
                )
                .map_err(|err| format!("stage1_prior_pde_only failed: {err}"))?;
                (stage, problem)
            }
            RequestedStage::Stage2 => {
                let problem = LinearPdeUqProblem {
                    state_prior: hard_prior.clone(),
                    system: hard_system.clone(),
                    uncertain_inputs: vec![LinearUncertainInputSpec {
                        name: "coil_source".to_string(),
                        operator: hard_source_operator,
                        prior: GaussianPriorSpec {
                            mean: vec![1.0; SOURCE_MODE_COUNT],
                            precision: diagonal_precision(SOURCE_MODE_COUNT, 1.0 / (0.35 * 0.35)),
                        },
                        preference: RepresentationPreference::ForceLatent,
                        collapsed_precision: None,
                    }],
                    physical_measurements: Vec::new(),
                    derived_quantities: derived_quantities.clone(),
                    pde_variance: Some(config.pde_variance),
                };
                let stage = run_stage(
                    "stage2_uncertain_source",
                    &config.solver,
                    problem.clone(),
                    &topology,
                    &truth_a.coeffs,
                    &flux_measurements,
                    &[],
                )
                .map_err(|err| format!("stage2_uncertain_source failed: {err}"))?;
                (stage, problem)
            }
            RequestedStage::Stage3 => {
                let problem = LinearPdeUqProblem {
                    state_prior: soft_prior.clone(),
                    system: soft_system.clone(),
                    uncertain_inputs: vec![
                        LinearUncertainInputSpec {
                            name: "coil_source".to_string(),
                            operator: soft_source_operator,
                            prior: GaussianPriorSpec {
                                mean: vec![1.0; SOURCE_MODE_COUNT],
                                precision: diagonal_precision(
                                    SOURCE_MODE_COUNT,
                                    1.0 / (0.35 * 0.35),
                                ),
                            },
                            preference: RepresentationPreference::ForceLatent,
                            collapsed_precision: None,
                        },
                        LinearUncertainInputSpec {
                            name: "core_neumann".to_string(),
                            operator: soft_neumann_operator,
                            prior: GaussianPriorSpec {
                                mean: vec![0.0; 2],
                                precision: diagonal_precision(2, 1.0),
                            },
                            preference: RepresentationPreference::ForceLatent,
                            collapsed_precision: None,
                        },
                    ],
                    physical_measurements: harmonic_measurements.clone(),
                    derived_quantities: derived_quantities.clone(),
                    pde_variance: Some(config.pde_variance),
                };
                let stage = run_stage(
                    "stage3_uncertain_source_and_bc",
                    &config.solver,
                    problem.clone(),
                    &topology,
                    &truth_a.coeffs,
                    &flux_measurements,
                    &harmonic_measurements,
                )
                .map_err(|err| format!("stage3_uncertain_source_and_bc failed: {err}"))?;
                (stage, problem)
            }
            RequestedStage::Stage4 => {
                let stage4_measurements = harmonic_measurements
                    .iter()
                    .cloned()
                    .chain(flux_measurements.iter().cloned())
                    .collect::<Vec<_>>();
                let problem = LinearPdeUqProblem {
                    state_prior: soft_prior.clone(),
                    system: soft_system.clone(),
                    uncertain_inputs: vec![
                        LinearUncertainInputSpec {
                            name: "coil_source".to_string(),
                            operator: build_source_mode_operator(
                                &topology,
                                &metric,
                                &coords,
                                &inverse_permeability,
                                &galmats,
                                &soft_boundary,
                                geom,
                                mu_0,
                            )?,
                            prior: GaussianPriorSpec {
                                mean: vec![1.0; SOURCE_MODE_COUNT],
                                precision: diagonal_precision(
                                    SOURCE_MODE_COUNT,
                                    1.0 / (0.35 * 0.35),
                                ),
                            },
                            preference: RepresentationPreference::ForceLatent,
                            collapsed_precision: None,
                        },
                        LinearUncertainInputSpec {
                            name: "core_neumann".to_string(),
                            operator: build_core_neumann_operator(
                                &topology,
                                &coords,
                                &galmats,
                                &soft_boundary,
                                geom,
                            )?,
                            prior: GaussianPriorSpec {
                                mean: vec![0.0; 2],
                                precision: diagonal_precision(2, 1.0),
                            },
                            preference: RepresentationPreference::ForceLatent,
                            collapsed_precision: None,
                        },
                    ],
                    physical_measurements: stage4_measurements,
                    derived_quantities: derived_quantities.clone(),
                    pde_variance: Some(config.pde_variance),
                };
                let stage = run_stage(
                    "stage4_with_flux_observations",
                    &config.solver,
                    problem.clone(),
                    &topology,
                    &truth_a.coeffs,
                    &flux_measurements,
                    &harmonic_measurements,
                )
                .map_err(|err| format!("stage4_with_flux_observations failed: {err}"))?;
                (stage, problem)
            }
        };

        println!("Completed {}", stage.name);
        println!(
            "relative_l2_error={:.3e} relative_b_l2_error={:.3e} mean_norm_ratio={:.3e} pde_residual_norm={:.3e} sensor_rmse={:.3e} harmonic_residual={:.3e}",
            stage.relative_l2_error,
            stage.relative_b_l2_error,
            stage.posterior_mean_norm / stage.truth_norm.max(1e-12),
            stage.pde_residual_norm,
            stage.sensor_rmse,
            stage.harmonic_residual_norm
        );
        print_stage_uq_summary(&stage, &truth_a.coeffs, &uq_subsets);
        if !config.skip_output_writing {
            write_stage_outputs(&topology, &coords, &truth_a.coeffs, OUT_DIR, &stage)?;
            println!("wrote stage outputs to {OUT_DIR}/{}", stage.name);
        }
        if let Some(num_samples) = config.debug_b_posterior_mc_samples {
            debug_magnetic_field_posterior_variance(
                &stage,
                &problem,
                &config.solver,
                num_samples,
                config.debug_b_face_count,
                if config.skip_output_writing {
                    None
                } else {
                    Some(Path::new(OUT_DIR).join(&stage.name))
                },
            )?;
        }
        return Ok(());
    }

    println!("running stage1_prior_pde_only");
    let stage1_start = Instant::now();
    let stage1 = {
        let mut system = hard_system.clone();
        subtract_from_bias(&mut system.residual_bias, &hard_source_rhs)?;
        run_stage(
            "stage1_prior_pde_only",
            &config.solver,
            LinearPdeUqProblem {
                state_prior: hard_prior.clone(),
                system,
                uncertain_inputs: Vec::new(),
                physical_measurements: Vec::new(),
                derived_quantities: derived_quantities.clone(),
                pde_variance: Some(config.pde_variance),
            },
            &topology,
            &truth_a.coeffs,
            &flux_measurements,
            &[],
        )
        .map_err(|err| format!("stage1_prior_pde_only failed: {err}"))?
    };
    println!(
        "completed stage1_prior_pde_only in {:?}",
        stage1_start.elapsed()
    );
    println!("running stage2_uncertain_source");
    let stage2_start = Instant::now();
    let stage2 = run_stage(
        "stage2_uncertain_source",
        &config.solver,
        LinearPdeUqProblem {
            state_prior: hard_prior.clone(),
            system: hard_system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "coil_source".to_string(),
                operator: hard_source_operator,
                prior: GaussianPriorSpec {
                    mean: vec![1.0; SOURCE_MODE_COUNT],
                    precision: diagonal_precision(SOURCE_MODE_COUNT, 1.0 / (0.35 * 0.35)),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: derived_quantities.clone(),
            pde_variance: Some(config.pde_variance),
        },
        &topology,
        &truth_a.coeffs,
        &flux_measurements,
        &[],
    )
    .map_err(|err| format!("stage2_uncertain_source failed: {err}"))?;
    println!(
        "completed stage2_uncertain_source in {:?}",
        stage2_start.elapsed()
    );
    println!("running stage3_uncertain_source_and_bc");
    let stage3_start = Instant::now();
    let stage3 = run_stage(
        "stage3_uncertain_source_and_bc",
        &config.solver,
        LinearPdeUqProblem {
            state_prior: soft_prior.clone(),
            system: soft_system.clone(),
            uncertain_inputs: vec![
                LinearUncertainInputSpec {
                    name: "coil_source".to_string(),
                    operator: soft_source_operator,
                    prior: GaussianPriorSpec {
                        mean: vec![1.0; SOURCE_MODE_COUNT],
                        precision: diagonal_precision(SOURCE_MODE_COUNT, 1.0 / (0.35 * 0.35)),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                },
                LinearUncertainInputSpec {
                    name: "core_neumann".to_string(),
                    operator: soft_neumann_operator,
                    prior: GaussianPriorSpec {
                        mean: vec![0.0; 2],
                        precision: diagonal_precision(2, 1.0),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                },
            ],
            physical_measurements: harmonic_measurements.clone(),
            derived_quantities: derived_quantities.clone(),
            pde_variance: Some(config.pde_variance),
        },
        &topology,
        &truth_a.coeffs,
        &flux_measurements,
        &harmonic_measurements,
    )
    .map_err(|err| format!("stage3_uncertain_source_and_bc failed: {err}"))?;
    println!(
        "completed stage3_uncertain_source_and_bc in {:?}",
        stage3_start.elapsed()
    );
    let stage4_measurements = harmonic_measurements
        .iter()
        .cloned()
        .chain(flux_measurements.iter().cloned())
        .collect::<Vec<_>>();
    println!("running stage4_with_flux_observations");
    let stage4_start = Instant::now();
    let stage4 = run_stage(
        "stage4_with_flux_observations",
        &config.solver,
        LinearPdeUqProblem {
            state_prior: soft_prior.clone(),
            system: soft_system.clone(),
            uncertain_inputs: vec![
                LinearUncertainInputSpec {
                    name: "coil_source".to_string(),
                    operator: build_source_mode_operator(
                        &topology,
                        &metric,
                        &coords,
                        &inverse_permeability,
                        &galmats,
                        &soft_boundary,
                        geom,
                        mu_0,
                    )?,
                    prior: GaussianPriorSpec {
                        mean: vec![1.0; SOURCE_MODE_COUNT],
                        precision: diagonal_precision(SOURCE_MODE_COUNT, 1.0 / (0.35 * 0.35)),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                },
                LinearUncertainInputSpec {
                    name: "core_neumann".to_string(),
                    operator: build_core_neumann_operator(
                        &topology,
                        &coords,
                        &galmats,
                        &soft_boundary,
                        geom,
                    )?,
                    prior: GaussianPriorSpec {
                        mean: vec![0.0; 2],
                        precision: diagonal_precision(2, 1.0),
                    },
                    preference: RepresentationPreference::ForceLatent,
                    collapsed_precision: None,
                },
            ],
            physical_measurements: stage4_measurements.clone(),
            derived_quantities: derived_quantities.clone(),
            pde_variance: Some(config.pde_variance),
        },
        &topology,
        &truth_a.coeffs,
        &flux_measurements,
        &harmonic_measurements,
    )
    .map_err(|err| format!("stage4_with_flux_observations failed: {err}"))?;
    println!(
        "completed stage4_with_flux_observations in {:?}",
        stage4_start.elapsed()
    );

    let source_equivalence = if config.skip_source_equivalence {
        println!("skipping source-equivalence side check");
        f64::NAN
    } else {
        println!("running source-equivalence side check");
        let check_start = Instant::now();
        let result = run_source_equivalence_side_check(
            &config.solver,
            &soft_prior,
            &soft_system,
            &soft_source_rhs,
        )?;
        println!(
            "completed source-equivalence side check in {:?}",
            check_start.elapsed()
        );
        result
    };

    if config.skip_output_writing {
        println!("skipping stage output writing");
    } else {
        println!("writing stage outputs and summary");
        let write_start = Instant::now();
        write_stage_outputs(&topology, &coords, &truth_a.coeffs, OUT_DIR, &stage1)?;
        write_stage_outputs(&topology, &coords, &truth_a.coeffs, OUT_DIR, &stage2)?;
        write_stage_outputs(&topology, &coords, &truth_a.coeffs, OUT_DIR, &stage3)?;
        write_stage_outputs(&topology, &coords, &truth_a.coeffs, OUT_DIR, &stage4)?;
        write_summary(
            OUT_DIR,
            &stage1,
            &stage2,
            &stage3,
            &stage4,
            &truth_a.coeffs,
            &uq_subsets,
            &flux_measurements,
            source_equivalence,
        )?;
        println!("completed output writing in {:?}", write_start.elapsed());
    }

    println!("Magnetostatic linear PDE UQ validation");
    println!("mesh={}", config.mesh_path);
    println!(
        "variance_mode={} rbmc_probes={} rbmc_batches={} seed={} pde_variance={:.3e}",
        variance_mode_name(config.solver.variance.mode),
        config.solver.variance.num_rbmc_probes,
        config.solver.variance.rbmc_batch_count,
        config.solver.variance.rng_seed,
        config.pde_variance
    );
    println!(
        "state_dofs_hard={} state_dofs_soft={} residual_dofs_soft={}",
        hard_system.state_dimension(),
        soft_system.state_dimension(),
        soft_system.residual_dimension()
    );
    println!(
        "stage1_rel_l2={:.3e} stage2_rel_l2={:.3e} stage3_rel_l2={:.3e} stage4_rel_l2={:.3e}",
        stage1.relative_l2_error,
        stage2.relative_l2_error,
        stage3.relative_l2_error,
        stage4.relative_l2_error
    );
    println!(
        "stage1_rel_b_l2={:.3e} stage2_rel_b_l2={:.3e} stage3_rel_b_l2={:.3e} stage4_rel_b_l2={:.3e}",
        stage1.relative_b_l2_error,
        stage2.relative_b_l2_error,
        stage3.relative_b_l2_error,
        stage4.relative_b_l2_error
    );
    println!(
        "stage1_mean_norm_ratio={:.3e} stage2_mean_norm_ratio={:.3e} stage3_mean_norm_ratio={:.3e} stage4_mean_norm_ratio={:.3e}",
        stage1.posterior_mean_norm / stage1.truth_norm.max(1e-12),
        stage2.posterior_mean_norm / stage2.truth_norm.max(1e-12),
        stage3.posterior_mean_norm / stage3.truth_norm.max(1e-12),
        stage4.posterior_mean_norm / stage4.truth_norm.max(1e-12)
    );
    println!(
        "stage1_pde_residual_norm={:.3e} stage2_pde_residual_norm={:.3e} stage3_pde_residual_norm={:.3e} stage4_pde_residual_norm={:.3e}",
        stage1.pde_residual_norm,
        stage2.pde_residual_norm,
        stage3.pde_residual_norm,
        stage4.pde_residual_norm
    );
    println!(
        "stage1_sensor_rmse={:.3e} stage2_sensor_rmse={:.3e} stage3_sensor_rmse={:.3e} stage4_sensor_rmse={:.3e}",
        stage1.sensor_rmse,
        stage2.sensor_rmse,
        stage3.sensor_rmse,
        stage4.sensor_rmse
    );
    println!(
        "stage3_harmonic_residual={:.3e} stage4_harmonic_residual={:.3e}",
        stage3.harmonic_residual_norm, stage4.harmonic_residual_norm
    );
    print_key_transition_summaries(&stage1, &stage2, &stage3, &stage4, &uq_subsets);
    if config.skip_output_writing {
        println!("output writing was skipped");
    } else {
        println!("wrote outputs to {OUT_DIR}");
    }
    Ok(())
}

fn parse_args() -> Result<ExampleConfig, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mesh-path" => {
                config.mesh_path = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --mesh-path"))?;
            }
            "--variance-mode" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --variance-mode"))?;
                config.solver.variance.mode = parse_variance_mode(&value)?;
            }
            "--pde-variance" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --pde-variance"))?;
                config.pde_variance = parse_f64_arg(&value, "--pde-variance")?;
            }
            "--only-stage" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --only-stage"))?;
                config.only_stage = Some(parse_requested_stage(&value)?);
            }
            "--num-rbmc-probes" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --num-rbmc-probes"))?;
                config.solver.variance.num_rbmc_probes =
                    parse_usize_arg(&value, "--num-rbmc-probes")?;
            }
            "--rbmc-batch-count" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --rbmc-batch-count"))?;
                config.solver.variance.rbmc_batch_count =
                    parse_usize_arg(&value, "--rbmc-batch-count")?;
            }
            "--rbmc-seed" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --rbmc-seed"))?;
                config.solver.variance.rng_seed = parse_u64_arg(&value, "--rbmc-seed")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--enable-stabilization" => {
                config.solver.stabilize_precision = true;
            }
            "--debug-stage3" => {
                config.debug_stage3 = true;
            }
            "--debug-b-posterior-mc" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --debug-b-posterior-mc"))?;
                config.debug_b_posterior_mc_samples =
                    Some(parse_usize_arg(&value, "--debug-b-posterior-mc")?);
            }
            "--debug-b-face-count" => {
                let value = args
                    .next()
                    .ok_or_else(|| invalid_input("missing value for --debug-b-face-count"))?;
                config.debug_b_face_count = parse_usize_arg(&value, "--debug-b-face-count")?;
            }
            "--skip-output-writing" => {
                config.skip_output_writing = true;
            }
            "--skip-source-equivalence" => {
                config.skip_source_equivalence = true;
            }
            other => return Err(invalid_input(format!("unrecognized argument `{other}`")).into()),
        }
    }
    Ok(config)
}

fn print_usage() {
    println!("Usage: cargo run --release -p feg-infer --example magnetostatic_linear_pde_uq -- [options]");
    println!("  --mesh-path <path>      Override the input mesh path");
    println!("  --variance-mode <mode>   exact | rbmc | rbmc-clipped");
    println!("  --pde-variance <value>   Scalar PDE observation variance");
    println!("  --only-stage <stage>    stage1 | stage2 | stage3 | stage4");
    println!("  --enable-stabilization  Re-enable precision symmetrization/jitter");
    println!("  --num-rbmc-probes <n>    RBMC probe count");
    println!("  --rbmc-batch-count <n>   RBMC batch count");
    println!("  --rbmc-seed <n>          RBMC RNG seed");
    println!("  --debug-stage3          Run stage-3 subcase isolation and exit");
    println!(
        "  --debug-b-posterior-mc <n>  Compare transformed B variances against n posterior samples"
    );
    println!("  --debug-b-face-count <n>    Number of B faces to compare in the MC diagnostic");
    println!("  --skip-source-equivalence  Skip the source-equivalence side check");
    println!("  --skip-output-writing     Skip VTK/summary output writing");
}

fn parse_variance_mode(value: &str) -> Result<LinearPdeVarianceMode, Box<dyn Error>> {
    match value {
        "exact" => Ok(LinearPdeVarianceMode::Exact),
        "rbmc" => Ok(LinearPdeVarianceMode::Rbmc),
        "rbmc-clipped" => Ok(LinearPdeVarianceMode::RbmcClipped),
        _ => Err(invalid_input(format!(
            "invalid variance mode `{value}`; expected one of: exact, rbmc, rbmc-clipped"
        ))
        .into()),
    }
}

fn variance_mode_name(mode: LinearPdeVarianceMode) -> &'static str {
    match mode {
        LinearPdeVarianceMode::Exact => "exact",
        LinearPdeVarianceMode::Rbmc => "rbmc",
        LinearPdeVarianceMode::RbmcClipped => "rbmc-clipped",
    }
}

fn parse_requested_stage(value: &str) -> Result<RequestedStage, Box<dyn Error>> {
    match value {
        "stage1" => Ok(RequestedStage::Stage1),
        "stage2" => Ok(RequestedStage::Stage2),
        "stage3" => Ok(RequestedStage::Stage3),
        "stage4" => Ok(RequestedStage::Stage4),
        _ => Err(invalid_input(format!(
            "invalid stage `{value}`; expected one of: stage1, stage2, stage3, stage4"
        ))
        .into()),
    }
}

fn parse_usize_arg(value: &str, flag: &str) -> Result<usize, Box<dyn Error>> {
    value
        .parse::<usize>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn parse_u64_arg(value: &str, flag: &str) -> Result<u64, Box<dyn Error>> {
    value
        .parse::<u64>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")).into())
}

fn parse_f64_arg(value: &str, flag: &str) -> Result<f64, Box<dyn Error>> {
    let parsed = value
        .parse::<f64>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(invalid_input(format!(
            "invalid value for {flag}: expected a finite positive number"
        ))
        .into());
    }
    Ok(parsed)
}

fn invalid_input(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidInput, message.into())
}

fn outer_boundary_predicate(point: CoordRef<'_>, geom: ToroidalInductorGeometry) -> bool {
    let d = (geom.box_half_length - point[0].abs())
        .min(geom.box_half_length - point[1].abs())
        .min(geom.box_half_length - point[2].abs());
    d < 10.0 * geom.target_air_cell_size
}

fn toroidal_radius(point: CoordRef<'_>, geom: ToroidalInductorGeometry) -> f64 {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    ((rho - geom.major_radius).powi(2) + point[2] * point[2]).sqrt()
}

fn toroidal_direction(point: CoordRef<'_>) -> [f64; 3] {
    let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
    if rho < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        [-point[1] / rho, point[0] / rho, 0.0]
    }
}

fn coil_mode(geom: ToroidalInductorGeometry, mu_0: f64, sector: Option<usize>) -> DiffFormClosure {
    let j0 = 1.0;
    let sigma = 0.18;
    let eps = 0.03;
    DiffFormClosure::one_form(
        move |point| {
            let rho = (point[0] * point[0] + point[1] * point[1]).sqrt();
            if rho < 1e-12 {
                return FeecVector::from_column_slice(&[0.0, 0.0, 0.0]);
            }
            let angle = point[1].atan2(point[0]);
            if let Some(sector_index) = sector {
                let width = 2.0 * PI / SOURCE_MODE_COUNT as f64;
                let wrapped = (angle + PI).rem_euclid(2.0 * PI);
                let candidate = ((wrapped / width).floor() as usize).min(SOURCE_MODE_COUNT - 1);
                if candidate != sector_index {
                    return FeecVector::from_column_slice(&[0.0, 0.0, 0.0]);
                }
            }

            let s = toroidal_radius(point, geom);
            let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);
            let inner = geom.core_minor_radius + eps;
            let outer = geom.coil_minor_radius - eps;
            let tin = ((s - inner) / eps).clamp(0.0, 1.0);
            let tout = ((outer - s) / eps).clamp(0.0, 1.0);
            let cutoff = smoothstep(tin) * smoothstep(tout);
            let s0 = 0.5 * (geom.core_minor_radius + geom.coil_minor_radius);
            let gauss = (-((s - s0) * (s - s0)) / (sigma * sigma)).exp();
            let amplitude = mu_0 * j0 * gauss * cutoff;
            let direction = toroidal_direction(point);
            FeecVector::from_column_slice(&[
                amplitude * direction[0],
                amplitude * direction[1],
                amplitude * direction[2],
            ])
        },
        3,
    )
}

fn unit_toroidal_boundary_form() -> DiffFormClosure {
    DiffFormClosure::one_form(
        move |point| {
            let direction = toroidal_direction(point);
            FeecVector::from_column_slice(&direction)
        },
        3,
    )
}

fn assemble_weighted_source(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    inverse_permeability: &InnerProductWeightClosure,
    source: &DiffFormClosure,
) -> FeecVector {
    assemble_galvec(
        topology,
        metric,
        SourceElVec::new_weighted(source, coords, None, inverse_permeability),
    )
}

fn sorted_boundary_dofs<P>(
    topology: &Complex,
    coords: &MeshCoords,
    dim: usize,
    predicate: P,
) -> Vec<usize>
where
    P: Fn(CoordRef<'_>) -> bool + Sync,
{
    let mut dofs = assemble::boundary_simplices_where_barycenter(topology, coords, dim, predicate);
    dofs.sort_unstable();
    dofs
}

fn build_whittle_prior(
    system: &formoniq::problems::linear_uq::ReducedLinearPdeSystem,
    stabilize_precision: bool,
) -> GaussianPriorSpec {
    let laplacian = core_triplet_to_feec_csr(&system.operator);
    let mass = core_triplet_to_feec_csr(&system.state_mass);
    let mass_inverse = core_triplet_to_feec_csr(
        system
            .state_mass_inverse
            .as_ref()
            .expect("1-form reduced system should expose an NC1 projected mass inverse"),
    );
    let a = add_sparse(&laplacian, &scale_matrix(&mass, 1.0));
    let precision = &a.transpose() * &(&mass_inverse * &a);
    let precision = if stabilize_precision {
        stabilize_spd_precision(precision)
    } else {
        precision
    };
    GaussianPriorSpec {
        mean: vec![0.0; system.state_dimension()],
        precision: csr_to_triplet(&precision),
    }
}

fn solve_full_feec_deterministic_reference(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    inverse_permeability: &InnerProductWeightClosure,
    source_galvec: &FeecVector,
    geom: ToroidalInductorGeometry,
) -> Cochain {
    let strong_state_dofs = sorted_boundary_dofs(topology, coords, 1, |point| {
        outer_boundary_predicate(point, geom)
    })
    .into_iter()
    .collect::<HashSet<_>>();
    let strong_aux_dofs = sorted_boundary_dofs(topology, coords, 0, |point| {
        outer_boundary_predicate(point, geom)
    })
    .into_iter()
    .collect::<HashSet<_>>();
    let strong_state_predicate = |sidx: KSimplexIdx| strong_state_dofs.contains(&sidx);
    let strong_aux_predicate = |sidx: KSimplexIdx| strong_aux_dofs.contains(&sidx);
    let zero_data = |_sidx: KSimplexIdx| 0.0;

    let (_, truth_a, _) =
        hodge_laplace::solve_weighted_hodge_laplace_source_with_boundary_conditions(
            topology,
            metric,
            None,
            source_galvec.clone(),
            1,
            1,
            coords,
            None,
            inverse_permeability,
            &strong_state_predicate,
            &zero_data,
            &strong_aux_predicate,
            &zero_data,
        );
    truth_a
}

fn stabilize_spd_precision(mut precision: FeecCsr) -> FeecCsr {
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        return precision;
    }

    precision = symmetrize_feec_csr(&precision);
    if feec_csr_to_gmrf(&precision).cholesky_sqrt_lower().is_ok() {
        return precision;
    }

    let (min_diag, max_abs_diag) = diagonal_stats_feec(&precision);
    let mut shift = if min_diag.is_finite() && min_diag <= 0.0 {
        (-min_diag) + max_abs_diag * 1e-8
    } else {
        max_abs_diag * 1e-12
    }
    .max(1e-10);
    for _ in 0..12 {
        let shifted = add_diagonal_shift(&precision, shift);
        if feec_csr_to_gmrf(&shifted).cholesky_sqrt_lower().is_ok() {
            return shifted;
        }
        shift *= 10.0;
        precision = shifted;
    }
    precision
}

fn symmetrize_feec_csr(matrix: &FeecCsr) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    FeecCsr::from(&coo)
}

fn diagonal_stats_feec(matrix: &FeecCsr) -> (f64, f64) {
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let min_diag = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max_abs_diag = diagonal.iter().copied().map(f64::abs).fold(0.0, f64::max);
    (min_diag, max_abs_diag.max(1.0))
}

fn build_source_mode_operator(
    topology: &Complex,
    metric: &MeshLengths,
    coords: &MeshCoords,
    inverse_permeability: &InnerProductWeightClosure,
    galmats: &MixedGalmats,
    boundary: &BoundarySpec,
    geom: ToroidalInductorGeometry,
    mu_0: f64,
) -> Result<SparseTripletMatrix, String> {
    let mut columns = Vec::with_capacity(SOURCE_MODE_COUNT);
    for sector in 0..SOURCE_MODE_COUNT {
        let mode = coil_mode(geom, mu_0, Some(sector));
        let rhs = assemble_weighted_source(topology, metric, coords, inverse_permeability, &mode);
        let reduced = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
            galmats,
            boundary,
            &FeecVector::zeros(galmats.sigma_len()),
            &rhs,
        )?;
        columns.push(reduced.scale(-1.0));
    }
    Ok(columns_to_sparse_matrix(&columns))
}

fn build_core_neumann_operator(
    topology: &Complex,
    coords: &MeshCoords,
    galmats: &MixedGalmats,
    boundary: &BoundarySpec,
    geom: ToroidalInductorGeometry,
) -> Result<SparseTripletMatrix, String> {
    let core_faces = sorted_boundary_dofs(topology, coords, 2, |point| {
        (toroidal_radius(point, geom) - geom.core_minor_radius).abs()
            <= 3.0 * geom.target_air_cell_size
    });
    let core_face_set = core_faces.iter().copied().collect::<HashSet<_>>();
    let toroidal_form = unit_toroidal_boundary_form();
    let mut columns = Vec::with_capacity(2);

    let upper_rhs =
        assemble_boundary_integral_term(topology, coords, 1, &toroidal_form, None, &|kidx| {
            face_in_core_patch(kidx, coords, topology, &core_face_set, |point| {
                point[2] >= 0.0
            })
        });
    let upper_reduced = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        galmats,
        boundary,
        &FeecVector::zeros(galmats.sigma_len()),
        &upper_rhs,
    )?;
    columns.push(scale_to_target_norm(&upper_reduced.scale(-1.0), 0.25));

    let lower_rhs =
        assemble_boundary_integral_term(topology, coords, 1, &toroidal_form, None, &|kidx| {
            face_in_core_patch(kidx, coords, topology, &core_face_set, |point| {
                point[2] < 0.0
            })
        });
    let lower_reduced = reduce_reduced_hodge_laplace_1form_rhs_with_galmats(
        galmats,
        boundary,
        &FeecVector::zeros(galmats.sigma_len()),
        &lower_rhs,
    )?;
    columns.push(scale_to_target_norm(&lower_reduced.scale(-1.0), 0.25));

    Ok(columns_to_sparse_matrix(&columns))
}

fn face_in_core_patch<P>(
    kidx: KSimplexIdx,
    coords: &MeshCoords,
    topology: &Complex,
    core_face_set: &HashSet<usize>,
    patch_predicate: P,
) -> bool
where
    P: Fn(CoordRef<'_>) -> bool,
{
    if !core_face_set.contains(&kidx) {
        return false;
    }
    let face = SimplexIdx::new(2, kidx).handle(topology);
    let face_coords = SimplexCoords::from_simplex_and_coords(&face, coords);
    patch_predicate(face_coords.barycenter().as_view())
}

fn build_weighted_harmonic_measurements(
    topology: &Complex,
    galmats: &MixedGalmats,
    system: &formoniq::problems::linear_uq::ReducedLinearPdeSystem,
    strong_aux_dofs: &[usize],
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
    let strong_aux_set = strong_aux_dofs.iter().copied().collect::<HashSet<_>>();
    let harmonics = hodge_laplace::solve_hodge_laplace_harmonics_with_galmats(
        topology,
        galmats,
        1,
        1,
        Some(&|kidx| strong_aux_set.contains(&kidx)),
        None,
    );
    let constraints =
        build_harmonic_orthogonality_constraints(&harmonics, &FeecCsr::from(galmats.mass_u()))?;
    Ok(dense_rows_to_measurements(
        &constraints,
        system.layout.full_dimension,
        1e-10,
        "harmonic",
    ))
}

fn build_flux_loop_measurements(
    topology: &Complex,
    coords: &MeshCoords,
    truth_a: FeecVector,
    geom: ToroidalInductorGeometry,
) -> Result<Vec<LinearGaussianMeasurementSpec>, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    let rows = csr_rows(&d1);
    let sensors = [
        ("flux_inner", [geom.major_radius - 1.05, 0.0, 0.00]),
        ("flux_top", [geom.major_radius, 0.0, 1.05]),
        ("flux_outer", [geom.major_radius + 1.05, 0.0, 0.00]),
    ];
    let mut measurements = Vec::with_capacity(sensors.len());

    for (name, center) in sensors {
        let operator = build_flux_patch_operator(topology, coords, &rows, center, 0.45, 0.18)?;
        let predicted = sparse_measurement_values(&operator, &[0.0], &truth_a)?;
        measurements.push(LinearGaussianMeasurementSpec {
            name: name.to_string(),
            operator,
            observations: predicted.iter().copied().collect(),
            bias: vec![0.0],
            variance: 1e-6,
        });
    }
    Ok(measurements)
}

fn build_magnetic_field_derived_quantities(
    topology: &Complex,
    coords: &MeshCoords,
) -> Result<Vec<LinearPdeDerivedQuantitySpec>, String> {
    if topology.dim() != 3 || coords.dim() != 3 {
        return Err(format!(
            "magnetic-field derived quantities require a 3D mesh, got topology dim {} and coordinate dim {}",
            topology.dim(),
            coords.dim()
        ));
    }

    let magnetic_field = build_exterior_derivative_row_operator(topology)?;
    let magnetic_field_vector_x = compose_sparse_row_operators(
        &build_sampled_magnetic_field_component_operator(topology, coords, 0)?,
        &magnetic_field,
    )?;
    let magnetic_field_vector_y = compose_sparse_row_operators(
        &build_sampled_magnetic_field_component_operator(topology, coords, 1)?,
        &magnetic_field,
    )?;
    let magnetic_field_vector_z = compose_sparse_row_operators(
        &build_sampled_magnetic_field_component_operator(topology, coords, 2)?,
        &magnetic_field,
    )?;

    Ok(vec![
        LinearPdeDerivedQuantitySpec {
            name: MAGNETIC_FIELD_DERIVED_NAME.to_string(),
            operator: magnetic_field,
        },
        LinearPdeDerivedQuantitySpec {
            name: MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME.to_string(),
            operator: magnetic_field_vector_x,
        },
        LinearPdeDerivedQuantitySpec {
            name: MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME.to_string(),
            operator: magnetic_field_vector_y,
        },
        LinearPdeDerivedQuantitySpec {
            name: MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME.to_string(),
            operator: magnetic_field_vector_z,
        },
    ])
}

fn build_exterior_derivative_row_operator(topology: &Complex) -> Result<SparseRowOperator, String> {
    let d1 = FeecCsr::from(&topology.exterior_derivative_operator(1));
    SparseRowOperator::new(d1.ncols(), csr_rows(&d1)).map_err(|err| err.to_string())
}

fn build_sampled_magnetic_field_component_operator(
    topology: &Complex,
    coords: &MeshCoords,
    component_index: usize,
) -> Result<SparseRowOperator, String> {
    if component_index >= 3 {
        return Err(format!(
            "magnetic-field vector component index {component_index} is out of range"
        ));
    }

    let topo_dim = topology.dim();
    if topo_dim != 3 || coords.dim() != 3 {
        return Err(format!(
            "sampled magnetic-field vector operator requires a 3D mesh, got topology dim {} and coordinate dim {}",
            topo_dim,
            coords.dim()
        ));
    }

    let bary_local = barycenter_local(topo_dim);
    let mut rows = Vec::with_capacity(topology.cells().len());
    for cell in topology.skeleton(topo_dim).handle_iter() {
        let cell_coords = cell.coord_simplex(coords);
        let mut row = Vec::new();
        for dof_simp in cell.mesh_subsimps(2) {
            let local_dof_simp = dof_simp.relative_to(&cell);
            let lsf = WhitneyLsf::standard(topo_dim, local_dof_simp);
            let ambient_value = cell_coords.lift_form(&lsf.at_point(&bary_local));
            let coeffs = ambient_value.coeffs();
            if coeffs.len() != 3 {
                return Err(format!(
                    "expected three 2-form coefficients for a 3D cell, found {}",
                    coeffs.len()
                ));
            }
            let coefficient = match component_index {
                0 => coeffs[2],
                1 => -coeffs[1],
                2 => coeffs[0],
                _ => unreachable!(),
            };
            if coefficient.abs() > 1e-12 {
                row.push((dof_simp.kidx(), coefficient));
            }
        }
        rows.push(row);
    }

    SparseRowOperator::new(topology.nsimplices(2), rows).map_err(|err| err.to_string())
}

fn compose_sparse_row_operators(
    lhs: &SparseRowOperator,
    rhs: &SparseRowOperator,
) -> Result<SparseRowOperator, String> {
    if lhs.ncols != rhs.nrows() {
        return Err(format!(
            "cannot compose sparse row operators with dimensions {}x{} and {}x{}",
            lhs.nrows(),
            lhs.ncols,
            rhs.nrows(),
            rhs.ncols
        ));
    }

    let mut rows = Vec::with_capacity(lhs.nrows());
    for lhs_row in &lhs.rows {
        let mut entries = BTreeMap::<usize, f64>::new();
        for (mid, lhs_value) in lhs_row {
            for (col, rhs_value) in &rhs.rows[*mid] {
                *entries.entry(*col).or_insert(0.0) += *lhs_value * *rhs_value;
            }
        }
        rows.push(
            entries
                .into_iter()
                .filter(|(_, value)| value.abs() > 1e-12)
                .collect(),
        );
    }

    SparseRowOperator::new(rhs.ncols, rows).map_err(|err| err.to_string())
}

fn build_flux_patch_operator(
    topology: &Complex,
    coords: &MeshCoords,
    face_rows: &[Vec<(usize, f64)>],
    center: [f64; 3],
    patch_radius: f64,
    y_half_width: f64,
) -> Result<SparseTripletMatrix, String> {
    let mut edge_weights = BTreeMap::<usize, f64>::new();
    let mut selected_face_count = 0usize;
    for face_index in 0..topology.nsimplices(2) {
        let face = SimplexIdx::new(2, face_index).handle(topology);
        let face_coords = SimplexCoords::from_simplex_and_coords(&face, coords);
        let bary = face_coords.barycenter();
        let dx = bary[0] - center[0];
        let dy = bary[1] - center[1];
        let dz = bary[2] - center[2];
        let radial = (dx * dx + dz * dz).sqrt();
        if radial > patch_radius || dy.abs() > y_half_width {
            continue;
        }

        let normal = face_normal(&face_coords);
        let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm <= 1e-12 {
            continue;
        }
        let alignment = normal[1].abs() / norm;
        if alignment < 0.55 {
            continue;
        }
        let sign = if normal[1] >= 0.0 { 1.0 } else { -1.0 };
        selected_face_count += 1;
        for (edge, value) in &face_rows[face_index] {
            *edge_weights.entry(*edge).or_insert(0.0) += sign * *value;
        }
    }

    if selected_face_count == 0 {
        return Err(format!(
            "flux sensor patch at ({:.3}, {:.3}, {:.3}) selected no faces",
            center[0], center[1], center[2]
        ));
    }

    Ok(SparseTripletMatrix::from_triplets(
        1,
        topology.nsimplices(1),
        edge_weights
            .into_iter()
            .filter(|(_, value)| value.abs() > 1e-12)
            .map(|(col, value)| SparseTriplet { row: 0, col, value }),
    ))
}

fn run_stage(
    name: &str,
    solver_config: &LinearPdeUqSolverConfig,
    problem: LinearPdeUqProblem,
    topology: &Complex,
    truth_a: &FeecVector,
    flux_measurements: &[LinearGaussianMeasurementSpec],
    harmonic_measurements: &[LinearGaussianMeasurementSpec],
) -> Result<StageOutcome, String> {
    let result = solve_linear_pde_uq_with_config(&problem, solver_config)?;
    print_factorization_summary(name, &result);
    let truth_norm = truth_a.norm();
    let posterior_mean_norm = result.posterior_mean.norm();
    let relative_l2_error = (&result.posterior_mean - truth_a).norm() / truth_norm.max(1e-12);
    let posterior_b = Cochain::new(1, result.posterior_mean.clone()).dif(topology);
    let truth_b = Cochain::new(1, truth_a.clone()).dif(topology);
    let relative_b_l2_error =
        (&posterior_b.coeffs - &truth_b.coeffs).norm() / truth_b.coeffs.norm().max(1e-12);
    let pde_residual_norm = result.pde_residual_mean.norm();
    let sensor_predictions = evaluate_measurements(flux_measurements, &result.posterior_mean)?;
    let sensor_truth = flux_measurements
        .iter()
        .map(|measurement| measurement.observations[0])
        .collect::<Vec<_>>();
    let sensor_rmse = rmse(&sensor_predictions, &sensor_truth)?;
    let harmonic_residual_norm = if harmonic_measurements.is_empty() {
        0.0
    } else {
        evaluate_measurements(harmonic_measurements, &result.posterior_mean)?.norm()
    };
    Ok(StageOutcome {
        name: name.to_string(),
        result,
        relative_l2_error,
        relative_b_l2_error,
        posterior_mean_norm,
        truth_norm,
        pde_residual_norm,
        sensor_predictions,
        sensor_rmse,
        harmonic_residual_norm,
    })
}

fn print_stage_uq_summary(stage: &StageOutcome, truth: &FeecVector, subsets: &UqSubsetIndices) {
    for (label, indices) in uq_subset_entries(subsets) {
        let summary = summarize_stage_uncertainty(truth, stage, indices);
        println!(
            "{} subset={} count={} prior_var_mean={:.3e} post_var_mean={:.3e} ratio_mean={:.3e} reduction_mean={:.3e} abs_error_mean={:.3e} norm_abs_error_mean={:.3e} within_1sigma={:.3} within_2sigma={:.3}",
            stage.name,
            label,
            summary.count,
            summary.prior_variance_mean,
            summary.posterior_variance_mean,
            summary.variance_ratio_mean,
            summary.variance_reduction_mean,
            summary.absolute_error_mean,
            summary.normalized_abs_error_mean,
            summary.truth_within_1sigma_fraction,
            summary.truth_within_2sigma_fraction
        );
    }
}

fn debug_magnetic_field_posterior_variance(
    stage: &StageOutcome,
    problem: &LinearPdeUqProblem,
    solver_config: &LinearPdeUqSolverConfig,
    num_samples: usize,
    face_count: usize,
    output_dir: Option<std::path::PathBuf>,
) -> Result<(), String> {
    if num_samples == 0 {
        return Err("magnetic-field posterior debug requires at least one sample".to_string());
    }
    if face_count == 0 {
        return Err("magnetic-field posterior debug requires at least one face".to_string());
    }

    let reported = stage
        .result
        .derived_variances
        .get(MAGNETIC_FIELD_DERIVED_NAME)
        .ok_or_else(|| {
            format!(
                "stage `{}` is missing derived quantity `{MAGNETIC_FIELD_DERIVED_NAME}`",
                stage.name
            )
        })?;
    let selected_faces = largest_indices_by_value(&reported.posterior_variance, face_count);
    let joint_posterior = build_linear_pde_joint_posterior_with_config(problem, solver_config)?;
    let magnetic_field_operator = joint_posterior
        .derived_quantities
        .get(MAGNETIC_FIELD_DERIVED_NAME)
        .ok_or_else(|| {
            format!("joint posterior is missing derived quantity `{MAGNETIC_FIELD_DERIVED_NAME}`")
        })?;
    let selected_operator = subset_sparse_row_operator(magnetic_field_operator, &selected_faces)?;

    let mut posterior = joint_posterior.posterior;
    let constraints = GmrfDenseMatrix::zeros(0, posterior.dimension());
    let exact = posterior
        .exact_transformed_variance_decomposition(&selected_operator, &constraints)
        .map_err(|err| format!("failed to compute exact magnetic-field marginals: {err}"))?;
    let mc = estimate_operator_mc_variances(
        &posterior,
        &selected_operator,
        num_samples,
        solver_config.variance.rng_seed.wrapping_add(1),
    )?;

    let report = MagneticFieldPosteriorVarianceDebugReport {
        stage_name: stage.name.clone(),
        variance_mode: solver_config.variance.mode,
        num_samples,
        face_count: selected_faces.len(),
        rows: selected_faces
            .iter()
            .enumerate()
            .map(
                |(row_index, face_index)| MagneticFieldPosteriorVarianceDebugRow {
                    face_index: *face_index,
                    prior_variance: reported.prior_variance[*face_index],
                    reported_posterior_variance: reported.posterior_variance[*face_index],
                    exact_posterior_variance: exact.constrained_diag[row_index],
                    mc_posterior_variance: mc[row_index],
                },
            )
            .collect(),
    };
    let formatted = format_magnetic_field_posterior_debug_report(&report);
    print!("{formatted}");
    if let Some(dir) = output_dir {
        fs::write(dir.join("b_posterior_variance_mc_debug.txt"), formatted).map_err(|err| {
            format!("failed to write magnetic-field posterior debug report: {err}")
        })?;
    }
    Ok(())
}

fn largest_indices_by_value(values: &FeecVector, count: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count.min(indices.len()));
    indices
}

fn subset_sparse_row_operator(
    operator: &SparseRowOperator,
    row_indices: &[usize],
) -> Result<SparseRowOperator, String> {
    let rows = row_indices
        .iter()
        .map(|index| {
            operator
                .rows
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("operator row index {} is out of bounds", index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    SparseRowOperator::new(operator.ncols, rows).map_err(|err| err.to_string())
}

fn estimate_operator_mc_variances(
    posterior: &gmrf_core::Gmrf,
    operator: &SparseRowOperator,
    num_samples: usize,
    rng_seed: u64,
) -> Result<GmrfVector, String> {
    let output_dim = operator.nrows();
    let mut means = vec![0.0; output_dim];
    let mut m2 = vec![0.0; output_dim];
    let mut sample_count = 0.0;
    let mut rng = rand::rngs::StdRng::seed_from_u64(rng_seed);

    for _ in 0..num_samples {
        let draw = posterior
            .sample_one_solve(&mut rng)
            .map_err(|err| format!("failed to sample posterior draw: {err}"))?;
        let transformed = operator
            .apply(&draw)
            .map_err(|err| format!("failed to apply magnetic-field operator to draw: {err}"))?;
        sample_count += 1.0;
        for row in 0..output_dim {
            let delta = transformed[row] - means[row];
            means[row] += delta / sample_count;
            let delta2 = transformed[row] - means[row];
            m2[row] += delta * delta2;
        }
    }

    Ok(GmrfVector::from_iterator(
        output_dim,
        (0..output_dim).map(|row| m2[row] / sample_count),
    ))
}

fn format_magnetic_field_posterior_debug_report(
    report: &MagneticFieldPosteriorVarianceDebugReport,
) -> String {
    let mut text = String::new();
    text.push_str(&format!(
        "{} magnetic_field_posterior_debug variance_mode={} samples={} selected_faces={}\n",
        report.stage_name,
        variance_mode_name(report.variance_mode),
        report.num_samples,
        report.face_count,
    ));
    for (rank, row) in report.rows.iter().enumerate() {
        text.push_str(&format!(
            "  rank={} face={} prior={:.8e} reported_post={:.8e} exact_post={:.8e} mc_post={:.8e} reported/exact={:.3e} mc/exact={:.3e} exact/prior={:.3e}\n",
            rank,
            row.face_index,
            row.prior_variance,
            row.reported_posterior_variance,
            row.exact_posterior_variance,
            row.mc_posterior_variance,
            debug_ratio(row.reported_posterior_variance, row.exact_posterior_variance),
            debug_ratio(row.mc_posterior_variance, row.exact_posterior_variance),
            debug_ratio(row.exact_posterior_variance, row.prior_variance),
        ));
    }
    if !report.rows.is_empty() {
        let mean_prior = report
            .rows
            .iter()
            .map(|row| row.prior_variance)
            .sum::<f64>()
            / report.rows.len() as f64;
        let mean_reported = report
            .rows
            .iter()
            .map(|row| row.reported_posterior_variance)
            .sum::<f64>()
            / report.rows.len() as f64;
        let mean_exact = report
            .rows
            .iter()
            .map(|row| row.exact_posterior_variance)
            .sum::<f64>()
            / report.rows.len() as f64;
        let mean_mc = report
            .rows
            .iter()
            .map(|row| row.mc_posterior_variance)
            .sum::<f64>()
            / report.rows.len() as f64;
        text.push_str(&format!(
            "  means prior={:.8e} reported_post={:.8e} exact_post={:.8e} mc_post={:.8e} reported/exact={:.3e} mc/exact={:.3e} exact/prior={:.3e}\n",
            mean_prior,
            mean_reported,
            mean_exact,
            mean_mc,
            debug_ratio(mean_reported, mean_exact),
            debug_ratio(mean_mc, mean_exact),
            debug_ratio(mean_exact, mean_prior),
        ));
    }
    text
}

fn print_key_transition_summaries(
    stage1: &StageOutcome,
    stage2: &StageOutcome,
    stage3: &StageOutcome,
    stage4: &StageOutcome,
    subsets: &UqSubsetIndices,
) {
    for (label, summary) in [
        (
            "stage2_over_stage1_coil",
            summarize_stage_transition(stage1, stage2, &subsets.coil_edges),
        ),
        (
            "stage3_over_stage2_outer_boundary",
            summarize_stage_transition(stage2, stage3, &subsets.outer_boundary_edges),
        ),
        (
            "stage3_over_stage2_core_boundary",
            summarize_stage_transition(stage2, stage3, &subsets.core_boundary_edges),
        ),
        (
            "stage4_over_stage3_sensor",
            summarize_stage_transition(stage3, stage4, &subsets.sensor_edges),
        ),
        (
            "stage4_over_stage3_background",
            summarize_stage_transition(stage3, stage4, &subsets.background_edges),
        ),
    ] {
        println!(
            "{} count={} baseline_post_var_mean={:.3e} comparison_post_var_mean={:.3e} delta_mean={:.3e} mean_ratio={:.3e} pointwise_ratio_mean={:.3e}",
            label,
            summary.count,
            summary.baseline_posterior_variance_mean,
            summary.comparison_posterior_variance_mean,
            summary.posterior_variance_delta_mean,
            summary.posterior_variance_mean_ratio,
            summary.posterior_variance_pointwise_ratio_mean
        );
    }
}

fn print_factorization_summary(stage_name: &str, result: &LinearPdeUqResult) {
    let prior = result.debug.prior_factorization;
    let posterior = result.debug.posterior_factorization;
    println!(
        "{stage_name} prior_factorization dim={} matrix_nnz={} lower_triangle_nnz={} factor_nnz={} fill_in_vs_lower={:.3}x factor_values_mib={:.3}",
        prior.dimension,
        prior.matrix_nnz,
        prior.matrix_lower_triangle_nnz,
        prior.factor_nnz,
        prior.fill_in_ratio_vs_lower_triangle,
        prior.factor_numeric_values_mib
    );
    println!(
        "{stage_name} posterior_factorization dim={} matrix_nnz={} lower_triangle_nnz={} factor_nnz={} fill_in_vs_lower={:.3}x factor_values_mib={:.3}",
        posterior.dimension,
        posterior.matrix_nnz,
        posterior.matrix_lower_triangle_nnz,
        posterior.factor_nnz,
        posterior.fill_in_ratio_vs_lower_triangle,
        posterior.factor_numeric_values_mib
    );
}

fn run_source_equivalence_side_check(
    solver_config: &LinearPdeUqSolverConfig,
    prior: &GaussianPriorSpec,
    system: &formoniq::problems::linear_uq::ReducedLinearPdeSystem,
    source_rhs: &FeecVector,
) -> Result<f64, String> {
    let residual_dim = system.residual_dimension();
    let residual_precision = diagonal_precision(residual_dim, 4.0);
    let latent = solve_linear_pde_uq_with_config(
        &LinearPdeUqProblem {
            state_prior: prior.clone(),
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "residual_source_latent".to_string(),
                operator: system.forcing_operator.clone(),
                prior: GaussianPriorSpec {
                    mean: source_rhs.iter().copied().collect(),
                    precision: residual_precision.clone(),
                },
                preference: RepresentationPreference::ForceLatent,
                collapsed_precision: None,
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: Some(0.25),
        },
        solver_config,
    )?;
    let collapsed = solve_linear_pde_uq_with_config(
        &LinearPdeUqProblem {
            state_prior: prior.clone(),
            system: system.clone(),
            uncertain_inputs: vec![LinearUncertainInputSpec {
                name: "residual_source_collapsed".to_string(),
                operator: system.forcing_operator.clone(),
                prior: GaussianPriorSpec {
                    mean: source_rhs.iter().copied().collect(),
                    precision: residual_precision,
                },
                preference: RepresentationPreference::Auto,
                collapsed_precision: Some(diagonal_precision(residual_dim, 1.0 / 0.5)),
            }],
            physical_measurements: Vec::new(),
            derived_quantities: Vec::new(),
            pde_variance: None,
        },
        solver_config,
    )?;
    Ok(max_abs_difference(
        &latent.posterior_mean,
        &collapsed.posterior_mean,
    ))
}

fn run_stage3_debug_cases(
    solver_config: &LinearPdeUqSolverConfig,
    soft_prior: &GaussianPriorSpec,
    soft_system: &formoniq::problems::linear_uq::ReducedLinearPdeSystem,
    soft_source_operator: &SparseTripletMatrix,
    soft_neumann_operator: &SparseTripletMatrix,
    harmonic_measurements: &[LinearGaussianMeasurementSpec],
) -> Result<(), Box<dyn Error>> {
    let source_input = LinearUncertainInputSpec {
        name: "coil_source".to_string(),
        operator: soft_source_operator.clone(),
        prior: GaussianPriorSpec {
            mean: vec![1.0; SOURCE_MODE_COUNT],
            precision: diagonal_precision(SOURCE_MODE_COUNT, 1.0 / (0.35 * 0.35)),
        },
        preference: RepresentationPreference::ForceLatent,
        collapsed_precision: None,
    };
    let neumann_input = LinearUncertainInputSpec {
        name: "core_neumann".to_string(),
        operator: soft_neumann_operator.clone(),
        prior: GaussianPriorSpec {
            mean: vec![0.0; 2],
            precision: diagonal_precision(2, 1.0),
        },
        preference: RepresentationPreference::ForceLatent,
        collapsed_precision: None,
    };

    let cases = vec![
        ("soft_pde_only", vec![], vec![]),
        ("soft_plus_source", vec![source_input.clone()], vec![]),
        ("soft_plus_neumann", vec![neumann_input.clone()], vec![]),
        ("soft_plus_harmonic", vec![], harmonic_measurements.to_vec()),
        (
            "soft_plus_source_harmonic",
            vec![source_input.clone()],
            harmonic_measurements.to_vec(),
        ),
        (
            "soft_plus_neumann_harmonic",
            vec![neumann_input.clone()],
            harmonic_measurements.to_vec(),
        ),
        (
            "soft_plus_source_neumann",
            vec![source_input.clone(), neumann_input.clone()],
            vec![],
        ),
        (
            "soft_plus_source_neumann_harmonic",
            vec![source_input, neumann_input],
            harmonic_measurements.to_vec(),
        ),
    ];

    println!("Stage 3 debug cases");
    print_soft_pde_only_debug_stats(soft_prior, soft_system)?;
    for (name, uncertain_inputs, physical_measurements) in cases {
        let problem = LinearPdeUqProblem {
            state_prior: soft_prior.clone(),
            system: soft_system.clone(),
            uncertain_inputs,
            physical_measurements,
            derived_quantities: Vec::new(),
            pde_variance: Some(1e-4),
        };
        match solve_linear_pde_uq_with_config(&problem, solver_config) {
            Ok(_) => println!("{name}: ok"),
            Err(err) => println!("{name}: fail: {err}"),
        }
    }
    Ok(())
}

fn print_soft_pde_only_debug_stats(
    soft_prior: &GaussianPriorSpec,
    soft_system: &formoniq::problems::linear_uq::ReducedLinearPdeSystem,
) -> Result<(), Box<dyn Error>> {
    let prior_precision = csr_to_gmrf_sparse(&core_triplet_to_feec_csr(&soft_prior.precision));
    let state_operator = feec_csr_to_gmrf(&core_triplet_to_feec_csr(&soft_system.operator));
    let pde_bias = GmrfVector::from_vec(soft_system.residual_bias.clone());
    let zero_observations = GmrfVector::zeros(soft_system.residual_dimension());

    let mut builder = LinearObservationStackBuilder::new(soft_system.state_dimension());
    builder.push_block(
        0,
        &state_operator,
        zero_observations.as_slice(),
        pde_bias.as_slice(),
        1e-4,
    )?;
    for measurement in &soft_system.boundary_measurements {
        builder.push_block(
            0,
            &feec_csr_to_gmrf(&core_triplet_to_feec_csr(&measurement.operator)),
            &measurement.observations,
            &measurement.bias,
            measurement.variance,
        )?;
    }
    let stacked = builder.finish();
    let (posterior_precision, _) = apply_gaussian_observations(
        &prior_precision,
        &stacked.matrix,
        &stacked.observations,
        Some(&stacked.bias),
        stacked.noise_variance,
    );

    let (prior_min_diag, prior_max_diag) = diagonal_min_max(&prior_precision);
    let (post_min_diag, post_max_diag) = diagonal_min_max(&posterior_precision);
    let sym = symmetrize_gmrf_sparse(&posterior_precision);
    let rel_shift = post_max_diag.abs().max(1.0) * 1e-8;
    println!(
        "soft_pde_only_debug prior_diag=[{prior_min_diag:.3e},{prior_max_diag:.3e}] post_diag=[{post_min_diag:.3e},{post_max_diag:.3e}] rel_shift={rel_shift:.3e}"
    );
    println!(
        "soft_pde_only_debug chol raw_prior={} raw_post={} sym_post={} sym_shift_post={}",
        prior_precision.cholesky_sqrt_lower().is_ok(),
        posterior_precision.cholesky_sqrt_lower().is_ok(),
        sym.cholesky_sqrt_lower().is_ok(),
        add_diagonal_shift_gmrf(&sym, rel_shift)
            .cholesky_sqrt_lower()
            .is_ok()
    );
    Ok(())
}

fn diagonal_min_max(matrix: &GmrfSparseMatrix) -> (f64, f64) {
    let mut diagonal = vec![0.0; matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            diagonal[row] += *value;
        }
    }
    let min = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    let max = diagonal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn symmetrize_gmrf_sparse(matrix: &GmrfSparseMatrix) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        if row == col {
            coo.push(row, col, *value);
        } else {
            coo.push(row, col, 0.5 * *value);
            coo.push(col, row, 0.5 * *value);
        }
    }
    GmrfSparseMatrix::from(&coo)
}

fn add_diagonal_shift_gmrf(matrix: &GmrfSparseMatrix, shift: f64) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    GmrfSparseMatrix::from(&coo)
}

fn csr_to_gmrf_sparse(matrix: &FeecCsr) -> GmrfSparseMatrix {
    let mut coo = GmrfCoo::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    GmrfSparseMatrix::from(&coo)
}

fn write_truth_outputs(
    topology: &Complex,
    coords: &MeshCoords,
    truth_a: &Cochain,
    out_dir: &str,
) -> Result<(), Box<dyn Error>> {
    write_cochain_vtk(
        format!("{out_dir}/truth_A.vtk"),
        coords,
        topology,
        truth_a,
        "truth_A",
    )?;
    write_1form_vector_field_vtk(
        format!("{out_dir}/truth_A_vector.vtk"),
        coords,
        topology,
        truth_a,
        "truth_A_vector",
    )?;
    let truth_b = truth_a.dif(topology);
    write_cochain_vtk(
        format!("{out_dir}/truth_B.vtk"),
        coords,
        topology,
        &truth_b,
        "truth_B",
    )?;
    write_2form_vector_field_vtk(
        format!("{out_dir}/truth_B_vector.vtk"),
        coords,
        topology,
        &truth_b,
        "truth_B_vector",
    )?;
    Ok(())
}

fn write_stage_outputs(
    topology: &Complex,
    coords: &MeshCoords,
    truth_a: &FeecVector,
    out_dir: &str,
    stage: &StageOutcome,
) -> Result<(), Box<dyn Error>> {
    let stage_dir = format!("{out_dir}/{}", stage.name);
    fs::create_dir_all(&stage_dir)?;
    let posterior_mean = Cochain::new(1, stage.result.posterior_mean.clone());
    let truth = Cochain::new(1, truth_a.clone());
    let truth_b = truth.dif(topology);
    let prior_variance = Cochain::new(1, stage.result.prior_variance.clone());
    let posterior_variance = Cochain::new(1, stage.result.posterior_variance.clone());
    let variance_ratio = Cochain::new(
        1,
        pointwise_variance_ratio(
            &stage.result.prior_variance,
            &stage.result.posterior_variance,
        ),
    );
    let absolute_error = Cochain::new(
        1,
        FeecVector::from_iterator(
            truth_a.len(),
            (0..truth_a.len()).map(|idx| (stage.result.posterior_mean[idx] - truth_a[idx]).abs()),
        ),
    );
    write_1cochain_vtk_fields(
        format!("{stage_dir}/A_fields.vtk"),
        coords,
        topology,
        &[
            ("truth_A", &truth),
            ("posterior_A", &posterior_mean),
            ("prior_variance", &prior_variance),
            ("posterior_variance", &posterior_variance),
            ("variance_ratio", &variance_ratio),
            ("absolute_error", &absolute_error),
        ],
    )?;
    write_1form_vector_field_vtk(
        format!("{stage_dir}/A_vector.vtk"),
        coords,
        topology,
        &posterior_mean,
        "posterior_A_vector",
    )?;
    let posterior_b = posterior_mean.dif(topology);
    let magnetic_field_variance = stage
        .result
        .derived_variances
        .get(MAGNETIC_FIELD_DERIVED_NAME)
        .ok_or_else(|| {
            invalid_input(format!(
                "missing derived variance output `{MAGNETIC_FIELD_DERIVED_NAME}`"
            ))
        })?;
    let magnetic_field_vector_x = stage
        .result
        .derived_variances
        .get(MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME)
        .ok_or_else(|| {
            invalid_input(format!(
                "missing derived variance output `{MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME}`"
            ))
        })?;
    let magnetic_field_vector_y = stage
        .result
        .derived_variances
        .get(MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME)
        .ok_or_else(|| {
            invalid_input(format!(
                "missing derived variance output `{MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME}`"
            ))
        })?;
    let magnetic_field_vector_z = stage
        .result
        .derived_variances
        .get(MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME)
        .ok_or_else(|| {
            invalid_input(format!(
                "missing derived variance output `{MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME}`"
            ))
        })?;
    let b_prior_variance = Cochain::new(2, magnetic_field_variance.prior_variance.clone());
    let b_posterior_variance = Cochain::new(2, magnetic_field_variance.posterior_variance.clone());
    let b_variance_ratio = Cochain::new(
        2,
        pointwise_variance_ratio(
            &magnetic_field_variance.prior_variance,
            &magnetic_field_variance.posterior_variance,
        ),
    );
    let b_absolute_error = Cochain::new(
        2,
        FeecVector::from_iterator(
            truth_b.coeffs.len(),
            truth_b
                .coeffs
                .iter()
                .zip(posterior_b.coeffs.iter())
                .map(|(truth_value, posterior_value)| (posterior_value - truth_value).abs()),
        ),
    );
    write_cochain_vtk(
        format!("{stage_dir}/B_cochain.vtk"),
        coords,
        topology,
        &posterior_b,
        "posterior_B",
    )?;
    write_cochain_vtk(
        format!("{stage_dir}/B_prior_variance.vtk"),
        coords,
        topology,
        &b_prior_variance,
        "prior_variance",
    )?;
    write_cochain_vtk(
        format!("{stage_dir}/B_posterior_variance.vtk"),
        coords,
        topology,
        &b_posterior_variance,
        "posterior_variance",
    )?;
    write_cochain_vtk(
        format!("{stage_dir}/B_variance_ratio.vtk"),
        coords,
        topology,
        &b_variance_ratio,
        "variance_ratio",
    )?;
    write_cochain_vtk(
        format!("{stage_dir}/B_absolute_error.vtk"),
        coords,
        topology,
        &b_absolute_error,
        "absolute_error",
    )?;
    write_2form_vector_field_vtk(
        format!("{stage_dir}/B_vector.vtk"),
        coords,
        topology,
        &posterior_b,
        "posterior_B_vector",
    )?;
    let truth_b_vectors = sample_2form_cell_vectors(coords, topology, &truth_b)?;
    let posterior_b_vectors = sample_2form_cell_vectors(coords, topology, &posterior_b)?;
    let posterior_b_magnitude = vector_magnitudes(&posterior_b_vectors);
    let prior_b_directional_variance = directional_variance_vectors(
        &magnetic_field_vector_x.prior_variance,
        &magnetic_field_vector_y.prior_variance,
        &magnetic_field_vector_z.prior_variance,
    );
    let posterior_b_directional_variance = directional_variance_vectors(
        &magnetic_field_vector_x.posterior_variance,
        &magnetic_field_vector_y.posterior_variance,
        &magnetic_field_vector_z.posterior_variance,
    );
    let b_vector_prior_trace = &magnetic_field_vector_x.prior_variance
        + &magnetic_field_vector_y.prior_variance
        + &magnetic_field_vector_z.prior_variance;
    let b_vector_posterior_trace = &magnetic_field_vector_x.posterior_variance
        + &magnetic_field_vector_y.posterior_variance
        + &magnetic_field_vector_z.posterior_variance;
    let b_vector_variance_ratio =
        pointwise_variance_ratio(&b_vector_prior_trace, &b_vector_posterior_trace);
    let b_vector_marginal_std = b_vector_posterior_trace.map(|value| value.max(0.0).sqrt());
    write_top_cell_vtk_fields(
        format!("{stage_dir}/B_vector_fields.vtk"),
        coords,
        topology,
        &[
            ("truth_B_vector", truth_b_vectors.as_slice()),
            ("posterior_B_vector", posterior_b_vectors.as_slice()),
            (
                "posterior_directional_variance",
                posterior_b_directional_variance.as_slice(),
            ),
            (
                "prior_directional_variance",
                prior_b_directional_variance.as_slice(),
            ),
        ],
        &[
            ("magnitude", posterior_b_magnitude.as_slice()),
            ("marginal_variance", b_vector_posterior_trace.as_slice()),
            ("marginal_std", b_vector_marginal_std.as_slice()),
            ("prior_marginal_variance", b_vector_prior_trace.as_slice()),
            (
                "marginal_variance_ratio",
                b_vector_variance_ratio.as_slice(),
            ),
        ],
    )?;
    let summary = format!(
        concat!(
            "relative_l2_error={:.8e}\n",
            "relative_b_l2_error={:.8e}\n",
            "posterior_mean_norm={:.8e}\n",
            "truth_norm={:.8e}\n",
            "pde_residual_norm={:.8e}\n",
            "sensor_rmse={:.8e}\n",
            "harmonic_residual_norm={:.8e}\n",
            "magnetic_field_prior_variance_mean={:.8e}\n",
            "magnetic_field_posterior_variance_mean={:.8e}\n",
            "magnetic_field_variance_ratio_mean={:.8e}\n",
            "magnetic_field_vector_prior_trace_mean={:.8e}\n",
            "magnetic_field_vector_posterior_trace_mean={:.8e}\n",
            "magnetic_field_vector_trace_ratio_mean={:.8e}\n",
            "input_representations={:?}\n"
        ),
        stage.relative_l2_error,
        stage.relative_b_l2_error,
        stage.posterior_mean_norm,
        stage.truth_norm,
        stage.pde_residual_norm,
        stage.sensor_rmse,
        stage.harmonic_residual_norm,
        mean(&magnetic_field_variance.prior_variance),
        mean(&magnetic_field_variance.posterior_variance),
        mean(&pointwise_variance_ratio(
            &magnetic_field_variance.prior_variance,
            &magnetic_field_variance.posterior_variance,
        )),
        mean(&b_vector_prior_trace),
        mean(&b_vector_posterior_trace),
        mean(&b_vector_variance_ratio),
        stage.result.debug.input_representations
    );
    fs::write(format!("{stage_dir}/summary.txt"), summary)?;
    Ok(())
}

fn directional_variance_vectors(x: &FeecVector, y: &FeecVector, z: &FeecVector) -> Vec<[f64; 3]> {
    assert_eq!(x.len(), y.len(), "x and y directional variances must align");
    assert_eq!(x.len(), z.len(), "x and z directional variances must align");
    (0..x.len())
        .map(|index| [x[index].max(0.0), y[index].max(0.0), z[index].max(0.0)])
        .collect()
}

fn vector_magnitudes(vectors: &[[f64; 3]]) -> Vec<f64> {
    vectors
        .iter()
        .map(|[x, y, z]| (x * x + y * y + z * z).sqrt())
        .collect()
}

fn write_summary(
    out_dir: &str,
    stage1: &StageOutcome,
    stage2: &StageOutcome,
    stage3: &StageOutcome,
    stage4: &StageOutcome,
    truth: &FeecVector,
    subsets: &UqSubsetIndices,
    flux_measurements: &[LinearGaussianMeasurementSpec],
    source_equivalence_mean_diff: f64,
) -> Result<(), Box<dyn Error>> {
    let sensor_truth = flux_measurements
        .iter()
        .map(|measurement| measurement.observations[0])
        .collect::<Vec<_>>();
    let mut sensor_lines = String::new();
    for (index, measurement) in flux_measurements.iter().enumerate() {
        sensor_lines.push_str(&format!(
            "{} truth={:.8e} stage1={:.8e} stage2={:.8e} stage3={:.8e} stage4={:.8e}\n",
            measurement.name,
            sensor_truth[index],
            stage1.sensor_predictions[index],
            stage2.sensor_predictions[index],
            stage3.sensor_predictions[index],
            stage4.sensor_predictions[index],
        ));
    }

    let stage_subset_lines =
        build_stage_subset_summary_lines(truth, stage1, stage2, stage3, stage4, subsets);
    let transition_lines = build_transition_summary_lines(stage1, stage2, stage3, stage4, subsets);

    let summary = format!(
        concat!(
            "coil_variance_mean_stage1={:.8e}\n",
            "coil_variance_mean_stage2={:.8e}\n",
            "outer_boundary_variance_mean_stage2={:.8e}\n",
            "outer_boundary_variance_mean_stage3={:.8e}\n",
            "core_boundary_variance_mean_stage2={:.8e}\n",
            "core_boundary_variance_mean_stage3={:.8e}\n",
            "sensor_edge_variance_mean_stage3={:.8e}\n",
            "sensor_edge_variance_mean_stage4={:.8e}\n",
            "stage1_rel_l2={:.8e}\n",
            "stage2_rel_l2={:.8e}\n",
            "stage3_rel_l2={:.8e}\n",
            "stage4_rel_l2={:.8e}\n",
            "stage1_rel_b_l2={:.8e}\n",
            "stage2_rel_b_l2={:.8e}\n",
            "stage3_rel_b_l2={:.8e}\n",
            "stage4_rel_b_l2={:.8e}\n",
            "stage1_mean_norm_ratio={:.8e}\n",
            "stage2_mean_norm_ratio={:.8e}\n",
            "stage3_mean_norm_ratio={:.8e}\n",
            "stage4_mean_norm_ratio={:.8e}\n",
            "stage1_pde_residual_norm={:.8e}\n",
            "stage2_pde_residual_norm={:.8e}\n",
            "stage3_pde_residual_norm={:.8e}\n",
            "stage4_pde_residual_norm={:.8e}\n",
            "stage1_sensor_rmse={:.8e}\n",
            "stage2_sensor_rmse={:.8e}\n",
            "stage3_sensor_rmse={:.8e}\n",
            "stage4_sensor_rmse={:.8e}\n",
            "stage3_harmonic_residual={:.8e}\n",
            "stage4_harmonic_residual={:.8e}\n",
            "source_equivalence_mean_diff={:.8e}\n",
            "\n[stage_subset_uq]\n{}",
            "\n[stage_transition_uq]\n{}",
            "\n[sensors]\n{}"
        ),
        mean_on_subset(&stage1.result.posterior_variance, &subsets.coil_edges),
        mean_on_subset(&stage2.result.posterior_variance, &subsets.coil_edges),
        mean_on_subset(
            &stage2.result.posterior_variance,
            &subsets.outer_boundary_edges
        ),
        mean_on_subset(
            &stage3.result.posterior_variance,
            &subsets.outer_boundary_edges
        ),
        mean_on_subset(
            &stage2.result.posterior_variance,
            &subsets.core_boundary_edges
        ),
        mean_on_subset(
            &stage3.result.posterior_variance,
            &subsets.core_boundary_edges
        ),
        mean_on_subset(&stage3.result.posterior_variance, &subsets.sensor_edges),
        mean_on_subset(&stage4.result.posterior_variance, &subsets.sensor_edges),
        stage1.relative_l2_error,
        stage2.relative_l2_error,
        stage3.relative_l2_error,
        stage4.relative_l2_error,
        stage1.relative_b_l2_error,
        stage2.relative_b_l2_error,
        stage3.relative_b_l2_error,
        stage4.relative_b_l2_error,
        stage1.posterior_mean_norm / stage1.truth_norm.max(1e-12),
        stage2.posterior_mean_norm / stage2.truth_norm.max(1e-12),
        stage3.posterior_mean_norm / stage3.truth_norm.max(1e-12),
        stage4.posterior_mean_norm / stage4.truth_norm.max(1e-12),
        stage1.pde_residual_norm,
        stage2.pde_residual_norm,
        stage3.pde_residual_norm,
        stage4.pde_residual_norm,
        stage1.sensor_rmse,
        stage2.sensor_rmse,
        stage3.sensor_rmse,
        stage4.sensor_rmse,
        stage3.harmonic_residual_norm,
        stage4.harmonic_residual_norm,
        source_equivalence_mean_diff,
        stage_subset_lines,
        transition_lines,
        sensor_lines
    );
    fs::write(Path::new(out_dir).join("summary.txt"), summary)?;
    Ok(())
}

fn edge_subset<P>(topology: &Complex, coords: &MeshCoords, predicate: P) -> Vec<usize>
where
    P: Fn(CoordRef<'_>) -> bool + Sync,
{
    let mut edges = (0..topology.nsimplices(1))
        .filter_map(|edge_index| {
            let edge = SimplexIdx::new(1, edge_index).handle(topology);
            let edge_coords = SimplexCoords::from_simplex_and_coords(&edge, coords);
            predicate(edge_coords.barycenter().as_view()).then_some(edge_index)
        })
        .collect::<Vec<_>>();
    edges.sort_unstable();
    edges
}

fn evaluate_measurements(
    measurements: &[LinearGaussianMeasurementSpec],
    state: &FeecVector,
) -> Result<Vec<f64>, String> {
    let mut values = Vec::with_capacity(measurements.len());
    for measurement in measurements {
        let predicted = sparse_measurement_values(&measurement.operator, &measurement.bias, state)?;
        values.extend(predicted.iter().copied());
    }
    Ok(values)
}

fn sparse_measurement_values(
    operator: &SparseTripletMatrix,
    bias: &[f64],
    state: &FeecVector,
) -> Result<FeecVector, String> {
    if operator.ncols() != state.len() {
        return Err(format!(
            "measurement operator columns {} must match state length {}",
            operator.ncols(),
            state.len()
        ));
    }
    if operator.nrows() != bias.len() {
        return Err(format!(
            "measurement bias length {} must match operator rows {}",
            bias.len(),
            operator.nrows()
        ));
    }
    let mut out = FeecVector::from_vec(bias.to_vec());
    for (row, col, value) in operator.triplet_iter() {
        out[row] += value * state[col];
    }
    Ok(out)
}

fn columns_to_sparse_matrix(columns: &[FeecVector]) -> SparseTripletMatrix {
    let nrows = columns.first().map(|column| column.len()).unwrap_or(0);
    SparseTripletMatrix::from_triplets(
        nrows,
        columns.len(),
        columns.iter().enumerate().flat_map(|(col, column)| {
            column.iter().enumerate().filter_map(move |(row, value)| {
                (value.abs() > 1e-12).then_some(SparseTriplet {
                    row,
                    col,
                    value: *value,
                })
            })
        }),
    )
}

fn dense_rows_to_measurements(
    matrix: &gmrf_core::types::DenseMatrix,
    state_dimension: usize,
    variance: f64,
    prefix: &str,
) -> Vec<LinearGaussianMeasurementSpec> {
    (0..matrix.nrows())
        .map(|row| LinearGaussianMeasurementSpec {
            name: format!("{prefix}_{row}"),
            operator: SparseTripletMatrix::from_triplets(
                1,
                state_dimension,
                (0..matrix.ncols()).filter_map(|col| {
                    let value = matrix[(row, col)];
                    (value.abs() > 1e-12).then_some(SparseTriplet { row: 0, col, value })
                }),
            ),
            observations: vec![0.0],
            bias: vec![0.0],
            variance,
        })
        .collect()
}

fn diagonal_precision(dimension: usize, diagonal_value: f64) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| SparseTriplet {
            row: index,
            col: index,
            value: diagonal_value,
        }),
    )
}

fn csr_to_triplet(matrix: &FeecCsr) -> SparseTripletMatrix {
    SparseTripletMatrix::from_triplets(
        matrix.nrows(),
        matrix.ncols(),
        matrix
            .triplet_iter()
            .map(|(row, col, value)| SparseTriplet {
                row,
                col,
                value: *value,
            }),
    )
}

fn add_diagonal_shift(matrix: &FeecCsr, shift: f64) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        coo.push(row, col, *value);
    }
    for index in 0..matrix.nrows() {
        coo.push(index, index, shift);
    }
    FeecCsr::from(&coo)
}

fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(lhs.nrows(), lhs.ncols());
    for (row, col, value) in lhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

fn scale_matrix(matrix: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let scaled = scale * *value;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn csr_rows(matrix: &FeecCsr) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); matrix.nrows()];
    for (row, col, value) in matrix.triplet_iter() {
        rows[row].push((col, *value));
    }
    rows
}

fn face_normal(face_coords: &SimplexCoords) -> [f64; 3] {
    let p0 = face_coords.coord(0);
    let p1 = face_coords.coord(1);
    let p2 = face_coords.coord(2);
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn scale_to_target_norm(vector: &FeecVector, target_norm: f64) -> FeecVector {
    let norm = vector.norm();
    if norm <= 1e-12 {
        vector.clone()
    } else {
        vector.scale(target_norm / norm)
    }
}

fn subtract_from_bias(bias: &mut [f64], rhs: &FeecVector) -> Result<(), String> {
    if bias.len() != rhs.len() {
        return Err(format!(
            "bias length {} must match rhs length {}",
            bias.len(),
            rhs.len()
        ));
    }
    for (entry, value) in bias.iter_mut().zip(rhs.iter()) {
        *entry -= *value;
    }
    Ok(())
}

fn build_uq_subset_indices(
    topology: &Complex,
    coords: &MeshCoords,
    outer_boundary_edges: &[usize],
    flux_measurements: &[LinearGaussianMeasurementSpec],
    geom: ToroidalInductorGeometry,
) -> UqSubsetIndices {
    let all_edges = (0..topology.nsimplices(1)).collect::<Vec<_>>();
    let coil_edges = edge_subset(topology, coords, |point| {
        let s = toroidal_radius(point, geom);
        s >= geom.core_minor_radius && s <= geom.coil_minor_radius
    });
    let core_boundary_edges = sorted_boundary_dofs(topology, coords, 1, |point| {
        (toroidal_radius(point, geom) - geom.core_minor_radius).abs()
            <= 3.0 * geom.target_air_cell_size
    });
    let sensor_edges = measurement_support(flux_measurements);
    let background_edges = complement_subset(
        topology.nsimplices(1),
        &[
            coil_edges.as_slice(),
            outer_boundary_edges,
            core_boundary_edges.as_slice(),
            sensor_edges.as_slice(),
        ],
    );
    UqSubsetIndices {
        all_edges,
        coil_edges,
        outer_boundary_edges: outer_boundary_edges.to_vec(),
        core_boundary_edges,
        sensor_edges,
        background_edges,
    }
}

fn uq_subset_entries(subsets: &UqSubsetIndices) -> [(&str, &[usize]); 6] {
    [
        ("all", subsets.all_edges.as_slice()),
        ("coil", subsets.coil_edges.as_slice()),
        ("outer_boundary", subsets.outer_boundary_edges.as_slice()),
        ("core_boundary", subsets.core_boundary_edges.as_slice()),
        ("sensor", subsets.sensor_edges.as_slice()),
        ("background", subsets.background_edges.as_slice()),
    ]
}

fn summarize_stage_uncertainty(
    truth: &FeecVector,
    stage: &StageOutcome,
    indices: &[usize],
) -> StageUncertaintySummary {
    if indices.is_empty() {
        return StageUncertaintySummary {
            count: 0,
            absolute_error_mean: f64::NAN,
            prior_variance_mean: f64::NAN,
            posterior_variance_mean: f64::NAN,
            variance_reduction_mean: f64::NAN,
            variance_ratio_mean: f64::NAN,
            posterior_std_mean: f64::NAN,
            normalized_abs_error_mean: f64::NAN,
            truth_within_1sigma_fraction: f64::NAN,
            truth_within_2sigma_fraction: f64::NAN,
        };
    }

    let mut absolute_error_mean = 0.0;
    let mut prior_variance_mean = 0.0;
    let mut posterior_variance_mean = 0.0;
    let mut variance_reduction_mean = 0.0;
    let mut variance_ratio_mean = 0.0;
    let mut posterior_std_mean = 0.0;
    let mut normalized_abs_error_mean = 0.0;
    let mut within_1sigma = 0usize;
    let mut within_2sigma = 0usize;

    for &index in indices {
        let abs_error = (stage.result.posterior_mean[index] - truth[index]).abs();
        let prior_variance = stage.result.prior_variance[index].max(0.0);
        let posterior_variance = stage.result.posterior_variance[index].max(0.0);
        let posterior_std = posterior_variance.sqrt();
        absolute_error_mean += abs_error;
        prior_variance_mean += prior_variance;
        posterior_variance_mean += posterior_variance;
        variance_reduction_mean += prior_variance - posterior_variance;
        variance_ratio_mean += safe_ratio(posterior_variance, prior_variance);
        posterior_std_mean += posterior_std;
        normalized_abs_error_mean += abs_error / posterior_std.max(METRIC_EPS);
        if abs_error <= posterior_std + METRIC_EPS {
            within_1sigma += 1;
        }
        if abs_error <= 2.0 * posterior_std + METRIC_EPS {
            within_2sigma += 1;
        }
    }

    let count = indices.len() as f64;
    StageUncertaintySummary {
        count: indices.len(),
        absolute_error_mean: absolute_error_mean / count,
        prior_variance_mean: prior_variance_mean / count,
        posterior_variance_mean: posterior_variance_mean / count,
        variance_reduction_mean: variance_reduction_mean / count,
        variance_ratio_mean: variance_ratio_mean / count,
        posterior_std_mean: posterior_std_mean / count,
        normalized_abs_error_mean: normalized_abs_error_mean / count,
        truth_within_1sigma_fraction: within_1sigma as f64 / count,
        truth_within_2sigma_fraction: within_2sigma as f64 / count,
    }
}

fn summarize_stage_transition(
    baseline: &StageOutcome,
    comparison: &StageOutcome,
    indices: &[usize],
) -> StageTransitionSummary {
    if indices.is_empty() {
        return StageTransitionSummary {
            count: 0,
            baseline_posterior_variance_mean: f64::NAN,
            comparison_posterior_variance_mean: f64::NAN,
            posterior_variance_delta_mean: f64::NAN,
            posterior_variance_mean_ratio: f64::NAN,
            posterior_variance_pointwise_ratio_mean: f64::NAN,
        };
    }

    let mut baseline_mean = 0.0;
    let mut comparison_mean = 0.0;
    let mut delta_mean = 0.0;
    let mut pointwise_ratio_mean = 0.0;
    for &index in indices {
        let baseline_value = baseline.result.posterior_variance[index].max(0.0);
        let comparison_value = comparison.result.posterior_variance[index].max(0.0);
        baseline_mean += baseline_value;
        comparison_mean += comparison_value;
        delta_mean += comparison_value - baseline_value;
        pointwise_ratio_mean += safe_ratio(comparison_value, baseline_value);
    }
    let count = indices.len() as f64;
    let baseline_posterior_variance_mean = baseline_mean / count;
    let comparison_posterior_variance_mean = comparison_mean / count;
    StageTransitionSummary {
        count: indices.len(),
        baseline_posterior_variance_mean,
        comparison_posterior_variance_mean,
        posterior_variance_delta_mean: delta_mean / count,
        posterior_variance_mean_ratio: safe_ratio(
            comparison_posterior_variance_mean,
            baseline_posterior_variance_mean,
        ),
        posterior_variance_pointwise_ratio_mean: pointwise_ratio_mean / count,
    }
}

fn build_stage_subset_summary_lines(
    truth: &FeecVector,
    stage1: &StageOutcome,
    stage2: &StageOutcome,
    stage3: &StageOutcome,
    stage4: &StageOutcome,
    subsets: &UqSubsetIndices,
) -> String {
    let mut lines = String::new();
    for stage in [stage1, stage2, stage3, stage4] {
        for (label, indices) in uq_subset_entries(subsets) {
            let summary = summarize_stage_uncertainty(truth, stage, indices);
            lines.push_str(&format!(
                "{}.{}.count={}\n{}.{}.absolute_error_mean={:.8e}\n{}.{}.prior_variance_mean={:.8e}\n{}.{}.posterior_variance_mean={:.8e}\n{}.{}.variance_reduction_mean={:.8e}\n{}.{}.variance_ratio_mean={:.8e}\n{}.{}.posterior_std_mean={:.8e}\n{}.{}.normalized_abs_error_mean={:.8e}\n{}.{}.truth_within_1sigma_fraction={:.8e}\n{}.{}.truth_within_2sigma_fraction={:.8e}\n",
                stage.name,
                label,
                summary.count,
                stage.name,
                label,
                summary.absolute_error_mean,
                stage.name,
                label,
                summary.prior_variance_mean,
                stage.name,
                label,
                summary.posterior_variance_mean,
                stage.name,
                label,
                summary.variance_reduction_mean,
                stage.name,
                label,
                summary.variance_ratio_mean,
                stage.name,
                label,
                summary.posterior_std_mean,
                stage.name,
                label,
                summary.normalized_abs_error_mean,
                stage.name,
                label,
                summary.truth_within_1sigma_fraction,
                stage.name,
                label,
                summary.truth_within_2sigma_fraction,
            ));
        }
    }
    lines
}

fn build_transition_summary_lines(
    stage1: &StageOutcome,
    stage2: &StageOutcome,
    stage3: &StageOutcome,
    stage4: &StageOutcome,
    subsets: &UqSubsetIndices,
) -> String {
    let mut lines = String::new();
    for (label, summary) in [
        (
            "stage2_over_stage1.all",
            summarize_stage_transition(stage1, stage2, &subsets.all_edges),
        ),
        (
            "stage2_over_stage1.coil",
            summarize_stage_transition(stage1, stage2, &subsets.coil_edges),
        ),
        (
            "stage3_over_stage2.all",
            summarize_stage_transition(stage2, stage3, &subsets.all_edges),
        ),
        (
            "stage3_over_stage2.outer_boundary",
            summarize_stage_transition(stage2, stage3, &subsets.outer_boundary_edges),
        ),
        (
            "stage3_over_stage2.core_boundary",
            summarize_stage_transition(stage2, stage3, &subsets.core_boundary_edges),
        ),
        (
            "stage4_over_stage3.all",
            summarize_stage_transition(stage3, stage4, &subsets.all_edges),
        ),
        (
            "stage4_over_stage3.sensor",
            summarize_stage_transition(stage3, stage4, &subsets.sensor_edges),
        ),
        (
            "stage4_over_stage3.background",
            summarize_stage_transition(stage3, stage4, &subsets.background_edges),
        ),
    ] {
        lines.push_str(&format!(
            "{}.count={}\n{}.baseline_posterior_variance_mean={:.8e}\n{}.comparison_posterior_variance_mean={:.8e}\n{}.posterior_variance_delta_mean={:.8e}\n{}.posterior_variance_mean_ratio={:.8e}\n{}.posterior_variance_pointwise_ratio_mean={:.8e}\n",
            label,
            summary.count,
            label,
            summary.baseline_posterior_variance_mean,
            label,
            summary.comparison_posterior_variance_mean,
            label,
            summary.posterior_variance_delta_mean,
            label,
            summary.posterior_variance_mean_ratio,
            label,
            summary.posterior_variance_pointwise_ratio_mean,
        ));
    }
    lines
}

fn complement_subset(total_count: usize, subsets: &[&[usize]]) -> Vec<usize> {
    let excluded = subsets
        .iter()
        .flat_map(|subset| subset.iter().copied())
        .collect::<BTreeSet<_>>();
    (0..total_count)
        .filter(|index| !excluded.contains(index))
        .collect()
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator.abs() <= METRIC_EPS {
        0.0
    } else {
        numerator / denominator
    }
}

fn debug_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn pointwise_variance_ratio(
    prior_variance: &FeecVector,
    posterior_variance: &FeecVector,
) -> FeecVector {
    assert_eq!(
        prior_variance.len(),
        posterior_variance.len(),
        "prior and posterior variance vectors must have matching lengths"
    );
    FeecVector::from_iterator(
        prior_variance.len(),
        prior_variance
            .iter()
            .zip(posterior_variance.iter())
            .map(|(prior, posterior)| safe_ratio((*posterior).max(0.0), (*prior).max(0.0))),
    )
}

fn mean_on_subset(values: &FeecVector, indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return f64::NAN;
    }
    indices.iter().map(|index| values[*index]).sum::<f64>() / indices.len() as f64
}

fn mean(values: &FeecVector) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn measurement_support(measurements: &[LinearGaussianMeasurementSpec]) -> Vec<usize> {
    let mut support = measurements
        .iter()
        .flat_map(|measurement| measurement.operator.triplet_iter().map(|(_, col, _)| col))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    support.sort_unstable();
    support
}

fn rmse(lhs: &[f64], rhs: &[f64]) -> Result<f64, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "rmse expected matching lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        ));
    }
    if lhs.is_empty() {
        return Ok(0.0);
    }
    let mse = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        / lhs.len() as f64;
    Ok(mse.sqrt())
}

fn max_abs_difference(lhs: &FeecVector, rhs: &FeecVector) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

trait VectorNormExt {
    fn norm(&self) -> f64;
}

impl VectorNormExt for Vec<f64> {
    fn norm(&self) -> f64 {
        self.iter().map(|value| value * value).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feg_infer::linear_pde::{LinearPdeFactorizationDebug, LinearPdeUqDebug};
    use manifold::gen::cartesian::CartesianMeshInfo;

    fn dummy_stage(
        posterior_mean: &[f64],
        prior_variance: &[f64],
        posterior_variance: &[f64],
    ) -> StageOutcome {
        StageOutcome {
            name: "dummy".to_string(),
            result: LinearPdeUqResult {
                posterior_mean: FeecVector::from_vec(posterior_mean.to_vec()),
                posterior_variance: FeecVector::from_vec(posterior_variance.to_vec()),
                prior_variance: FeecVector::from_vec(prior_variance.to_vec()),
                derived_variances: BTreeMap::new(),
                reduced_posterior_mean: FeecVector::from_vec(posterior_mean.to_vec()),
                reduced_posterior_variance: FeecVector::from_vec(posterior_variance.to_vec()),
                pde_residual_mean: FeecVector::zeros(posterior_mean.len()),
                debug: LinearPdeUqDebug {
                    input_representations: Vec::new(),
                    joint_dimension: posterior_mean.len(),
                    prior_factorization: LinearPdeFactorizationDebug {
                        dimension: posterior_mean.len(),
                        matrix_nnz: 0,
                        matrix_lower_triangle_nnz: 0,
                        factor_nnz: 0,
                        fill_in_ratio_vs_lower_triangle: 0.0,
                        factor_numeric_values_mib: 0.0,
                    },
                    posterior_factorization: LinearPdeFactorizationDebug {
                        dimension: posterior_mean.len(),
                        matrix_nnz: 0,
                        matrix_lower_triangle_nnz: 0,
                        factor_nnz: 0,
                        fill_in_ratio_vs_lower_triangle: 0.0,
                        factor_numeric_values_mib: 0.0,
                    },
                },
            },
            relative_l2_error: 0.0,
            relative_b_l2_error: 0.0,
            posterior_mean_norm: posterior_mean
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt(),
            truth_norm: posterior_mean
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt(),
            pde_residual_norm: 0.0,
            sensor_predictions: Vec::new(),
            sensor_rmse: 0.0,
            harmonic_residual_norm: 0.0,
        }
    }

    #[test]
    fn parse_f64_arg_accepts_positive_finite_values() {
        let parsed = parse_f64_arg("1e-6", "--pde-variance").expect("parse should succeed");
        assert!((parsed - 1e-6).abs() < 1e-18);
        assert!(parse_f64_arg("0.0", "--pde-variance").is_err());
        assert!(parse_f64_arg("-1.0", "--pde-variance").is_err());
        assert!(parse_f64_arg("NaN", "--pde-variance").is_err());
    }

    #[test]
    fn summarize_stage_uncertainty_computes_ratios_and_coverage() {
        let truth = FeecVector::from_vec(vec![0.0, 0.0]);
        let stage = dummy_stage(&[1.0, 2.0], &[4.0, 9.0], &[1.0, 4.0]);
        let summary = summarize_stage_uncertainty(&truth, &stage, &[0, 1]);

        assert_eq!(summary.count, 2);
        assert!((summary.absolute_error_mean - 1.5).abs() < 1e-12);
        assert!((summary.prior_variance_mean - 6.5).abs() < 1e-12);
        assert!((summary.posterior_variance_mean - 2.5).abs() < 1e-12);
        assert!((summary.variance_reduction_mean - 4.0).abs() < 1e-12);
        assert!((summary.variance_ratio_mean - ((0.25 + (4.0 / 9.0)) / 2.0)).abs() < 1e-12);
        assert!((summary.posterior_std_mean - 1.5).abs() < 1e-12);
        assert!((summary.normalized_abs_error_mean - 1.0).abs() < 1e-12);
        assert!((summary.truth_within_1sigma_fraction - 1.0).abs() < 1e-12);
        assert!((summary.truth_within_2sigma_fraction - 1.0).abs() < 1e-12);
    }

    #[test]
    fn summarize_stage_transition_tracks_variance_inflation_and_shrinkage() {
        let baseline = dummy_stage(&[0.0, 0.0], &[1.0, 1.0], &[2.0, 8.0]);
        let comparison = dummy_stage(&[0.0, 0.0], &[1.0, 1.0], &[4.0, 4.0]);
        let summary = summarize_stage_transition(&baseline, &comparison, &[0, 1]);

        assert_eq!(summary.count, 2);
        assert!((summary.baseline_posterior_variance_mean - 5.0).abs() < 1e-12);
        assert!((summary.comparison_posterior_variance_mean - 4.0).abs() < 1e-12);
        assert!((summary.posterior_variance_delta_mean + 1.0).abs() < 1e-12);
        assert!((summary.posterior_variance_mean_ratio - 0.8).abs() < 1e-12);
        assert!((summary.posterior_variance_pointwise_ratio_mean - 1.25).abs() < 1e-12);
    }

    #[test]
    fn pointwise_variance_ratio_handles_zero_and_negative_entries() {
        let prior = FeecVector::from_vec(vec![4.0, 0.0, -3.0, 9.0]);
        let posterior = FeecVector::from_vec(vec![1.0, 2.0, -1.0, 0.0]);
        let ratio = pointwise_variance_ratio(&prior, &posterior);

        assert_eq!(ratio.len(), 4);
        assert!((ratio[0] - 0.25).abs() < 1e-12);
        assert_eq!(ratio[1], 0.0);
        assert_eq!(ratio[2], 0.0);
        assert_eq!(ratio[3], 0.0);
    }

    #[test]
    fn debug_ratio_allows_tiny_positive_denominators() {
        assert!((debug_ratio(2.0e-16, 1.0e-16) - 2.0).abs() < 1e-12);
        assert_eq!(debug_ratio(1.0, 0.0), 0.0);
    }

    #[test]
    fn write_stage_outputs_writes_magnetic_field_variance_files() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.nsimplices(1);
        let face_count = topology.nsimplices(2);
        let cell_count = topology.cells().len();

        let mut stage = dummy_stage(
            &vec![0.0; edge_count],
            &vec![1.0; edge_count],
            &vec![0.5; edge_count],
        );
        stage.name = "stage".to_string();
        stage.result.derived_variances = BTreeMap::from([
            (
                MAGNETIC_FIELD_DERIVED_NAME.to_string(),
                feg_infer::linear_pde::LinearPdeDerivedMarginalResult {
                    prior_variance: FeecVector::from_element(face_count, 1.0),
                    posterior_variance: FeecVector::from_element(face_count, 0.25),
                },
            ),
            (
                MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME.to_string(),
                feg_infer::linear_pde::LinearPdeDerivedMarginalResult {
                    prior_variance: FeecVector::from_element(cell_count, 1.0),
                    posterior_variance: FeecVector::from_element(cell_count, 0.25),
                },
            ),
            (
                MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME.to_string(),
                feg_infer::linear_pde::LinearPdeDerivedMarginalResult {
                    prior_variance: FeecVector::from_element(cell_count, 1.0),
                    posterior_variance: FeecVector::from_element(cell_count, 0.25),
                },
            ),
            (
                MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME.to_string(),
                feg_infer::linear_pde::LinearPdeDerivedMarginalResult {
                    prior_variance: FeecVector::from_element(cell_count, 1.0),
                    posterior_variance: FeecVector::from_element(cell_count, 0.25),
                },
            ),
        ]);

        let out_dir = std::env::temp_dir().join(format!(
            "magnetostatic_uq_stage_outputs_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&out_dir).expect("temp output dir should be created");
        write_stage_outputs(
            &topology,
            &coords,
            &FeecVector::zeros(edge_count),
            out_dir
                .to_str()
                .expect("temp dir path should be valid utf-8"),
            &stage,
        )
        .expect("stage outputs should be written");

        let stage_dir = out_dir.join("stage");
        assert!(stage_dir.join("A_fields.vtk").exists());
        assert!(stage_dir.join("B_prior_variance.vtk").exists());
        assert!(stage_dir.join("B_posterior_variance.vtk").exists());
        assert!(stage_dir.join("B_variance_ratio.vtk").exists());
        assert!(stage_dir.join("B_vector_fields.vtk").exists());

        std::fs::remove_dir_all(out_dir).ok();
    }

    #[test]
    fn magnetic_field_derived_quantities_have_expected_dimensions() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();

        let derived = build_magnetic_field_derived_quantities(&topology, &coords)
            .expect("3D magnetic field operators should assemble");
        let names = derived
            .iter()
            .map(|operator| operator.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            vec![
                MAGNETIC_FIELD_DERIVED_NAME,
                MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME,
                MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME,
                MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME,
            ]
        );

        assert_eq!(derived[0].operator.ncols, topology.nsimplices(1));
        assert_eq!(derived[0].operator.nrows(), topology.nsimplices(2));
        for operator in derived.iter().skip(1) {
            assert_eq!(operator.operator.ncols, topology.nsimplices(1));
            assert_eq!(operator.operator.nrows(), topology.cells().len());
        }
    }

    #[test]
    fn magnetic_field_operator_matches_cochain_derivative() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let derived = build_magnetic_field_derived_quantities(&topology, &coords)
            .expect("3D magnetic field operators should assemble");
        let state = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|index| index as f64 + 1.0),
        );
        let magnetic_field = derived
            .iter()
            .find(|operator| operator.name == MAGNETIC_FIELD_DERIVED_NAME)
            .expect("magnetic field operator should be present");

        let applied = magnetic_field
            .operator
            .apply(&GmrfVector::from_vec(state.iter().copied().collect()))
            .expect("magnetic field operator should apply");
        let direct = Cochain::new(1, state).dif(&topology);

        assert_eq!(applied.len(), direct.coeffs.len());
        for index in 0..applied.len() {
            assert!(
                (applied[index] - direct.coeffs[index]).abs() < 1e-12,
                "face {index} mismatch: operator={} direct={}",
                applied[index],
                direct.coeffs[index]
            );
        }
    }

    #[test]
    fn magnetic_field_vector_operators_match_barycentric_sampling() {
        let mesh = CartesianMeshInfo::new_unit_scaled(3, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let derived = build_magnetic_field_derived_quantities(&topology, &coords)
            .expect("3D magnetic field operators should assemble");
        let state = FeecVector::from_iterator(
            topology.nsimplices(1),
            (0..topology.nsimplices(1)).map(|index| (index as f64 + 1.0) / 3.0),
        );
        let gmrf_state = GmrfVector::from_vec(state.iter().copied().collect());
        let direct_b = Cochain::new(1, state).dif(&topology);
        let sampled_vectors =
            sample_2form_cell_vectors(&coords, &topology, &direct_b).expect("sampling should work");

        for (name, component) in [
            (MAGNETIC_FIELD_VECTOR_X_DERIVED_NAME, 0usize),
            (MAGNETIC_FIELD_VECTOR_Y_DERIVED_NAME, 1usize),
            (MAGNETIC_FIELD_VECTOR_Z_DERIVED_NAME, 2usize),
        ] {
            let operator = derived
                .iter()
                .find(|spec| spec.name == name)
                .expect("component operator should be present");
            let applied = operator
                .operator
                .apply(&gmrf_state)
                .expect("component operator should apply");
            assert_eq!(applied.len(), sampled_vectors.len());
            for cell_index in 0..applied.len() {
                assert!(
                    (applied[cell_index] - sampled_vectors[cell_index][component]).abs() < 1e-12,
                    "cell {cell_index} component {component} mismatch: operator={} direct={}",
                    applied[cell_index],
                    sampled_vectors[cell_index][component]
                );
            }
        }
    }
}
