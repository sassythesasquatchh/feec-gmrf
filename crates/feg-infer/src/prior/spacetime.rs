use crate::sparse::{core_triplet_to_feec_csr, feec_csr_to_gmrf};
use common::linalg::nalgebra::CsrMatrix as FeecCsr;
use feg_core::{BoundarySpec, SpatialPriorSlice};
use formoniq::problems::spacetime_prior::{
    build_spatial_prior_slice_0form, build_spatial_prior_slice_1form,
    build_spatial_prior_slice_2form, Hodge1PriorConfig, Hodge2PriorConfig, ScalarPriorConfig,
};
use gmrf_core::{BlockTridiagonalPrecision, GmrfError};
use manifold::{geometry::metric::mesh::MeshLengths, topology::complex::Complex};

#[derive(Debug, Clone)]
pub struct SpacetimePriorConfig {
    pub times: Vec<f64>,
}

impl SpacetimePriorConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.times.is_empty() {
            return Err("time grid must contain at least one state".to_string());
        }
        for pair in self.times.windows(2) {
            let [t0, t1] = pair else { unreachable!() };
            if t1 <= t0 {
                return Err(format!(
                    "time grid must be strictly increasing, found {} followed by {}",
                    t0, t1
                ));
            }
        }
        Ok(())
    }

    pub fn slice_count(&self) -> usize {
        self.times.len()
    }
}

#[derive(Debug, Clone)]
pub struct SpacetimePrior {
    pub slice: SpatialPriorSlice,
    pub precision: BlockTridiagonalPrecision,
}

pub fn build_spacetime_prior_from_slice(
    slice: SpatialPriorSlice,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    config.validate()?;

    let mass = core_triplet_to_feec_csr(&slice.mass);
    let drift = core_triplet_to_feec_csr(&slice.drift);
    let q_init = core_triplet_to_feec_csr(&slice.initial_precision);
    let q_w = core_triplet_to_feec_csr(&slice.driving_noise_precision);

    let mut diagonal_blocks = vec![
        FeecCsr::from(&common::linalg::nalgebra::CooMatrix::new(
            slice.state_dimension(),
            slice.state_dimension(),
        ));
        config.slice_count()
    ];
    let mut lower_blocks = Vec::with_capacity(config.slice_count().saturating_sub(1));

    diagonal_blocks[0] = add_sparse(&diagonal_blocks[0], &q_init);
    let mass_t = mass.transpose();
    for dt_window in config.times.windows(2) {
        let [t0, t1] = dt_window else { unreachable!() };
        let dt = t1 - t0;
        let inv_dt = 1.0 / dt;
        let g = add_sparse(&mass, &scale_matrix(&drift, dt));
        let mt_qw_m = scaled_triple_product(&mass_t, &q_w, &mass, inv_dt);
        let gt_qw_g = scaled_triple_product(&g.transpose(), &q_w, &g, inv_dt);
        let gt_qw_m = scaled_triple_product(&g.transpose(), &q_w, &mass, -inv_dt);

        let step = lower_blocks.len();
        diagonal_blocks[step] = add_sparse(&diagonal_blocks[step], &mt_qw_m);
        diagonal_blocks[step + 1] = add_sparse(&diagonal_blocks[step + 1], &gt_qw_g);
        lower_blocks.push(gt_qw_m);
    }

    let precision = BlockTridiagonalPrecision::new(
        diagonal_blocks.iter().map(feec_csr_to_gmrf).collect(),
        lower_blocks.iter().map(feec_csr_to_gmrf).collect(),
    )
    .map_err(gmrf_error_to_string)?;

    Ok(SpacetimePrior { slice, precision })
}

pub fn build_0form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: ScalarPriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    let slice = build_spatial_prior_slice_0form(topology, geometry, boundary, spatial)?;
    build_spacetime_prior_from_slice(slice, config)
}

pub fn build_1form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge1PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    let slice = build_spatial_prior_slice_1form(topology, geometry, boundary, spatial)?;
    build_spacetime_prior_from_slice(slice, config)
}

pub fn build_2form_spacetime_prior(
    topology: &Complex,
    geometry: &MeshLengths,
    boundary: &BoundarySpec,
    spatial: Hodge2PriorConfig,
    config: &SpacetimePriorConfig,
) -> Result<SpacetimePrior, String> {
    let slice = build_spatial_prior_slice_2form(topology, geometry, boundary, spatial)?;
    build_spacetime_prior_from_slice(slice, config)
}

fn scaled_triple_product(left: &FeecCsr, middle: &FeecCsr, right: &FeecCsr, scale: f64) -> FeecCsr {
    scale_matrix(&(left * middle * right), scale)
}

fn add_sparse(lhs: &FeecCsr, rhs: &FeecCsr) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::from(lhs);
    for (row, col, value) in rhs.triplet_iter() {
        coo.push(row, col, *value);
    }
    FeecCsr::from(&coo)
}

fn scale_matrix(matrix: &FeecCsr, scale: f64) -> FeecCsr {
    let mut coo = common::linalg::nalgebra::CooMatrix::new(matrix.nrows(), matrix.ncols());
    for (row, col, value) in matrix.triplet_iter() {
        let scaled = *value * scale;
        if scaled != 0.0 {
            coo.push(row, col, scaled);
        }
    }
    FeecCsr::from(&coo)
}

fn gmrf_error_to_string(err: GmrfError) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use feg_core::{BoundaryRegionSpec, BoundaryTreatment};
    use manifold::gen::cartesian::CartesianMeshInfo;

    #[test]
    fn nonuniform_time_grid_builds_factorizable_block_precision() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 2, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let geometry = coords.to_edge_lengths(&topology);
        let boundary = BoundarySpec::default().with_state_region(BoundaryRegionSpec::new(
            "hard",
            topology
                .boundary_subcomplex_simplices(0)
                .into_iter()
                .take(1)
                .map(|simp| simp.kidx)
                .collect(),
            vec![0.0],
            BoundaryTreatment::HardEssential,
        ));
        let prior = build_0form_spacetime_prior(
            &topology,
            &geometry,
            &boundary,
            ScalarPriorConfig {
                kappa: 1.0,
                tau: 1.0,
            },
            &SpacetimePriorConfig {
                times: vec![0.0, 0.1, 0.3],
            },
        )
        .expect("spacetime prior should assemble");

        assert_eq!(prior.precision.block_count(), 3);
        assert_eq!(
            prior.precision.dimension(),
            prior.slice.state_dimension() * 3
        );
        prior
            .precision
            .to_sparse()
            .cholesky_sqrt_lower()
            .expect("spacetime precision should factorize");
    }
}
