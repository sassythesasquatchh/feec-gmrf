mod sparse;
mod torus_1form_kappa0_support;

pub use sparse::{core_triplet_to_feec_csr, lift_vector_with_layout, reduce_vector_with_layout};

pub mod diagnostics;
pub mod hodge_1form_conditioning;
pub mod linear_conditioning;
pub mod matern_0form;
pub mod matern_1form;
pub mod matern_2form;
pub mod model;
pub mod prior;
pub mod torus_0form_conditioning;
pub mod torus_1form_conditioning;
pub mod torus_1form_hodge_conditioning;
pub mod torus_1form_mass_inverse_isolation;
pub mod torus_1form_pde_conditioning;
pub mod torus_1form_pde_hodge_conditioning;
pub mod util;
pub mod vtk;
