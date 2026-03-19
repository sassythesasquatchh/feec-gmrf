use ddf::cochain::Cochain;
use manifold::geometry::coord::mesh::MeshCoords;
use manifold::topology::complex::Complex;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Write multiple 0-cochain scalar fields into a single VTK file (POINT_DATA on vertices).
pub fn write_0cochain_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    if fields.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "at least one cochain field is required",
        ));
    }

    if coords.dim() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VTK export supports up to 3D coordinates",
        ));
    }

    let topo_dim = topology.dim();
    let skeleton = topology.skeleton(topo_dim);
    let ncells = skeleton.len();
    let nverts_per_cell = topo_dim + 1;

    for (name, cochain) in fields {
        if cochain.dim() != 0 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Expected a 0-cochain for field {name}"),
            ));
        }
        if cochain.len() != coords.nvertices() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Cochain length {} does not match vertex count {} for field {name}",
                    cochain.len(),
                    coords.nvertices()
                ),
            ));
        }
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "0-cochain fields")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} double", coords.nvertices())?;
    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
        let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
        writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
    }

    writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
    for cell in skeleton.handle_iter() {
        write!(w, "{nverts_per_cell}")?;
        for vertex in cell.vertices.iter() {
            write!(w, " {}", vertex)?;
        }
        writeln!(w)?;
    }

    let cell_type = match topo_dim {
        0 => 1,
        1 => 3,
        2 => 5,
        3 => 10,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unsupported topology dimension {topo_dim}"),
            ))
        }
    };

    writeln!(w, "CELL_TYPES {}", ncells)?;
    for _ in 0..ncells {
        writeln!(w, "{cell_type}")?;
    }

    writeln!(w, "POINT_DATA {}", coords.nvertices())?;
    for (name, cochain) in fields {
        writeln!(w, "SCALARS {} double 1", name)?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for coeff in cochain.coeffs.iter() {
            writeln!(w, "{coeff:.12}")?;
        }
    }

    Ok(())
}

/// Write multiple 1-cochain scalar fields into a single VTK file (CELL_DATA on edges).
pub fn write_1cochain_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    if fields.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "at least one cochain field is required",
        ));
    }

    if coords.dim() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VTK export supports up to 3D coordinates",
        ));
    }

    let edge_skeleton = topology.skeleton(1);
    let ncells = edge_skeleton.len();
    for (name, cochain) in fields {
        if cochain.dim() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Expected a 1-cochain for field {name}"),
            ));
        }
        if cochain.len() != ncells {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Cochain length {} does not match edge skeleton size {} for field {name}",
                    cochain.len(),
                    ncells
                ),
            ));
        }
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "1-cochain fields")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} double", coords.nvertices())?;
    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
        let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
        writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
    }

    let nverts_per_cell = 2;
    writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
    for edge in edge_skeleton.handle_iter() {
        writeln!(
            w,
            "{nverts_per_cell} {} {}",
            edge.vertices[0], edge.vertices[1]
        )?;
    }

    writeln!(w, "CELL_TYPES {}", ncells)?;
    for _ in 0..ncells {
        writeln!(w, "3")?;
    }

    writeln!(w, "CELL_DATA {}", ncells)?;
    for (name, cochain) in fields {
        writeln!(w, "SCALARS {} double 1", name)?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for coeff in cochain.coeffs.iter() {
            writeln!(w, "{coeff:.12}")?;
        }
    }

    Ok(())
}

/// Write a 1-form vector proxy plus additional 1-cochain scalar fields into a single VTK file.
pub fn write_1form_vector_proxy_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vector_cochain: &Cochain,
    scalar_fields: &[(&str, &Cochain)],
) -> io::Result<()> {
    if vector_cochain.dim() != 1 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Expected a 1-cochain for vectors ({vector_name})"),
        ));
    }

    if coords.dim() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VTK export supports up to 3D coordinates",
        ));
    }

    let edge_skeleton = topology.skeleton(1);
    let ncells = edge_skeleton.len();
    if vector_cochain.len() != ncells {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "Cochain length {} does not match edge skeleton size {} for vectors ({vector_name})",
                vector_cochain.len(),
                ncells
            ),
        ));
    }

    for (name, cochain) in scalar_fields {
        if cochain.dim() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Expected a 1-cochain for field {name}"),
            ));
        }
        if cochain.len() != ncells {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Cochain length {} does not match edge skeleton size {} for field {name}",
                    cochain.len(),
                    ncells
                ),
            ));
        }
    }

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "1-form vector proxy fields")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} double", coords.nvertices())?;
    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
        let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
        writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
    }

    let nverts_per_cell = 2;
    writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
    for edge in edge_skeleton.handle_iter() {
        writeln!(
            w,
            "{nverts_per_cell} {} {}",
            edge.vertices[0], edge.vertices[1]
        )?;
    }

    writeln!(w, "CELL_TYPES {}", ncells)?;
    for _ in 0..ncells {
        writeln!(w, "3")?;
    }

    writeln!(w, "CELL_DATA {}", ncells)?;
    writeln!(w, "VECTORS {} double", vector_name)?;
    for edge in edge_skeleton.handle_iter() {
        let v0 = coords.coord(edge.vertices[0]);
        let v1 = coords.coord(edge.vertices[1]);
        let mut dir = (v1 - v0).into_owned();
        let length = dir.norm();
        if length > 0.0 {
            let scale = vector_cochain[edge] / length;
            dir *= scale;
        } else {
            dir.fill(0.0);
        }

        let vx = dir[0];
        let vy = if dir.len() > 1 { dir[1] } else { 0.0 };
        let vz = if dir.len() > 2 { dir[2] } else { 0.0 };
        writeln!(w, "{vx:.12} {vy:.12} {vz:.12}")?;
    }

    for (name, cochain) in scalar_fields {
        writeln!(w, "SCALARS {} double 1", name)?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for coeff in cochain.coeffs.iter() {
            writeln!(w, "{coeff:.12}")?;
        }
    }

    Ok(())
}

/// Write multiple scalar fields defined on top-dimensional cells into a single VTK file.
pub fn write_top_cell_scalar_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    fields: &[(&str, &[f64])],
) -> io::Result<()> {
    if fields.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "at least one cell scalar field is required",
        ));
    }

    if coords.dim() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VTK export supports up to 3D coordinates",
        ));
    }

    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let ncells = cell_skeleton.len();
    let nverts_per_cell = topo_dim + 1;
    for (name, field) in fields {
        if field.len() != ncells {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "field length {} does not match top-cell count {} for field {name}",
                    field.len(),
                    ncells,
                ),
            ));
        }
    }

    let cell_type = match topo_dim {
        0 => 1,
        1 => 3,
        2 => 5,
        3 => 10,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unsupported topology dimension {topo_dim}"),
            ))
        }
    };

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "top-cell scalar fields")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} double", coords.nvertices())?;
    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
        let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
        writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
    }

    writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
    for cell in cell_skeleton.handle_iter() {
        write!(w, "{nverts_per_cell}")?;
        for vertex in cell.vertices.iter() {
            write!(w, " {}", vertex)?;
        }
        writeln!(w)?;
    }

    writeln!(w, "CELL_TYPES {}", ncells)?;
    for _ in 0..ncells {
        writeln!(w, "{cell_type}")?;
    }

    writeln!(w, "CELL_DATA {}", ncells)?;
    for (name, field) in fields {
        writeln!(w, "SCALARS {} double 1", name)?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for value in field.iter().copied() {
            writeln!(w, "{value:.12}")?;
        }
    }

    Ok(())
}

/// Write a vector field plus scalar fields defined on top-dimensional cells into a single VTK file.
pub fn write_top_cell_vector_vtk_fields(
    path: impl AsRef<Path>,
    coords: &MeshCoords,
    topology: &Complex,
    vector_name: &str,
    vectors: &[[f64; 3]],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    if coords.dim() > 3 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "VTK export supports up to 3D coordinates",
        ));
    }

    let topo_dim = topology.dim();
    let cell_skeleton = topology.skeleton(topo_dim);
    let ncells = cell_skeleton.len();
    let nverts_per_cell = topo_dim + 1;
    if vectors.len() != ncells {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!(
                "vector count {} does not match top-cell count {} for vector field {vector_name}",
                vectors.len(),
                ncells,
            ),
        ));
    }
    for (name, field) in scalar_fields {
        if field.len() != ncells {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "field length {} does not match top-cell count {} for field {name}",
                    field.len(),
                    ncells,
                ),
            ));
        }
    }

    let cell_type = match topo_dim {
        0 => 1,
        1 => 3,
        2 => 5,
        3 => 10,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("unsupported topology dimension {topo_dim}"),
            ))
        }
    };

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "top-cell vector and scalar fields")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;

    writeln!(w, "POINTS {} double", coords.nvertices())?;
    for coord in coords.coord_iter() {
        let x = coord[0];
        let y = if coords.dim() > 1 { coord[1] } else { 0.0 };
        let z = if coords.dim() > 2 { coord[2] } else { 0.0 };
        writeln!(w, "{x:.6} {y:.6} {z:.6}")?;
    }

    writeln!(w, "CELLS {} {}", ncells, ncells * (nverts_per_cell + 1))?;
    for cell in cell_skeleton.handle_iter() {
        write!(w, "{nverts_per_cell}")?;
        for vertex in cell.vertices.iter() {
            write!(w, " {}", vertex)?;
        }
        writeln!(w)?;
    }

    writeln!(w, "CELL_TYPES {}", ncells)?;
    for _ in 0..ncells {
        writeln!(w, "{cell_type}")?;
    }

    writeln!(w, "CELL_DATA {}", ncells)?;
    writeln!(w, "VECTORS {} double", vector_name)?;
    for [vx, vy, vz] in vectors.iter().copied() {
        writeln!(w, "{vx:.12} {vy:.12} {vz:.12}")?;
    }
    for (name, field) in scalar_fields {
        writeln!(w, "SCALARS {} double 1", name)?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for value in field.iter().copied() {
            writeln!(w, "{value:.12}")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use common::linalg::nalgebra::Vector as FeecVector;
    use manifold::gen::cartesian::CartesianMeshInfo;
    use std::fs;

    #[test]
    fn write_0cochain_vtk_fields_writes_multiple_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let vertex_count = coords.nvertices();

        let a = Cochain::new(0, FeecVector::from_element(vertex_count, 1.0));
        let b = Cochain::new(0, FeecVector::from_element(vertex_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_0cochain_fields_{stamp}.vtk"));

        write_0cochain_vtk_fields(&path, &coords, &topology, &[("a", &a), ("b", &b)])
            .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("POINT_DATA"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1cochain_vtk_fields_writes_multiple_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();

        let a = Cochain::new(1, FeecVector::from_element(edge_count, 1.0));
        let b = Cochain::new(1, FeecVector::from_element(edge_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1cochain_fields_{stamp}.vtk"));

        write_1cochain_vtk_fields(&path, &coords, &topology, &[("a", &a), ("b", &b)])
            .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("CELL_DATA"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_1form_vector_proxy_vtk_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let edge_count = topology.skeleton(1).len();

        let vec = Cochain::new(1, FeecVector::from_element(edge_count, 1.0));
        let var = Cochain::new(1, FeecVector::from_element(edge_count, 2.0));

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_1form_proxy_fields_{stamp}.vtk"));

        write_1form_vector_proxy_vtk_fields(
            &path,
            &coords,
            &topology,
            "proxy",
            &vec,
            &[("var", &var)],
        )
        .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("VECTORS proxy double"));
        assert!(content.contains("SCALARS var double 1"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_top_cell_vector_vtk_fields_writes_vectors_and_scalars() {
        let mesh = CartesianMeshInfo::new_unit_scaled(2, 1, 1.0);
        let (topology, coords) = mesh.compute_coord_complex();
        let cell_count = topology.cells().len();
        let vectors = vec![[1.0, 0.0, 0.5]; cell_count];
        let a = vec![0.25; cell_count];
        let b = vec![0.75; cell_count];

        let mut path = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should be monotone")
            .as_nanos();
        path.push(format!("feg_infer_top_cell_vector_fields_{stamp}.vtk"));

        write_top_cell_vector_vtk_fields(
            &path,
            &coords,
            &topology,
            "cell_vectors",
            &vectors,
            &[("a", &a), ("b", &b)],
        )
        .expect("vtk write should succeed");

        let content = fs::read_to_string(&path).expect("vtk should be readable");
        assert!(content.contains("VECTORS cell_vectors double"));
        assert!(content.contains("SCALARS a double 1"));
        assert!(content.contains("SCALARS b double 1"));

        let _ = fs::remove_file(path);
    }
}
