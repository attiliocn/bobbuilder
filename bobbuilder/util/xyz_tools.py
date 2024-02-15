import numpy as np
from scipy.spatial.distance import pdist

from util.kabsch import kabsch_algorithm
from util.graph_tools import find_neighbors
from util.geometry import axisangle_to_q, qv_mult

def build_xyz_file(elements, coordinates):
    elements = elements.reshape(-1,1)
    coordinates = coordinates.reshape(-1,3)
    xyz_content = []
    xyz_content.append(f"{str(len(coordinates))}\n")
    xyz_content.append('\n')
    for i in np.concatenate((elements, coordinates), axis=1):
        xyz_content.append("{: >3} {: >10} {: >10} {: >10}\n".format(*i))
    xyz_content = ''.join(xyz_content)
    return xyz_content
    
def obabel(input_string, input_format, output_format):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(input_format, output_format)
    obabel_mol = openbabel.OBMol()
    obConversion.ReadString(obabel_mol,input_string)
    return obConversion.WriteString(obabel_mol)

def reorder_xyz(xyz_matrix, src_numbering, dest_numbering):
            n_atoms = len(xyz_matrix)
            modified_matrix = np.zeros(n_atoms*4).reshape(n_atoms,4)
            modified_matrix = modified_matrix.astype(xyz_matrix.dtype)

            unmodified_in_source = []
            for i in range(n_atoms):
                if i not in src_numbering:
                    unmodified_in_source.append(i)
            unmodified_in_source = np.array(unmodified_in_source)
            
            empty_in_destination = []
            for i in range(n_atoms):
                if i not in dest_numbering:
                    empty_in_destination.append(i)
            empty_in_destination = np.array(empty_in_destination)

            for src, dest in zip(unmodified_in_source, empty_in_destination):
                modified_matrix[dest] = xyz_matrix[src]

            for src, dest in zip(src_numbering, dest_numbering):
                modified_matrix[dest] = xyz_matrix[src]
            
            return modified_matrix

def rotate_dihedral(atom1, atom2, coordinates, adj_matrix, rad):
    z_axis = np.array([0,0,1])

    # translate atom 1 (fixed) to origin
    translation = coordinates[atom1].copy()
    coordinates -= translation
    # align the bond to z-axis
    bond_axis = coordinates[atom1] - coordinates[atom2]
    R, t = kabsch_algorithm(
        bond_axis.reshape(3,-1),
        z_axis.reshape(3,-1),
        center=False
    )
    coordinates = (R @ coordinates.T).T
    # rotate atoms bonded to atom2
    atoms_to_rotate = find_neighbors(adj_matrix, atom_number=atom2, excluded_atoms=[atom1])
    atoms_fixed = list(set([i for i in range(len(coordinates))]).difference(set(atoms_to_rotate)))
    
    coords_rotated = coordinates.copy()
    for i in atoms_fixed:
        coords_rotated[i] = np.array([0.,0.,0.])

    quat = axisangle_to_q(z_axis, rad)
    for i, atom_coordinates in enumerate(coords_rotated):
        new_coordinates = qv_mult(quat, tuple(atom_coordinates))
        coords_rotated[i] = new_coordinates

    for i in atoms_to_rotate:
        coordinates[i] = coords_rotated[i]

    # undo rotation and translation
    coordinates = (R.T @ coordinates.T).T
    coordinates += translation

    return coordinates

def get_conformers(coordinates, adjacency_matrix, rotatable_bonds, numconfs=500, threshold=.950):
    conformers = []
    for _ in range(numconfs):
            rotation_angles = np.random.uniform(np.radians(-180), np.radians(180), len(rotatable_bonds))
            rot = coordinates.copy()
            for bond, angle in zip(rotatable_bonds, rotation_angles):
                rot = rotate_dihedral(bond[0], bond[1], rot, adjacency_matrix, angle)
            distances = pdist(rot, 'euclidean')
            if (distances > threshold).all():
                conformers.append(rot)
    return conformers

