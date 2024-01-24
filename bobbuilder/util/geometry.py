import numpy as np 
from math import sin, cos, acos, sqrt
from util.kabsch import kabsch_algorithm
from util.graph_tools import find_neighbors

# quaternion rotations
def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z

def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta

def sphere_intersection_volume(radius_a, radius_b, distance_between_centers):
    if distance_between_centers >= radius_a + radius_b:
        return 0  # Spheres are not intersecting

    if distance_between_centers <= abs(radius_a - radius_b):
        smaller_radius = min(radius_a, radius_b)
        return (4/3) * np.pi * smaller_radius**3  # Smaller sphere is completely inside the larger one

    d = distance_between_centers
    r_a = radius_a
    r_b = radius_b

    # Calculate volume of intersection using the formula for spherical cap volume
    volume = (np.pi / 12) * ((r_a + r_b - d)**2) * (d**2 + 2 * d * (r_a + r_b) - 3 * (r_a - r_b)**2)

    return volume

def sphere_intersection_volumes(matrix_a, matrix_b, radii_a, radii_b):
    volumes = []

    for center_a, radius_a in zip(matrix_a, radii_a):
        for center_b, radius_b in zip(matrix_b, radii_b):
            distance_between_centers = np.linalg.norm(center_a - center_b)

            volume = sphere_intersection_volume(radius_a, radius_b, distance_between_centers)
            volumes.append(volume)

    return volumes

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