# core
core_file = 'lib/iron-py.xyz'
core_atoms = np.array([1,42,43,44,45,46,47]) - 1
core_replace_atom = (46-1)

core_elements, core_coordinates = morfeus.read_xyz(core_file)
core_adj_matrix = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

core_mol = Chem.rdmolfiles.MolFromXYZFile(core_file)
rdDetermineBonds.DetermineConnectivity(core_mol)
core_mol

# fragment
fragment_file = 'lib/phenol-3.xyz'
fragment_connection_atom = (1-1)
connection_axis = np.array([3]) - 1

fragment_elements, fragment_coordinates = morfeus.read_xyz(fragment_file)
fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)

fragment_mol = Chem.rdmolfiles.MolFromXYZFile(fragment_file)
rdDetermineBonds.DetermineConnectivity(fragment_mol)
rdDetermineBonds.DetermineBondOrders(fragment_mol, charge=0)
display(fragment_mol)

RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
rotatable_bonds = fragment_mol.GetSubstructMatches(RotatableBond)

# check if core is a terminal-type replacement
# if not, remove all neighbours of the replacement point (except for core and core-neighbours)
core_neighbours = np.array(find_neighbors(core_adj_matrix, core_replace_atom, excluded_atoms=core_atoms))
if len(core_neighbours) > 1:
    core_neighbours = np.delete(core_neighbours, np.where(core_neighbours == core_replace_atom))
    core_replace_atom -= len(np.where(core_neighbours < core_replace_atom)[0])
    for i,atom in enumerate(core_atoms):
        core_atoms[i] -= len(np.where(core_neighbours < atom)[0])
    core_coordinates = np.delete(core_coordinates, core_neighbours, axis=0)
    core_elements = np.delete(core_elements, core_neighbours, axis=0)
    core_adj_matrix = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

xyz_file = build_xyz_file(core_elements, core_coordinates)
with open('tmp/core-mod.xyz', mode='w') as f:
    f.write(xyz_file)

# remove fragment hydrogens or side chain as per user request
def remove_sidechain(arg):
    pass

coordinates_all = []
intersection_volume_all = []
elements_join = np.concatenate([core_elements, fragment_elements])

# generate a set of conformers for the fragment by dihedral rotations
fragment_conformers = []
for rotation in range(500):
    rotation_angles = np.random.uniform(np.radians(-180), np.radians(180), len(rotatable_bonds))
    rot = fragment_coordinates.copy()
    for bond, angle in zip(rotatable_bonds, rotation_angles):
        rot = rotate_dihedral(bond[0], bond[1], rot, fragment_adj_matrix, angle)
    distances = pdist(rot, 'euclidean')
    if (distances > .990).all():
        fragment_conformers.append(rot) 

for frag_conf_coordinates in fragment_conformers:
    # copy the core coordinates and elements
    core_coordinates_ = core_coordinates.copy()
    core_elements_ = core_elements.copy()

    fragment_coordinates_ = frag_conf_coordinates.copy()
    fragment_elements_ = fragment_elements.copy()

    # translate core to origin
    core_translation = core_coordinates_[core_replace_atom].copy()
    core_coordinates_ -= core_translation

    # translate fragment to origin
    fragment_coordinates_ -= fragment_coordinates_[fragment_connection_atom]

    #identify the core-axis
    axis_point2_atom = np.where(core_adj_matrix[core_replace_atom] == 1)[0][0]
    core_axis = core_coordinates_[core_replace_atom] - core_coordinates_[axis_point2_atom]

    # set fragment axis coordinates
    # len(axis) = 1, take the coordinates of this atom
    # len(axis) > 1, then take the average of the atoms coordinates

    if len(connection_axis) == 1:
        fragment_axis = fragment_coordinates_[connection_axis[0]]
    elif len(connection_axis) > 1:
        _ = np.array([fragment_coordinates_[i] for i in connection_axis])
        fragment_axis = np.mean(_, axis=0)

    # rotate the fragment such the fragment axis matches the core axis
    R, t = kabsch_algorithm(
        fragment_axis.reshape(3,-1),
        core_axis.reshape(3,-1),
        center=False
    )
    fragment_coordinates_ = (R @ fragment_coordinates_.T).T

    # remove the core atom
    core_coordinates_ = np.delete(core_coordinates_, core_replace_atom, axis=0)
    core_elements_ = np.delete(core_elements_, core_replace_atom)
    elements_join = np.concatenate([core_elements_, fragment_elements_])

    # rotate the fragment 360 degrees about the fragment axis
    # find the optimal positioning for the rigid fragment
    # by reducing vdW spheres superposition
    rotation_steps = 50
    rotation_stepsize_rad = np.radians(360) / rotation_steps

    for rotation in range(rotation_steps-1):
        quat = axisangle_to_q(core_axis, rotation_stepsize_rad)
        for i, atom_coordinates in enumerate(fragment_coordinates_):
            new_coordinates = qv_mult(quat, tuple(atom_coordinates))
            fragment_coordinates_[i] = new_coordinates

        coordinates_join = np.vstack([core_coordinates_, fragment_coordinates_])
        distances = pdist(coordinates_join, 'euclidean')
        if (distances > .990).all():
            coordinates_all.append(coordinates_join)
        else:
            continue

        intersection_volumes = sphere_intersection_volumes(
            core_coordinates_,
            fragment_coordinates_,
            radii_a=[vdw_radii[i] for i in core_elements_],
            radii_b =[vdw_radii[i] for i in fragment_elements_]
        )
        intersection_volume = np.array(intersection_volumes).sum()
        intersection_volume_all.append(intersection_volume)

intersection_volume_all = np.array(intersection_volume_all)
optimal_rotation = intersection_volume_all.argmin()

best_coordinates = coordinates_all[optimal_rotation]
best_coordinates += core_translation

xyz_file = build_xyz_file(elements_join, best_coordinates)
with open('tmp/rotation.xyz', mode='w') as f:
    f.write(xyz_file)