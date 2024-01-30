#!/usr/bin/env python3

import argparse
import json
import numpy as np
import morfeus
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.spatial.distance import pdist

from util.geometry import rotate_dihedral
from util.graph_tools import find_neighbors
from util.xyz_tools import build_xyz_file
from util.kabsch import kabsch_algorithm
from util.geometry import axisangle_to_q, qv_mult, sphere_intersection_volumes

vdw_radii = None

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Standard JSON input for BobBuilder')
# TODO -> include "keep-numbers" option to mantain the core numbering intact
# TODO -> include a "verbose" option for debug
# TODO -> include an option to write a template of the input file then exit
args = parser.parse_args()

with open(args.input) as f:
    input_data = json.load(f)

# convert atom numbers from the input file to 0-indexed numpy arrays
input_data['core atoms'] = np.array(input_data['core atoms']) - 1
for decoration in input_data['decorations']:
    for key in ['replace at', 'connecting atoms', 'bond axis atoms']:
        decoration[key] = np.array(decoration[key]) - 1


# prepare core stuff
core_file = input_data['core']
core_atoms = input_data['core atoms']

core_elements, core_coordinates = morfeus.read_xyz(core_file)
core_adj_matrix = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

core_mol = Chem.rdmolfiles.MolFromXYZFile(core_file)
rdDetermineBonds.DetermineConnectivity(core_mol)

# prepare fragment stuff
# TODO -> loop for each decoration

RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

for decoration in input_data['decorations'][0:1]:
    core_replace_atom = decoration['replace at'][0] #TODO -> this setting is defined as a list
                                                    # in the json setup file. Regardless of this
                                                    # definition, it is converted to np.array then 0-indexed
                                                    # When converting to np.array, select the first element only,
                                                    # because this variable has to be an integer.

    fragment_file = decoration['fragment']
    fragment_connection_atom = decoration['connecting atoms']
    connection_axis = decoration['bond axis atoms']

    fragment_elements, fragment_coordinates = morfeus.read_xyz(fragment_file)
    fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)

    fragment_mol = Chem.rdmolfiles.MolFromXYZFile(fragment_file)
    rdDetermineBonds.DetermineConnectivity(fragment_mol)
    rdDetermineBonds.DetermineBondOrders(fragment_mol, charge=0)
    
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
    with open('rotation.xyz', mode='w') as f:
        f.write(xyz_file)