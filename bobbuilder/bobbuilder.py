#!/usr/bin/env python3

import argparse
import os
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

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Standard JSON input for BobBuilder')
parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity level')
# TODO -> include "reset-core" option to reset core number to 1-->n at the end of the loop (default: keep numbering)
# TODO -> include an option to write a template of the input file then exit
args = parser.parse_args()

# load the periodic table database
with open(os.path.join(__location__,'data','periodictable.json')) as f:
    periodic_table = json.load(f)

# load bobbuilder input file
with open(args.input) as f:
    input_data = json.load(f)

# convert atom numbers from the input file to 0-indexed numpy arrays
input_data['core atoms'] = np.array(input_data['core atoms']) - 1
for decoration in input_data['decorations']:
    for key in ['replace at', 'connecting atoms', 'bond axis atoms']:
        decoration[key] = np.array(decoration[key]) - 1

# core details
core_file = input_data['core']
core_atoms = input_data['core atoms']

core_elements, core_coordinates = morfeus.read_xyz(core_file)
#core_adj_matrix = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

core_mol = Chem.rdmolfiles.MolFromXYZFile(core_file)
rdDetermineBonds.DetermineConnectivity(core_mol)

RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

decoration_i = 0
for decoration in input_data['decorations']:

    if decoration_i == 0:
        core_atoms_ = core_atoms.copy()
        core_elements_ = core_elements.copy()
        core_coordinates_ = core_coordinates.copy()
        core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)
    
    replacement_i = 0
    for core_replace_atom in decoration['replace at']:

        if args.verbose:
            print(f"Running decoration {decoration_i} replacing No {replacement_i}")
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-0.xyz', mode='w') as f:
                f.write(xyz_file)

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
        core_neighbours = np.array(find_neighbors(core_adj_matrix_, core_replace_atom, excluded_atoms=core_atoms_))
        core_neighbours = np.delete(core_neighbours, np.where(core_neighbours == core_replace_atom))

        if len(core_neighbours) > 0:
            core_replace_atom_ = core_replace_atom.copy()
            core_replace_atom -= len(np.where(core_neighbours < core_replace_atom)[0])

            for i,atom in enumerate(core_atoms_):
                core_atoms_[i] -= len(np.where(core_neighbours < atom)[0])

            core_coordinates_ = np.delete(core_coordinates_, core_neighbours, axis=0)
            core_elements_ = np.delete(core_elements_, core_neighbours, axis=0)
            core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)

            # you need to renumber core atoms after this procedure

        else:
            core_replace_atom_ = core_replace_atom.copy()
            core_coordinates_ = core_coordinates_.copy()
            core_elements_ = core_elements_.copy()

        if args.verbose:
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-1.xyz', mode='w') as f:
                f.write(xyz_file)

        # remove fragment hydrogens or side chain as per user request
        def remove_sidechain(arg):
            pass

        coordinates_all = []
        intersection_volume_all = []
        elements_join = np.concatenate([core_elements_, fragment_elements])

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
            fragment_coordinates_ = frag_conf_coordinates.copy()
            fragment_elements_ = fragment_elements.copy()

            # translate core to origin
            core_translation = core_coordinates_[core_replace_atom_].copy()
            _core_coordinates = core_coordinates_.copy()
            _core_coordinates -= core_translation

            # translate fragment to origin
            fragment_coordinates_ -= fragment_coordinates_[fragment_connection_atom]

            #identify the core-axis
            axis_point2_atom = np.where(core_adj_matrix_[core_replace_atom] == 1)[0][0]
            core_axis = _core_coordinates[core_replace_atom] - _core_coordinates[axis_point2_atom]

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
            _core_coordinates = np.delete(_core_coordinates, core_replace_atom, axis=0)
            _core_elements = np.delete(core_elements_, core_replace_atom)
            elements_join = np.concatenate([_core_elements, fragment_elements_])

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

                coordinates_join = np.vstack([_core_coordinates, fragment_coordinates_])
                distances = pdist(coordinates_join, 'euclidean')

                # if args.verbose:
                #     xyz_file = build_xyz_file(elements_join, coordinates_join)
                #     with open(f'd{decoration_i}r{replacement_i}_rotations.xyz', mode='a') as f:
                #         f.write(xyz_file)

                if (distances > .990).all():
                    coordinates_all.append(coordinates_join)
                else:
                    continue

                intersection_volumes = sphere_intersection_volumes(
                    _core_coordinates,
                    fragment_coordinates_,
                    radii_a=[periodic_table['vdw radii'][i] for i in _core_elements],
                    radii_b =[periodic_table['vdw radii'][i] for i in fragment_elements_]
                )
                intersection_volume = np.array(intersection_volumes).sum()
                intersection_volume_all.append(intersection_volume)

        if args.verbose:
            print(f"No of valid geometries {len(coordinates_all)}")
        intersection_volume_all = np.array(intersection_volume_all)
        optimal_rotation = intersection_volume_all.argmin()

        best_coordinates = coordinates_all[optimal_rotation]
        best_coordinates += core_translation

        # rewrite best coordinates + elements, conserving the core numbering scheme
        # rplt atom coordinates number = core that it replaced number
        # other core atoms -> some correspondence is needed

        joint_coordinates = np.concatenate([elements_join.reshape(-1,1), best_coordinates.reshape(-1,3)], axis=1)
        
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
        
        _ = reorder_xyz(joint_coordinates, [44-1,43-1,46-1,], [2-1,3-1,4-1])
        # elements_join = _[:,0]
        # best_coordinates = _[:,1:]

        # dest numbers are the initial core numbering
        # source numbering has to be a match of each core number except for the one that was substituted (this comes for free)

        original_atom_numbers = core_atoms_
        updated_atom_numbers = []
        for atom_idx in original_atom_numbers:
            atom_coords = core_coordinates_[atom_idx].round(4)
            atom_idx_updated = np.where(np.all(best_coordinates.round(4) == atom_coords, axis=1))[0][0]
            updated_atom_numbers.append(atom_idx_updated)

        _ = reorder_xyz(joint_coordinates, updated_atom_numbers, original_atom_numbers)
        # elements_join = _[:,0]
        # best_coordinates = _[:,1:]

        core_elements_ = _[:,0].reshape(-1).astype(np.str_)
        core_coordinates_ = _[:,1:].astype(float)
        core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

        if args.verbose:
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-2.xyz', mode='w') as f:
                f.write(xyz_file)

        replacement_i += 1
    decoration_i += 1

# xyz_file = build_xyz_file(core_elements, core_coordinates)
# with open('rotation.xyz', mode='w') as f:
#     f.write(xyz_file)