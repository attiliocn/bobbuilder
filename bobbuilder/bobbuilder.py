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
from util.xyz_tools import build_xyz_file, reorder_xyz
from util.kabsch import kabsch_algorithm
from util.geometry import axisangle_to_q, qv_mult, sphere_intersection_volumes

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Standard JSON input for BobBuilder')
parser.add_argument('-v', '--verbose', action='store_true', help='Increase verbosity level')
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
core_mol = Chem.rdmolfiles.MolFromXYZFile(core_file)
rdDetermineBonds.DetermineConnectivity(core_mol)

# rdkit smarts pattern for rotatable bondss
RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

# remap core to 1,...,n atom numbers
fixed_core = np.array([i for i in range(len(core_atoms))])
_ = reorder_xyz(
    np.concatenate([core_elements.reshape(-1,1), core_coordinates.reshape(-1,3)], axis=1),
    core_atoms,
    fixed_core,
)
core_atoms_ = fixed_core
core_elements_ = _[:,0].reshape(-1).astype(np.str_)
core_coordinates_ = _[:,1:].astype(float)
core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)

# main engine
decoration_i = 0
for decoration in input_data['decorations']:

    replacement_i = 0
    for core_replace_atom in decoration['replace at']:

        # map current core_replace_atom to the actual core atom number
        core_replace_atom_coords = core_coordinates[core_replace_atom].round(4)
        core_replace_atom_ = np.where(np.all(core_coordinates_.round(4) == core_replace_atom_coords, axis=1))[0][0]

        if args.verbose:
            print(f"Running Decoration Nº {decoration_i}.{replacement_i}")
            print(f"Original core atom number: {core_replace_atom+1}, Updated core atom number: {core_replace_atom_+1}")
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-0.xyz', mode='w') as f:
                f.write(xyz_file)

        # check if core is a terminal-type replacement
        # if not, remove all neighbours of the replacement point (except for core and core-neighbours)
        core_neighbours = np.array(find_neighbors(core_adj_matrix_, core_replace_atom_, excluded_atoms=core_atoms_))
        core_neighbours = np.delete(core_neighbours, np.where(core_neighbours == core_replace_atom_))

        if len(core_neighbours) > 0:
            if args.verbose:
                print('Neighbours to the connection point will be removed')
                print(f"Removing atoms {core_neighbours+1}")

            core_replace_atom_ -= len(np.where(core_neighbours < core_replace_atom_)[0])

            for i,atom in enumerate(core_atoms_):
                core_atoms_[i] -= len(np.where(core_neighbours < atom)[0])

            core_coordinates_ = np.delete(core_coordinates_, core_neighbours, axis=0)
            core_elements_ = np.delete(core_elements_, core_neighbours, axis=0)

        if args.verbose:
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-1.xyz', mode='w') as f:
                f.write(xyz_file)
            print(f"Core atoms: {core_atoms_+1}")
            print(f"Replace at: {core_replace_atom_+1}")

        # read fragment data
        fragment_file = decoration['fragment']
        fragment_connection_atom = decoration['connecting atoms']
        connection_axis = decoration['bond axis atoms']

        fragment_elements, fragment_coordinates = morfeus.read_xyz(fragment_file)
        fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)

        fragment_mol = Chem.rdmolfiles.MolFromXYZFile(fragment_file)
        rdDetermineBonds.DetermineConnectivity(fragment_mol)
        rdDetermineBonds.DetermineBondOrders(fragment_mol, charge=0)
        
        rotatable_bonds = fragment_mol.GetSubstructMatches(RotatableBond)

        # deprotonate fragment
        if decoration['deprotonate'] == True:
            if args.verbose:
                print("Fragment will be deprotonated")

            for i, atom in enumerate(fragment_connection_atom):
                if fragment_elements[atom] in ['C', 'O', 'N', 'S']:
                    fragment_neighbours = np.array(find_neighbors(fragment_adj_matrix, atom, excluded_atoms=connection_axis))
                    fragment_neighbours = np.delete(fragment_neighbours, np.where(fragment_neighbours == atom))
                    
                    if args.verbose:
                        print(f"Current neighbours of atom {atom+1}: {fragment_neighbours+1}")

                    for neighbour in fragment_neighbours:
                        if fragment_elements[neighbour] == 'H':
                            if args.verbose:
                                print(f"Neighbour {neighbour+1} of atom {atom+1} will be deleted")
                            fragment_coordinates = np.delete(fragment_coordinates, neighbour, axis=0)
                            fragment_elements = np.delete(fragment_elements, neighbour, axis=0)
                            fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)
                            if atom > neighbour:
                                fragment_connection_atom[i] -= 1
                                connection_axis -= 1
                            # TODO: Increase the efficiency of this implementation!
                            rotatable_bonds_ = np.array(rotatable_bonds)
                            rotatable_bonds_[rotatable_bonds_ > neighbour] -= 1
                            rotatable_bonds = rotatable_bonds_
                            break
        elif decoration['make terminal'] == True:
            for i, atom in enumerate(fragment_connection_atom):
                if fragment_elements[atom] in ['C', 'O', 'N', 'S']:
                    fragment_neighbours = np.array(find_neighbors(fragment_adj_matrix, atom, excluded_atoms=connection_axis))
                    fragment_neighbours = np.delete(fragment_neighbours, np.where(fragment_neighbours == atom))
                    
                    if args.verbose:
                        print("Running the MAKE_TERMINAL routine")
                        print(f"Current neighbours of atom {atom+1}: {fragment_neighbours+1}")

                    for neighbour in fragment_neighbours:
                        if args.verbose:
                            print(f"Neighbour {neighbour+1} of atom {atom+1} will be deleted")
                        fragment_coordinates = np.delete(fragment_coordinates, neighbour, axis=0)
                        fragment_elements = np.delete(fragment_elements, neighbour, axis=0)
                        fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)
                        if atom > neighbour:
                            fragment_connection_atom[i] -= 1
                            connection_axis -= 1
                        rotatable_bonds_ = np.array(rotatable_bonds)
                        rotatable_bonds_[rotatable_bonds_ > neighbour] -= 1
                        rotatable_bonds = rotatable_bonds_

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

            if args.verbose:
                xyz_file = build_xyz_file(fragment_elements_, fragment_coordinates_)
                with open(f'd{decoration_i}r{replacement_i}_confs.xyz', mode='a') as f:
                    f.write(xyz_file)

            # translate core to origin
            core_translation = core_coordinates_[core_replace_atom_].copy()
            _core_coordinates = core_coordinates_.copy()
            _core_coordinates -= core_translation

            # translate fragment to origin
            fragment_coordinates_ -= fragment_coordinates_[fragment_connection_atom]

            #identify the core-axis
            axis_point2_atom = np.where(core_adj_matrix_[core_replace_atom_] == 1)[0][0]
            core_axis = _core_coordinates[core_replace_atom_] - _core_coordinates[axis_point2_atom]

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
            _core_coordinates = np.delete(_core_coordinates, core_replace_atom_, axis=0)
            _core_elements = np.delete(core_elements_, core_replace_atom_)
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
                #     print(f"{rotation} {distances.min()}")

                if (distances > .950).all():
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
        
        # update core coordinates for the next decoration cycle
        core_elements_ = elements_join
        core_coordinates_ = best_coordinates

        # identify current core atom numbers
        # then remap core to 1,...,n atom numbers again
        current_core_atoms = []
        for atom_idx in core_atoms:
            atom_coords = core_coordinates[atom_idx].round(4)
            atom_idx_updated = np.where(np.all(core_coordinates_.round(4) == atom_coords, axis=1))[0][0]
            current_core_atoms.append(atom_idx_updated)
        current_core_atoms = np.array(current_core_atoms)
        
        if args.verbose:
            print(f"Current core atom numbers: {current_core_atoms+1}")
    
        _ = reorder_xyz(
            np.concatenate([core_elements_.reshape(-1,1), core_coordinates_.reshape(-1,3)], axis=1),
            current_core_atoms,
            fixed_core,
        )
        core_atoms_ = fixed_core
        core_elements_ = _[:,0].reshape(-1).astype(np.str_)
        core_coordinates_ = _[:,1:].astype(float)
        core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)
        #TODO -> keep track of the added ligand coordinates to exclude them from the core neighbours search

        if args.verbose:
            print(f"The core has been renumbered.\nCurrent core atom numbers:{core_atoms_}")

        if args.verbose:
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'd{decoration_i}rep{replacement_i}-2.xyz', mode='w') as f:
                f.write(xyz_file)
            print("Decoration done\n\n")
        replacement_i += 1
    decoration_i += 1

xyz_file = build_xyz_file(core_elements_, core_coordinates_)
with open('rotation.xyz', mode='w') as f:
    f.write(xyz_file)