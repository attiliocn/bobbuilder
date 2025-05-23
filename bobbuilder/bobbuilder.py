#!/usr/bin/env python3

import argparse
import os
import json
import numpy as np
import morfeus
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.spatial.distance import pdist

from util.graph_tools import find_neighbors
from util.xyz_tools import build_xyz_file, reorder_xyz, rotate_dihedral, get_conformers
from util.kabsch import kabsch_algorithm
from util.geometry import axisangle_to_q, qv_mult, sphere_intersection_volumes,rmsd_matrix, get_duplicates_rmsd_matrix

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

# round core coordinates to 4 decimal places
core_coordinates = core_coordinates.round(4)

# rdkit smarts pattern for rotatable bondss
RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

# remap core to 1,...,n atom numbers
fixed_core = np.array([i for i in range(len(core_atoms))])
_ = reorder_xyz(
    np.concatenate([core_elements.reshape(-1,1), core_coordinates.reshape(-1,3)], axis=1),
    core_atoms,
    fixed_core,
)
if args.verbose:
    print(f"Core atoms were renumbered\nOriginal core numbers: {core_atoms+1}\nCurrent core numbers:{fixed_core+1}\n")
core_atoms_ = fixed_core
core_elements_ = _[:,0].reshape(-1).astype(np.str_)
core_coordinates_ = _[:,1:].astype(float)
core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)


# keep track of this information
added_fragments_coordinates = np.array([[],[],[]]).reshape(-1,3)
added_fragments_atom_numbers = np.array([])

# engine (v3)
for decoration_i, decoration in enumerate(input_data['decorations'], 1):
    if args.verbose:
        #print(f"Running decoration number {decoration_i}")
        print("{:#^25}".format(f" DECORATION {decoration_i} "))

    current_decoration = {}

    # map added coordinates to actual atom numbers
    if decoration_i > 0:
        for coordinate in added_fragments_coordinates:
            atom_number = np.where(np.all(core_coordinates_.round(4) == coordinate, axis=1))[0][0]
            added_fragments_atom_numbers = np.append(added_fragments_atom_numbers, atom_number)
        ignore_atoms_ = np.append(core_atoms_, added_fragments_atom_numbers)

        if args.verbose:
            print(f"New atoms added at positions: {added_fragments_atom_numbers+1}\n")
            print(f"Current atoms to ignore: {ignore_atoms_+1}")
        
    else:
        ignore_atoms_ = core_atoms_.copy()
        if args.verbose:
            print(f"Current atoms to ignore: {ignore_atoms_+1}")
    
    if args.verbose:
        print(f"Processing fragment {decoration['fragment']} of decoration {decoration_i}")

    fragment_path = decoration['fragment']
    fragment_elements, fragment_coordinates = morfeus.read_xyz(fragment_path)
    fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(
        fragment_coordinates, fragment_elements
    )

    fragment_connecting_atoms = decoration['connecting atoms']
    fragment_bond_axis_atoms = decoration['bond axis atoms']

    fragment_mol = Chem.rdmolfiles.MolFromXYZFile(fragment_path)
    rdDetermineBonds.DetermineConnectivity(fragment_mol)
    rdDetermineBonds.DetermineBondOrders(fragment_mol, charge=0) # TODO -> 
                                                                 # write code to handle cases 
                                                                 # where charge !=0
    rotatable_bonds = fragment_mol.GetSubstructMatches(RotatableBond)

    if args.verbose:
        print(f"Current rotatable bonds for fragment {fragment_path}:")
        print(np.array(rotatable_bonds)+1)
        print(f"Number of rotatable bonds: {len(rotatable_bonds)}")
    
    # run the deprotonate procedure
    if decoration['deprotonate'] == True:

        if args.verbose:
            print("Fragment will be deprotonated")

        for i, atom in enumerate(fragment_connecting_atoms):
            if decoration['make terminal'] == True:
                _neighbours = find_neighbors(
                    fragment_adj_matrix, atom, excluded_atoms=decoration['bond axis atoms'],
                    recursive=True
                    )
            elif decoration['make terminal'] == False:
                _neighbours = find_neighbors(
                    fragment_adj_matrix, atom, excluded_atoms=decoration['bond axis atoms'],
                    recursive=False
                    )
            _neighbours = np.array(_neighbours)
            _neighbours = np.delete(_neighbours, np.where(_neighbours == atom))

            if args.verbose:
                print(f"Found neighbours of atom {atom+1}: {_neighbours+1}")

            deleted_neighbours = np.array([])
            for i, neighbour in enumerate(_neighbours):
                _ = neighbour.copy()
                if (neighbour > deleted_neighbours[:i]).any():
                        neighbour -= i
                        if args.verbose:
                            print(f"Neighbour atom number updated from {_+1} to {neighbour+1}")
                if fragment_elements[neighbour] == 'H' or decoration['make terminal'] == True:
                    if args.verbose:
                        print(f"Neighbour {neighbour+1} of atom {atom+1} will be deleted")

                    fragment_coordinates = np.delete(fragment_coordinates, neighbour, axis=0)
                    fragment_elements = np.delete(fragment_elements, neighbour, axis=0)
                    fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)

                    deleted_neighbours = np.append(deleted_neighbours, _)
                    if args.verbose:
                        print(f"Atom {neighbour+1} deleted")
                        print(f"Current deleted atoms: {deleted_neighbours+1}")
                    
                    if atom > neighbour:
                        fragment_connecting_atoms[i] -= 1
                        fragment_bond_axis_atoms -= 1
                    _rotatable_bonds = np.array(rotatable_bonds)
                    _rotatable_bonds[_rotatable_bonds > neighbour] -= 1
                    rotatable_bonds = _rotatable_bonds.copy()
                    if decoration['make terminal'] == False:
                        break

        if args.verbose:
            print(f"Updated rotatable bonds for fragment {fragment_path}:")
            print(np.array(rotatable_bonds)+1)
            print(f"Number of rotatable bonds: {len(rotatable_bonds)}")
    
    fragment_conformers = get_conformers(
        coordinates = fragment_coordinates,
        adjacency_matrix = fragment_adj_matrix,
        rotatable_bonds = rotatable_bonds,
        numconfs=(len(fragment_coordinates)*len(rotatable_bonds)*2),
        threshold=.960
    )

    fragment_conformers.append(fragment_coordinates)

    if args.verbose:
        print(f"Number of conformers: {len(fragment_conformers)}")
        with open(f"tmp.decor{decoration_i}.confs0.xyz", mode='w') as f:
            for coordinates in fragment_conformers:
                xyz_file = build_xyz_file(fragment_elements, coordinates)
                f.write(xyz_file)

    rmsd_distance_matrix = rmsd_matrix(fragment_conformers)
    to_delete = get_duplicates_rmsd_matrix(rmsd_distance_matrix, threshold=0.55)
    fragment_conformers = [fragment_conformers[i] for i in range(len(fragment_conformers)) if i not in to_delete]

    if args.verbose:
        print(f"Remaining conformers after deduplication: {len(fragment_conformers)}")
        with open(f"tmp.decor{decoration_i}.confs1.xyz", mode='w') as f:
            for coordinates in fragment_conformers:
                xyz_file = build_xyz_file(fragment_elements, coordinates)
                f.write(xyz_file)
    
    # replacement level
    for replacement_i, core_replace_atom in enumerate(decoration['replace at']):
        if args.verbose:
            print(f"Running replacement {decoration_i}.{replacement_i}")

        # map current core_replace_atom to the actual core atom number
        core_replace_atom_coords = core_coordinates[core_replace_atom].round(4)
        core_replace_atom_ = np.where(np.all(core_coordinates_.round(4) == core_replace_atom_coords, axis=1))[0][0]

        if args.verbose:
            print(f"Original core atom numbers: {core_replace_atom+1}\nUpdated core atom number: {core_replace_atom_+1}")
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'tmp.decor{decoration_i}-{replacement_i}_core0.xyz', mode='w') as f:
                f.write(xyz_file)

        # check if core is a terminal-type replacement
        # if not, remove all neighbours of the replacement point (except for core and core-neighbours)
        # TODO -> adjust every occurence of adj_matrix updates
        core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)
        core_neighbours = np.array(find_neighbors(core_adj_matrix_, core_replace_atom_, excluded_atoms=ignore_atoms_))
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
            with open(f'tmp.decor{decoration_i}-{replacement_i}_core1.xyz', mode='w') as f:
                f.write(xyz_file)
            print(f"Core atoms: {core_atoms_+1}")
            print(f"Replace at: {core_replace_atom_+1}")

        # translate core to origin
        core_translation = core_coordinates_[core_replace_atom_].copy()
        _core_coordinates = core_coordinates_.copy()
        _core_coordinates -= core_translation
        #identify the core-axis
        # TODO -> clean up these disgusting workarounds for adj_matrix recalculations

        # IMPORTANT -> The scale_factor for this connectivity below should be higher than the default of 1.2
        # that is because C-H bonds should be streched in the template, otherwise bobbuilder will crash because
        # atoms will flag as collisions after decorating. This elongation causes the _core_adj_matrix below report that
        # such H has no neighbours and thus the core-vector is measured from the H to atom 0, thus 
        # resulting in distortions to the final decorated molecule

        # TODO -> Fix DISGUSTING workarounds such those described above!

        _core_adj_matrix = morfeus.utils.get_connectivity_matrix(_core_coordinates,core_elements_, scale_factor=1.50)
        if len(np.where(_core_adj_matrix[core_replace_atom_] == 1)[0]) == 0:
            axis_point2_atom = 0
        else:
            axis_point2_atom = np.where(_core_adj_matrix[core_replace_atom_] == 1)[0][0]
        core_axis = _core_coordinates[core_replace_atom_] - _core_coordinates[axis_point2_atom]
        if args.verbose:
            print(f"Current core-axis atoms: {core_replace_atom_+1} -> {axis_point2_atom+1}")
            print(f"Current fragment-axis atoms: {fragment_connecting_atoms+1} -> {fragment_bond_axis_atoms+1}")
        # remove the core atom
        _core_coordinates = np.delete(_core_coordinates, core_replace_atom_, axis=0)
        _core_elements = np.delete(core_elements_, core_replace_atom_)

        coordinates_all = []
        fragments_all = []
        intersection_volume_all = []
        all_distances = np.array([])
        elements_join = np.concatenate([core_elements_, fragment_elements])

        # test each conformer for the best fit with core
        for frag_conf_coordinates in fragment_conformers:
            fragment_coordinates_ = frag_conf_coordinates.copy()
            fragment_elements_ = fragment_elements.copy()

            # translate fragment to origin
            fragment_coordinates_ -= fragment_coordinates_[fragment_connecting_atoms]

            # set fragment axis coordinates
            # len(axis) = 1, take the coordinates of this atom
            # len(axis) > 1, then take the average of the atoms coordinates
            if len(fragment_bond_axis_atoms) == 1:
                fragment_axis = fragment_coordinates_[fragment_bond_axis_atoms[0]]
            elif len(fragment_bond_axis_atoms) > 1:
                _ = np.array([fragment_coordinates_[i] for i in fragment_bond_axis_atoms])
                fragment_axis = np.mean(_, axis=0)

            # rotate the fragment such the fragment axis matches the core axis
            R, t = kabsch_algorithm(
                fragment_axis.reshape(3,-1),
                core_axis.reshape(3,-1),
                center=False
            )
            fragment_coordinates_ = (R @ fragment_coordinates_.T).T
                        
            # join core elements with fragment elements
            elements_join = np.concatenate([_core_elements, fragment_elements_])

            # rotate the fragment 360 degrees about the fragment axis
            # find the optimal positioning for the rigid fragment
            # by reducing the vdW spheres superposition
            rotation_steps = 15
            rotation_stepsize_rad = np.radians(360) / rotation_steps

            for rotation in range(rotation_steps-1):
                quat = axisangle_to_q(core_axis, rotation_stepsize_rad)
                for i, atom_coordinates in enumerate(fragment_coordinates_):
                    new_coordinates = qv_mult(quat, tuple(atom_coordinates))
                    fragment_coordinates_[i] = new_coordinates
                coordinates_join = np.vstack([_core_coordinates, fragment_coordinates_])

                distances = pdist(coordinates_join, 'euclidean')

                coordinates_without_H = coordinates_join[np.where(elements_join != 'H')[0]]
                distances_without_H = pdist(coordinates_without_H)

                if distances_without_H.min() < 1.15:
                    continue
                else: 
                    all_distances = np.append(all_distances, distances.min())
                    fragments_all.append(fragment_coordinates_.copy())
                    coordinates_all.append(coordinates_join.copy())

        threshold_955 = (all_distances >= .955)
        threshold_950 = (all_distances >= .950)
        threshold_890 = (all_distances >= .850)
        if args.verbose:
            print(f"Total geometries: {len(all_distances)}")
            print(f"Number of valid geometries (.955 threshold): {threshold_955.sum()}")
            print(f"Number of valid geometries (.950 threshold): {threshold_950.sum()}")
            print(f"Number of valid geometries (.850 threshold): {threshold_890.sum()}")
        if threshold_955.sum() > 0:
            coordinates_all = [coordinates_all[i] for i in np.where(threshold_955)[0]]
            fragments_all = [fragments_all[i] for i in np.where(threshold_955)[0]]
        elif threshold_950.sum() > 0:
            coordinates_all = [coordinates_all[i] for i in np.where(threshold_950)[0]]
            fragments_all = [fragments_all[i] for i in np.where(threshold_950)[0]]
        elif threshold_890.sum() > 0:
            coordinates_all = [coordinates_all[i] for i in np.where(threshold_890)[0]]
            fragments_all = [fragments_all[i] for i in np.where(threshold_890)[0]]
        else:
            if args.verbose:
                for threshold in np.arange(0.740,0.955,0.005):
                    print(f"Threshold: {threshold.round(3):.3f} => {(all_distances >= threshold).sum()} valid geometries")
                with open(f"tmp.decor{decoration_i}-{replacement_i}.failed.confs.xyz", mode='w') as f:
                    for coordinates in coordinates_all:
                        xyz_file = build_xyz_file(elements_join, coordinates)
                        f.write(xyz_file)
            raise ValueError("Impossible to fit this ligand")
        
        if args.verbose:
            with open(f"tmp.decor{decoration_i}-{replacement_i}.confs.xyz", mode='w') as f:
                for coordinates in coordinates_all:
                    xyz_file = build_xyz_file(elements_join, coordinates)
                    f.write(xyz_file)

        for fragment_coordinates_ in fragments_all:
            intersection_volumes = sphere_intersection_volumes(
                _core_coordinates,
                fragment_coordinates_,
                radii_a=[periodic_table['vdw radii'][i] for i in _core_elements],
                radii_b =[periodic_table['vdw radii'][i] for i in fragment_elements_]
            )
            intersection_volume = np.array(intersection_volumes).sum()
            intersection_volume_all.append(intersection_volume)
            
        intersection_volume_all = np.array(intersection_volume_all)
        optimal_rotation = intersection_volume_all.argmin()

        if args.verbose:
            print(f"Best Rotation Intersection Volume: {intersection_volume_all.min().round(1)}")
            print(f"Worst Rotation Intersection Volume: {intersection_volume_all.max().round(1)}")

        best_coordinates = coordinates_all[optimal_rotation] + core_translation
        best_fragment = fragments_all[optimal_rotation] + core_translation
        
        # update core coordinates for the next decoration cycle
        core_elements_ = elements_join
        core_coordinates_ = best_coordinates

        # identify current core atom numbers
        # then remap core to 1,...,n atom numbers again
        if args.verbose:
            print(f"Current core atom numbers: {core_atoms_+1}")
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'tmp.decor{decoration_i}-{replacement_i}_core2.xyz', mode='w') as f:
                f.write(xyz_file)

        current_core_atoms = []
        for atom_idx in core_atoms:
            atom_coords = core_coordinates[atom_idx].round(4)
            if args.verbose:
                print(f"Atom (original number): {atom_idx+1} -> {atom_coords}")
            atom_idx_updated = np.where(np.all(core_coordinates_.round(4) == atom_coords, axis=1))[0][0]
            if args.verbose:
                print(f"  New number -> {atom_idx_updated+1}")
            current_core_atoms.append(atom_idx_updated)
        current_core_atoms = np.array(current_core_atoms)
        
        if args.verbose:
            print(f"Identified core atom numbers: {current_core_atoms+1}")

        _ = reorder_xyz(
            np.concatenate([core_elements_.reshape(-1,1), core_coordinates_.reshape(-1,3)], axis=1),
            current_core_atoms,
            fixed_core,
        )
        core_atoms_ = fixed_core
        core_elements_ = _[:,0].reshape(-1).astype(np.str_)
        core_coordinates_ = _[:,1:].astype(float)
        core_adj_matrix_ = morfeus.utils.get_connectivity_matrix(core_coordinates_,core_elements_)

        added_fragments_coordinates = np.append(added_fragments_coordinates, best_fragment, axis=0)
        added_fragments_coordinates = added_fragments_coordinates.round(4)

        if args.verbose:
            print(f"The core has been renumbered.\nCurrent core atom numbers: {core_atoms_+1}")

        if args.verbose:
            xyz_file = build_xyz_file(core_elements_, core_coordinates_)
            with open(f'tmp.decor{decoration_i}-{replacement_i}_core3.xyz', mode='w') as f:
                f.write(xyz_file)
            print("Decoration done\n\n")

if not args.verbose:
    xyz_file = build_xyz_file(core_elements_, core_coordinates_)
    with open(args.input.replace('json','xyz'), mode='w') as f:
        f.write(xyz_file)
