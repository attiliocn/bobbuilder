#!/usr/bin/env python3

import argparse
import json
import numpy as np
import morfeus
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='Standard JSON input for BobBuilder')
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
core_elements, core_coordinates = morfeus.read_xyz(core_file)
core_adj_matrix = morfeus.utils.get_connectivity_matrix(core_coordinates,core_elements)

core_mol = Chem.rdmolfiles.MolFromXYZFile(core_file)
rdDetermineBonds.DetermineConnectivity(core_mol)

# prepare fragment stuff
# TODO -> loop for each decoration

RotatableBond = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

for decoration in input_data['decorations'][0:1]:

    fragment_file = decoration['fragment']
    fragment_connection_atom = decoration['connecting atoms']
    connection_axis = decoration['bond axis atoms']

    fragment_elements, fragment_coordinates = morfeus.read_xyz(fragment_file)
    fragment_adj_matrix = morfeus.utils.get_connectivity_matrix(fragment_coordinates,fragment_elements)

    fragment_mol = Chem.rdmolfiles.MolFromXYZFile(fragment_file)
    rdDetermineBonds.DetermineConnectivity(fragment_mol)
    rdDetermineBonds.DetermineBondOrders(fragment_mol, charge=0)
    
    rotatable_bonds = fragment_mol.GetSubstructMatches(RotatableBond)