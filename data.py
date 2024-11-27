# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:01:17 2024

@author: jspark
"""

import pandas as pd
import json
from torch_geometric.loader import DataLoader


from mol_to_graph import AtomFeaturizer, smiles_to_graph



def load_smiles(input_path):
    
    if '.xlsx' in input_path:
        df = pd.read_excel(input_path)
    elif '.csv' in input_path:
        df = pd.read_csv(input_path)
    elif '.txt' in input_path:
        df = pd.read_table(input_path)
    else:
        raise Exception('Input type is wrong (Only txt, csv, xlsx)')
    
    smiles_list = list(df['SMILES'])
    
    return smiles_list


def Dataloader(smiles_list, config):
    
    # Featurizer setting
    atom_featurizer = AtomFeaturizer(
        allowable_sets = {
            "symbol" : ['C','N','O','S', 'Cl', 'Br', 'I', 'F', 'H', 'ELSE'],
            "degree" : [1, 2, 3, 4],
            "charge" : [-1, 0, 1],
            'chirality' : ['chi_unspecified','chi_tetrahedral_cw','chi_tetrahedral_ccw'],
            'n_hydrogens' : [0, 1, 2, 3],
            'hybridization' : ['sp3', 'sp2', 'sp', 's', 'sp3d', 'sp3d2'],
            'ring' : [0, 1],
            'aromatic' : [0, 1],
            'E_acceptor' : [0, 1]
        }
    )
    
    dataset = []
    for smiles in smiles_list:
        dataset.append(smiles_to_graph(smiles, atom_featurizer=atom_featurizer))
    
    dataloader = DataLoader(dataset, batch_size = config['train_config']['batch_size'])
    
    return dataloader