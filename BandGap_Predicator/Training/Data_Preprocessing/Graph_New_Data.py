import numpy as np
from rdkit import Chem

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import networkx as nx
import os

def atom_features(atom):
    symbles = ['Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C', 'Ca', 'Cd',
                 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F', 'Fe', 'Ga', 'Ge', 'H',
                 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na',
                 'Nb', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb',
                'Sc', 'Se', 'Si', 'Sn', 'Sr', 'Ta',
                 'Te', 'Ti', 'Tl', 'U', 'V', 'W', 'Y', 'Zn', 'Zr', 'Unknown']
    
    hybridizations =  [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
            'other',
    ]
    
    stereos =  [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    
    symbol = [0.] * len(symbles)
    symbol[symbles.index(atom.GetSymbol())] = 1.
    #comment degree from 6 to 8
    print(atom.GetDegree())
    degree = [0.] * 15
    
    degree[atom.GetDegree()] = 1.
    formal_charge = atom.GetFormalCharge()
    radical_electrons = atom.GetNumRadicalElectrons()
#     hybridization = [0.] * len(hybridizations)
#     hybridization[hybridizations.index(atom.GetHybridization())] = 1.
    aromaticity = 1. if atom.GetIsAromatic() else 0.
    hydrogens = [0.] * 5
    hydrogens[atom.GetTotalNumHs()] = 1.
    chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
    chirality_type = [0.] * 2
    if atom.HasProp('_CIPCode'):
        chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
    
    return np.array(symbol + degree + [formal_charge] +
                                 [radical_electrons] +
                                 [aromaticity] + hydrogens + [chirality] +
                                 chirality_type)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        return None
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index





class Molecule_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', y=None, transform=None,
                 pre_transform=None,smile_graph=None,smiles=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
#             print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(y,smile_graph,smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, y,smile_graph,smiles):
       
        data_list = []
        data_len = len(y)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smile = smiles[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smile]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            
            
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])