import requests
import sys
import os
import re
import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from scipy.spatial import distance
import biographs as bg
import networkx as nx
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# df_train_grouped = pd.read_hdf('./data/data3.h5')


def get_file(name):
    """
    Queries Uniprot for the accession name of the protein in the provided dataset so that it can match it to an AF2
    .PDB file which stores edge information for the protein graph.
    :return: .PDB file location with graph
    """
    other = {
        'CMC2_HUMAN': 'Q9UJS0',
        'ST2A1_HUMAN': 'Q06520',
        'TNS2_HUMAN': 'Q63HR2',
        'M4K3_HUMAN': 'Q8IVH8',
        'AL3A1_HUMAN': 'P30838',
    }
    if name in other.keys():
        accession = other[name]
    else:
        requestURL = "https://rest.uniprot.org/uniprotkb/search?fields=accession%2Creviewed%2Cid%2Cprotein_name" \
                     "%2Cgene_names%2Corganism_name%2Clength&query=%28{}%29".format(name)
        r = requests.get(requestURL, headers={"Accept": "application/json"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        accession = r.json()['results'][0]['primaryAccession']

    rootdir = './data/AF2 Dataset'
    regex = re.compile(f"^.*{accession}.*$")
    for file in os.listdir(rootdir):
        if regex.match(file):
            return rootdir + '/' + file

    print(f'FILE DOES NOT EXIST!!: {name}')


def reference_embedding(row):
    protein_name = row['entry']
    grouped_row = df_train_grouped.loc[df_train_grouped['entry'] == protein_name]
    protein_embedding = grouped_row['embeddings'].to_numpy()[0]
    index = row['entry_index']
    return protein_embedding[index]


class LigandBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(LigandBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['af2_dataset_training_labeled.csv']

    @property
    def processed_file_names(self):
        return ['processed.dataset']

    def download(self):
        pass

    def process(self):
        debug = True
        data_list = []

        skip_list = ['NF1_HUMAN', 'MACF1_HUMAN', 'HUWE1_HUMAN', 'DMD_HUMAN']
        parser = PDBParser()

        df_train = pd.read_csv(self.raw_paths[0]).drop('Unnamed: 0', axis=1)
        df_train['embeddings'] = df_train.apply(reference_embedding, axis =1)

        df = df_train

        bool_cols = [col for col in df.columns if df[col].dtype == bool]
        df[bool_cols] = df[bool_cols].astype(int)
        grouped_df = df.groupby('entry')

        for entry, group in tqdm(grouped_df):
            drop_cols = ['y_Ligand', 'annotation_sequence', 'annotation_atomrec', 'entry', 'embeddings']
            x = group.loc[group['entry'] == entry, group.columns] \
                .sort_values(by='entry_index') \
                .drop(drop_cols, axis=1).values
            y = group['y_Ligand'].values

            edges = []

            if entry in skip_list:
                print('Manually looking for edges...')
                for i in range(len(group)):
                    for j in range(i + 1, min(i + 50, len(group))):
                        a = np.asarray(group.iloc[i][['coord_X', 'coord_Y', 'coord_Z']])
                        b = np.asarray(group.iloc[j][['coord_X', 'coord_Y', 'coord_Z']])
                        dist = distance.euclidean(a, b)
                        if dist <= 6:
                            edges.append([i, j])
                edges = np.asarray(edges).T
            else:
                file = get_file(entry)
                structure = parser.get_structure(1, file)
                p1 = {chain.id:seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}['A']
                p2 = "".join(df[df['entry'] == entry]['annotation_sequence'].values)

                molecule = bg.Pmolecule(file)
                network = molecule.network()

                for i in range(len(p2) + 1, len(p1) + 1):
                    node_to_remove = 'A' + str(i)
                    network.remove_node(node_to_remove)

                edges = np.asarray(list(network.edges)).T
                edges = [[int(s[1:]) for s in edges[0]], [int(s[1:]) for s in edges[1]]]

            x = torch.FloatTensor(x)
            x_acc = []
            for i in range(len(x)):
                x_acc.append(torch.cat([x[i], group['embeddings'].iloc[i]], 0).numpy())
            x = torch.FloatTensor(x_acc)

            y = torch.FloatTensor(y)
            edges = torch.tensor(edges, dtype=torch.long)

            if edges[0][0] == 1:
                edges = edges - 1

            if debug:
                print(x[0])
                debug = False

            graph = Data(x=x, y=y, edge_index=edges)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
