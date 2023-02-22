import torch
from tqdm import tqdm
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
from scipy.spatial import distance


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
        print(self.raw_paths)
        print(self.processed_paths)

        data_list = []

        df = pd.read_csv(self.raw_paths[0]).drop('Unnamed: 0', axis=1)

        bool_cols = [col for col in df.columns if df[col].dtype == bool]
        df[bool_cols] = df[bool_cols].astype(int)
        grouped_df = df.groupby('entry')

        for entry, group in tqdm(grouped_df):
            drop_cols = ['y_Ligand', 'annotation_sequence', 'annotation_atomrec', 'entry']
            x = group.loc[group['entry'] == entry, group.columns] \
                     .sort_values(by='entry_index') \
                     .drop(drop_cols, axis=1).values
            y = group['y_Ligand'].values

            edge_list = []
            for i in range(len(group)):
                for j in range(i + 1, min(i + 10, len(group))):
                    a = np.asarray(group.iloc[i][['coord_X', 'coord_Y', 'coord_Z']])
                    b = np.asarray(group.iloc[j][['coord_X', 'coord_Y', 'coord_Z']])
                    dist = distance.euclidean(a, b)
                    if dist <= 6:
                        edge_list.append([i, j])

            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y)
            edge_list = torch.tensor(edge_list, dtype=torch.long)

            graph = Data(x=x, y=y, edge_index=edge_list)
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
