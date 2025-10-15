from joblib import Parallel, delayed
import dgl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from HHGNN.graph import create_heterograph
import numpy as np
import scipy.io as sio
import torch
import os
import sys





class DataLoader():
    def __init__(self,opt):
        print("Initializing DataLoader...")
        self.opt =opt

        self.num_classes = opt.args.num_classes
        self.subject_ids, self.group_dict,self.labels = self.load_subjects_and_labels()
        # self.fc_data = self.load_fc_matrices()
        self.k=opt.args.k
        self.num_nodes = len(self.subject_ids)
        mapping_sizes = [500, 400, 300, 200, 100]
        self.atlas_dir = opt.args.atlas_dir

        mapping_paths = []
        for i, src in enumerate(mapping_sizes):
            for dst in mapping_sizes[i + 1:]:
                path = os.path.join(self.atlas_dir, f"mapping_{src}to{dst}.mat")
                mapping_paths.append(path)


        self.pool_matrices = self.load_pooling_matrices(mapping_paths)
        #
        self.graphs = self.create_graphs()

        self.num_workers = self.opt.args.num_workers
        self.dataset = GraphDataset(self.graphs, self.labels)
        print(f"DataLoader initialized with {self.num_nodes} subjects and {self.num_classes} classes.\n")


    def load_subjects_and_labels(self):


        group_folders  = {"NC": 0, "EMCI": 1}
        subject_ids = []
        group_dict = {}
        labels = []

        for group_name, label in group_folders.items():
            group_path = os.path.join(self.opt.args.netmatrix_schaefer_path, group_name, "NetMatrix_fisherz")

            for filename in os.listdir(group_path):
                if filename.endswith(".mat"):
                    subject_id = filename.split("_")[-1].split(".")[0]
                    subject_ids.append(subject_id)
                    group_dict[subject_id] = group_name
                    labels.append(label)
        labels = np.array(labels)

        return np.array(subject_ids), group_dict, labels



    def load_fc_matrix_for_subject(self, sub_id):

        folder = self.get_folder_for_subject(sub_id, self.group_dict)
        mat_file_path = os.path.join(folder, "NetMatrix_fisherz", f"NetworkMatrix_{sub_id}.mat")
        matrix = self.load_mat_matrix(mat_file_path)

        return matrix

    def load_fc_matrices(self):
        fc_data = []

        for sub_id in self.subject_ids:
            matrix = self.load_fc_matrix_for_subject(sub_id)
            if matrix is not None:
                fc_data.append(matrix)

        print(f"Loaded FC matrices for {len(fc_data)} subjects.")
        return np.array(fc_data)

    def load_pooling_mappings(self, mapping_paths):

        pooling_mappings = []
        for path in mapping_paths:
            mat_contents = sio.loadmat(path)
            pooling_mappings.append(mat_contents['mapping'])
        return pooling_mappings

    def load_pooling_matrices(self, mapping_paths):
        pool_matrices = self.load_pooling_mappings(mapping_paths)
        return pool_matrices


    def get_folder_for_subject(self, sub_id, group_dict):

        group = group_dict.get(sub_id, "UNKNOWN")
        valid_groups = ["EMCI", "NC"]
        if group in valid_groups:
            return os.path.join(self.opt.args.netmatrix_schaefer_path, group)


    def load_mat_matrix(self, mat_file_path):

        mat_data = sio.loadmat(mat_file_path)
        fc_matrix = mat_data.get('NetworkMatrix', None)

        return fc_matrix

    def create_graphs(self):
        k = self.k
        graphs = []

        for fc in self.fc_data:
            g = create_heterograph(fc, self.pool_matrices, k)
            graphs.append(g)

        return graphs

    def get_all_data(self):

        return self.graphs, self.labels



    def get_fold_batches(self, indices, batch_size, shuffle=False):


        fold_graphs = [self.graphs[i] for i in indices]
        fold_labels = [self.labels[i] for i in indices]
        fold_graphs = [graph.to(self.opt.args.device) for graph in fold_graphs]

        dataset = GraphDataset(fold_graphs, fold_labels)
        dataloader = TorchDataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=self.opt.args.num_workers,
                                     collate_fn=self.collate_fn
                                     )
        return dataloader


    def collate_fn(self, batch):

        graphs, labels = zip(*batch)
        batched_graph = dgl.batch(graphs)

        labels = torch.tensor(labels, dtype=torch.long, device=self.opt.args.device)

        return batched_graph, labels


class GraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.tensor(self.labels[idx], dtype=torch.long)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass