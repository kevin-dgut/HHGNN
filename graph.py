import pickle
import sys

import torch
from scipy.spatial import distance

import dgl

import numpy as np

def create_heterograph(fcn_matrix, pool_matrices, k):

    num_nodes_per_sample = fcn_matrix.shape[0]

    node_ranges = {
        'scale1': range(0, 100),
        'scale2': range(100, 300),
        'scale3': range(300, 600),
        'scale4': range(600, 1000),
        'scale5': range(1000, 1500)
    }



    with open("pooling_map.pkl", "rb") as f:
        pooling_map = pickle.load(f)



    scale_levels = ['scale5', 'scale4', 'scale3', 'scale2','scale1']
    # overlapped Threshold
    OT = 0

    knn_neighbors = knn_select_neighbors(fcn_matrix, k)
    edge_types = []
    scales = list(node_ranges.keys())


    for scale in scales:
        edge_types.append((scale, f'intra_0{scale[-1]}{scale[-1]}', scale))

    for i in range(len(scales)):
        for j in range(i):
            scale_from, scale_to = scales[i], scales[j]
            edge_types.append((scale_from, f'inter_2{scale_from[-1]}{scale_to[-1]}', scale_to))


    for i in range(len(scales)):
        for j in range(i + 1, len(scales)):
            scale_from, scale_to = scales[i], scales[j]
            edge_types.append((scale_from, f'inter_1{scale_from[-1]}{scale_to[-1]}', scale_to))
            edge_types.append((scale_to, f'inter_1{scale_to[-1]}{scale_from[-1]}', scale_from))





    graph_data = {key: ([], []) for key in edge_types}

    for i in range(num_nodes_per_sample):
        k_neighbors = knn_neighbors[i]
        for j in k_neighbors:
            if i == j:
                continue
            src_type, src_local_id  = get_scale_type(i, node_ranges)
            dst_type, dst_local_id  = get_scale_type(j, node_ranges)
            if src_type == dst_type:
                etype = f'intra_0{src_type[-1]}{src_type[-1]}'
                graph_data[(src_type, etype, dst_type)][0].append(src_local_id)
                graph_data[(src_type, etype, dst_type)][1].append(dst_local_id)
                graph_data[(src_type, etype, dst_type)][0].append(dst_local_id)
                graph_data[(src_type, etype, dst_type)][1].append(src_local_id)

            for s1_idx in range(len(scale_levels)):
                for s2_idx in range(s1_idx + 1, len(scale_levels)):

                    scale_x, scale_y = scale_levels[s1_idx], scale_levels[s2_idx]
                    pool_idx = pooling_map[(scale_x, scale_y)]


                    if src_type ==scale_x and dst_type ==scale_y :

                        if pool_matrices[pool_idx][src_local_id,dst_local_id] >OT:
                            pool_etype = f'inter_2{src_type[-1]}{dst_type[-1]}'
                            graph_data[(scale_x, pool_etype, scale_y)][0].append(src_local_id)
                            graph_data[(scale_x, pool_etype, scale_y)][1].append(dst_local_id)

                        if pool_matrices[pool_idx][src_local_id,dst_local_id] == OT and fcn_matrix[i, j] > 0:

                            func_etype = f'inter_1{src_type[-1]}{dst_type[-1]}'
                            reverse_func_etype = f'inter_1{dst_type[-1]}{src_type[-1]}'

                            graph_data[(scale_x, func_etype, scale_y)][0].append(src_local_id)
                            graph_data[(scale_x, func_etype, scale_y)][1].append(dst_local_id)

                            graph_data[(scale_y, reverse_func_etype, scale_x)][0].append(dst_local_id)
                            graph_data[(scale_y, reverse_func_etype, scale_x)][1].append(src_local_id)


    g = dgl.heterograph(graph_data)



    scale_feats = {ntype: torch.zeros((len(node_ranges[ntype]), fcn_matrix.shape[0])) for ntype in node_ranges.keys()}
    for node_idx in range(num_nodes_per_sample):
        ntype, local_idx = get_scale_type(node_idx, node_ranges)
        node_features = fcn_matrix[node_idx, :]

        node_features = torch.from_numpy(np.nan_to_num(node_features, nan=0.0, posinf=1.0, neginf=-1.0)).float()
        scale_feats[ntype][local_idx, :] = node_features
        scale_feats[ntype][local_idx, :] = node_features


    for ntype, feats in scale_feats.items():
        g.nodes[ntype].data['feat'] = feats

    print(g)
    return g

def knn_select_neighbors(fcn_matrix, k):
    num_nodes = fcn_matrix.shape[0]
    neighbors = []
    dist_matrix = distance.cdist(fcn_matrix, fcn_matrix, metric='euclidean')
    for i in range(num_nodes):
        dist_matrix[i, i] = np.inf
        k_neighbors = np.argsort(dist_matrix[i])[:k]
        neighbors.append(k_neighbors)
    neighbors = np.array(neighbors)
    return neighbors


def get_scale_type(node_idx, node_ranges):
    for scale, node_range in node_ranges.items():
        if node_idx in node_range:
            return scale, node_idx - node_range.start