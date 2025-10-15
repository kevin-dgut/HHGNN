import torch
import sys


import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv,HeteroGraphConv



from torch import nn

from dgl import function as fn


from HHGNN.opt import parase_opt





class HGConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, device):
        super(HGConv, self).__init__()


        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.device = device

        self.mgcn = GraphConv(in_channels, hidden_channels)

        self.linear_proj1 = torch.nn.Linear(hidden_channels, in_channels).to(self.device)

        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)
        self.leakyrule = torch.nn.LeakyReLU()


        self.reset_parameters()

    def reset_parameters(self):

        torch.nn.init.xavier_uniform_(self.linear_proj1.weight)

    def forward(self, g, edge_types, src_scale, dst_scale, apply_mapping):
        hsg = dgl.edge_type_subgraph(g, edge_types).to(self.device)

        H_init = hsg.ndata['feat'][dst_scale].to(self.device)

        mod1 = {etype: self.mgcn for etype in edge_types}
        conv1 = HeteroGraphConv(mod1, aggregate='sum').to(self.device)

        hsg_feat = hsg.ndata['new_feat'][src_scale].to(self.device)
        if apply_mapping:
            feat = hsg.ndata['new_feat'][src_scale].to(self.device)
            feat_mapped = self.linear_proj1(feat)
            hsg_feat = {src_scale: feat_mapped, dst_scale: H_init}

        H = conv1(hsg, hsg_feat)[dst_scale]
        H = self.batch_norm(H)
        H = self.leakyrule(H)
        return H

class pool(nn.Module):
    def __init__(self, in_channels,device):
        super(pool, self).__init__()

        self.in_channels = in_channels
        self.device = device

    def forward(self, g, scale_from, scale_to, pool_edge_type):
        hsg = dgl.edge_type_subgraph(g, [pool_edge_type]).to(self.device)
        hsg.nodes[scale_from].data['pool_feat'] = torch.zeros(hsg.num_nodes(scale_from), self.in_channels).to(g.device)
        hsg.nodes[scale_from].data['pool_feat'] = g.ndata['pool_feat'][scale_from]
        hsg.nodes[scale_to].data['pool_feat'] = torch.zeros(
            hsg.num_nodes(scale_to),
            hsg.ndata['pool_feat'][scale_from].shape[1]
        ).to(g.device)
        with hsg.local_scope():
            hsg.update_all(
                fn.copy_u('pool_feat', 'm'),
                fn.sum('m', 'pool_feat')
            )


            g.nodes[scale_to].data['pool_feat'] = hsg.nodes[scale_to].data['pool_feat']

        return hsg.nodes[scale_to].data['pool_feat']

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)

class GraphFusionModule(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super(GraphFusionModule, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.query_fc = nn.Linear(hidden_channels, hidden_channels)
        self.key_fc = nn.Linear(hidden_channels, hidden_channels)
        self.value_fc = nn.Linear(hidden_channels, hidden_channels)
        self.head_dim = int(hidden_channels / num_heads)
        self.scale_fc = nn.Linear(hidden_channels, hidden_channels)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)


    def forward(self, H):


        batch_size,num_scales,_ = H.shape

        Q = self.query_fc(H)
        K = self.key_fc(H)
        V = self.value_fc(H)



        Q = Q.view(batch_size, num_scales, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, num_scales, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, num_scales, self.num_heads, self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        heads = torch.matmul(attention_weights, V)

        heads = heads.transpose(1,2).contiguous().view(batch_size, num_scales, self.hidden_channels)

        H_scale = self.scale_fc(heads)

        H_target = H_scale.sum(dim=1)


        return H_target


class HierarchicalLearningModule(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels):
        super(HierarchicalLearningModule,self).__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.device = parase_opt().args.device
        self.mgcn=GCN(in_channels,hidden_channels)
        self.hgconv = HGConv(in_channels, hidden_channels,self.device)
        self.pool = pool(in_channels,self.device)




        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.linear_proj = nn.Linear(in_channels, hidden_channels).to(self.device)
        self.linear_proj1 = nn.Linear(self.hidden_channels, self.in_channels).to(self.device)
        self.fusion_weights = nn.Parameter(torch.randn(4, 2))

        self.mlp_transform = nn.Sequential(
            nn.Linear((in_channels+hidden_channels), hidden_channels*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels*2, hidden_channels),
        ).to(self.device)
        self.leakyrule=nn.LeakyReLU()

    def get_inter_scale_features(self, g, node_type, higher_scales):
        inter_feats = []
        for hs in higher_scales:
            src_id, tgt_id = hs[-1], node_type[-1]
            non_overlap = self.hgconv(
                g,
                [f'inter_1{src_id}{tgt_id}', f'inter_1{tgt_id}{src_id}'],
                hs, node_type, apply_mapping=True
            )
            overlap = self.pool(g, hs, node_type, f'inter_2{src_id}{tgt_id}')
            inter_feats.append(torch.concat([non_overlap, overlap], dim=-1))
        return sum(inter_feats)


    def forward(self, g, node_type):

        g = g.to(self.device)
        hsg = dgl.edge_type_subgraph(g, [f'intra_0{node_type[-1]}{node_type[-1]}']).to(self.device)
        feat = hsg.ndata['feat'].to(self.device)


        H_intra = self.mgcn(hsg, feat)
        g.nodes[node_type].data['pool_feat'] = feat


        if node_type == 'scale5':
            g.nodes[node_type].data['new_feat'] = H_intra


            return g


        scale_hierarchy = {
            'scale4': ['scale5'],
            'scale3': ['scale5', 'scale4'],
            'scale2': ['scale5', 'scale4', 'scale3'],
            'scale1': ['scale5', 'scale4', 'scale3', 'scale2']
        }

        higher_scales = scale_hierarchy[node_type]


        H_inter = self.get_inter_scale_features(g, node_type, higher_scales)
        H_inter = self.mlp_transform(H_inter)


        idx = {'scale4': 0, 'scale3': 1, 'scale2': 2, 'scale1': 3}[node_type]
        weights = F.softmax(self.fusion_weights[idx], dim=0)
        weight_intra, weight_inter = weights[0], weights[1]
        H_final = weight_intra * H_intra + weight_inter * H_inter


        g.nodes[node_type].data['new_feat'] = H_final.to(self.device)
        return g



class HeteroGNN(nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels, num_classes):
        super(HeteroGNN, self).__init__()


        self.HierarchicalLearningModule = HierarchicalLearningModule(in_channels,hidden_channels, out_channels)
        self.attention=GraphFusionModule(hidden_channels)
        self.device = parase_opt().args.device
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes)
        )



    def forward(self, g):
        scales = ['scale5', 'scale4', 'scale3', 'scale2', 'scale1']
        for scale in scales:
            self.HierarchicalLearningModule(g, scale)  #

        aggregated_feats = [
            dgl.sum_nodes(g, feat='new_feat', ntype=scale).to(self.device)
            for scale in scales
        ]

        H = torch.stack(aggregated_feats, dim=1)

        H_graph = self.attention(H)

        out = self.fc(H_graph)
        return out

class GCN(torch.nn.Module):
    def __init__(self, input_channels,hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.leakyrule = torch.nn.LeakyReLU()


        self.device = parase_opt().args.device
    def forward(self, g,feat):
        g = dgl.add_self_loop(g)
        x = self.conv1(g, feat.to(self.device))
        x = self.batch_norm1(x)
        x = self.leakyrule(x)

        return x