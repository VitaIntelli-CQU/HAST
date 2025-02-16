import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import HypergraphConv
from .preprocess import fix_seed

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 

def build_adj_hypergraph(adjacency_matrix, num_neighbors=3):
    n, m = adjacency_matrix.shape
    hypergraph_edges = []
    edge_weights = []

    for i in range(n):
        Neighbor_distances = adjacency_matrix[i, :]
        distances = Neighbor_distances

        _, nearest_neighbors = torch.topk(distances, k=num_neighbors + 1)

        for neighbor in nearest_neighbors:
            hypergraph_edges.append([i, neighbor])
            edge_weights.append(1)

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return hypergraph_edges

class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, graph_neigh_g, adji, adje, adjg, device, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.graph_neigh_g = graph_neigh_g
        self.dropout = dropout
        self.act = act
        self.device = device
        self.adji = adji
        self.adje = adje
        self.adjg = adjg

        self.hadje = build_adj_hypergraph(self.adje).to(self.device)
        self.hadjg = build_adj_hypergraph(self.adjg).to(self.device)
        self.hadji = build_adj_hypergraph(self.adji).to(self.device)

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.hyg1 = HypergraphConv(self.in_features, self.out_features)
        self.hyg2 = HypergraphConv(self.in_features, self.out_features)
        self.hyg3 = HypergraphConv(self.in_features, self.out_features)
        self.hyg4 = HypergraphConv(self.out_features, self.out_features)
        self.w_a = torch.FloatTensor([1/3., 1/3., 1/3.]).to(self.device)
        self.adj_fused = None
        self.linear = nn.Sequential(
            nn.Linear(self.out_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.in_features)
        )

        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)

        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def adaptive_weight_fusion(self, adji, adje, adjg):
        if self.adj_fused is None:
            self.adj_fused = (adji + adje + adjg) / 3.0 
        
        adj_list = [adji, adje, adjg]
        for v in range(3):
            diff = self.adj_fused - adj_list[v]
            self.w_a[v] = 1 / (torch.norm(diff, p='fro') + 1e-8)
        
        device = self.w_a.device
        
        numerator = torch.zeros((self.adj_fused.shape[0], self.adj_fused.shape[0])).to(device)
        denominator = torch.sum(self.w_a)
        # print(self.w_a.device, adji.device, adje.device, adjg.device)
        # print(numerator.device, denominator.device)
        for v in range(3):
            numerator += self.w_a[v] * adj_list[v]
        self.adj_fused = numerator / denominator
        return self.adj_fused
    
    def forward(self, feat, feat_a):
        z = F.dropout(feat, self.dropout, self.training)
        z0 = torch.mm(z, self.weight1)
        self.adj_fused = self.adaptive_weight_fusion(self.adji, self.adje, self.adjg).to(self.device)
        # z = torch.mm(adj, z)
        hadj_fused = build_adj_hypergraph(self.adj_fused, num_neighbors=4).to(self.device)
        
        z1 = self.hyg1(z, self.hadje)
        z2 = self.hyg2(z, self.hadjg)
        z3 = self.hyg3(z, self.hadji)

        z = z1 + z2 + z3 + z0
        emb = self.act(z)
        hiden_emb = z
        h = self.linear(z)

  
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z0_a = torch.mm(z_a, self.weight1)
        z1_a = self.hyg1(z_a, self.hadje)
        z2_a = self.hyg2(z_a, self.hadjg)
        z3_a = self.hyg3(z_a, self.hadji)

        z_a = z1_a + z2_a + z3_a + z0_a
        emb_a = self.act(z_a)

        g = self.hyg4(z, hadj_fused)
        g = self.sigm(g)  

        g_a = self.hyg4(z_a, hadj_fused)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a

