from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Batch
from TGRankEncoder.TGEncoder import TSARNet
from torch_geometric.nn.aggr import Aggregation


class HCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers,feat_dim, str_dim, net):
        super(HCL, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder_net = net
        #self.edge_time_emb_dim = edge_time_emb_dim
        self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers)
        #.to(device)
        # self.encoder_feat = encode_net()
        # self.encoder_str = 
        #print(hidden_dim)
        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))
  
    
    def forward(self,x, x_s, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Pass through the encoder_net
        n_x, g_x = self.encoder_net(x, edge_index, edge_attr, batch)
        xpool = [global_add_pool(x, batch) for x in g_x]
        n_x = torch.cat(xpool, 1)
        g_x = torch.cat(g_x, 1)

        # Repeat for the second set of node features
        n_xs, g_xs = self.encoder_net(x_s, edge_index, edge_attr, batch)
        xpool = [global_add_pool(x, batch) for x in g_xs]
        n_xs = torch.cat(xpool, 1)
        g_xs = torch.cat(g_xs, 1)

        return n_x, g_x, n_xs, g_xs


    @staticmethod
    def calc_loss_n(x, x_aug, batch_size, temperature=0.2):

        batch_size, _ = x.size()



        positive_distance = nn.functional.pairwise_distance(x, x_aug)
        
        # Negative pairs: compute pairwise distance between all other combinations in the batch
        negative_distance = torch.cdist(x, x_aug, p=2)

        # Ensure the diagonal (positive pairs) is excluded from the negative pairs
        negative_distance = negative_distance + torch.eye(batch_size, device=x.device) * temperature
        
        # Select the minimum distance for negative pairs
        negative_distance, _ = torch.min(negative_distance, dim=1)
        
        # Contrastive loss computation
        loss_positive = torch.pow(positive_distance, 2)
        loss_negative = torch.pow(torch.clamp(temperature - negative_distance, min=0.0), 2)
        loss = torch.mean(loss_positive + loss_negative)


        return loss

        
    

    @staticmethod
    def calc_loss_g(x, x_aug, batch_size, temperature=0.2):
        batch_size, _ = x.size()

        positive_distance = nn.functional.pairwise_distance(x, x_aug)
        
        # Negative pairs: compute pairwise distance between all other combinations in the batch
        negative_distance = torch.cdist(x, x_aug, p=2)

        # Ensure the diagonal (positive pairs) is excluded from the negative pairs
        negative_distance = negative_distance + torch.eye(batch_size, device=x.device) * temperature
        
        # Select the minimum distance for negative pairs
        negative_distance, _ = torch.min(negative_distance, dim=1)
        
        # Contrastive loss computation
        loss_positive = torch.pow(positive_distance, 2)
        loss_negative = torch.pow(torch.clamp(temperature - negative_distance, min=0.0), 2)
        loss = torch.mean(loss_positive + loss_negative)

        return loss


class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)