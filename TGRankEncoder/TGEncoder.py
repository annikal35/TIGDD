
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from TGRankEncoder.TimeEncode import TimeEncoder
from TGRankEncoder.MLP import MLP
from torch_geometric.nn import GINConv, global_add_pool, GCNConv, global_mean_pool
import numpy as np
import os

class TSARLayer(torch.nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_size, reduce):
        super(TSARLayer, self).__init__()
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.edge_time_emb_size = edge_time_emb_size
        self.reduce = reduce
        self.msg_function = torch.nn.Linear((self.emb_dim + self.edge_attr_size + self.edge_time_emb_size),
                                            self.emb_dim, bias=True)
        self.linear = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        self.layer_norm = torch.nn.LayerNorm(self.emb_dim, elementwise_affine=True, eps=1e-5)
        self.layer_relation_embedding = torch.nn.Parameter(
            torch.rand(self.emb_dim), requires_grad=True)

    def forward(self, node_feature_view, augmented_view, edge_index, edge_attr, edge_time_emb, boundary_condition):
        # node_feature_view: [num_nodes, emb_dim]
        # augmented_view: [num_augmented_nodes, emb_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_attr_size]
        # edge_time_emb: [num_edges, edge_time_emb_size] (if applicable)
        # boundary_condition: [num_nodes, emb_dim]

        num_nodes = node_feature_view.size(0)
        num_augmented_nodes = augmented_view.size(0)
        
        assert node_feature_view.shape == boundary_condition.shape

        # Message construction
        msg = node_feature_view[edge_index[0]]
        if edge_time_emb is not None:
            msg = torch.cat((msg, edge_attr, edge_time_emb), dim=1)
        else:
            msg = torch.cat((msg, edge_attr), dim=1)

        msg = F.relu(self.msg_function(msg))

        # Concatenate message with boundary condition
        msg_aug = torch.cat([msg, boundary_condition])

        # Adding self-loop to augmented nodes
        self_loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        self_loop_index = self_loop_index.unsqueeze(0).repeat(2, 1)
        idx_aug = torch.cat([edge_index[1], self_loop_index[1]])

        # Aggregating messages
        out = scatter(msg_aug, idx_aug, dim=0, reduce=self.reduce, dim_size=num_augmented_nodes)

        # Pass through linear layer and normalization
        out = self.linear(out)
        out = F.dropout(F.relu(self.layer_norm(out)), p=0.1, training=self.training)

        return out


# class TSARLayer(torch.nn.Module):
# 	def __init__(self, emb_dim, edge_attr_size, edge_time_emb_size, reduce):
# 		super(TSARLayer, self).__init__()
# 		self.emb_dim = emb_dim
# 		self.edge_attr_size = edge_attr_size
# 		self.edge_time_emb_size = edge_time_emb_size
# 		self.reduce = reduce
# 		self.msg_function = torch.nn.Linear((self.emb_dim + self.edge_attr_size + self.edge_time_emb_size),
# 		                                    self.emb_dim, bias=True)
# 		self.linear = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=True)
# 		self.layer_norm = torch.nn.LayerNorm(self.emb_dim, elementwise_affine=True, eps=1e-5)
# 		self.layer_relation_embedding = torch.nn.Parameter(
# 			torch.rand(self.emb_dim), requires_grad=True)

# 	def forward(self, hidden, edge_index, edge_attr, edge_time_emb, boundary_condition):
# 		# hidden of shape [num_nodes, emb_dim]
# 		# edge_index of shape [2, num_edges]
# 		# boundary_condition of shape [num_nodes, emb_dim]

# 		num_nodes = hidden.size(0)
# 		# print("nodes:", num_nodes)
# 		# print("hidden:", hidden.shape)
# 		# print("edge 0:", edge_index[0])
# 		assert hidden.shape == boundary_condition.shape

# 		# msg = (hidden[edge_index[0]] + self.la,-1yer_relation_embedding)
# 		msg = (hidden[edge_index[0]])
# 		# msg has shape [num_edges, emb_dim]
# 		if edge_time_emb is not None:
# 			# print("msg:", msg.shape)
# 			# print("edge_attr:", edge_attr.shape)
# 			# print("edge_time_emb:",edge_time_emb.shape)
# 			# exit()
# 			msg = torch.cat((msg, edge_attr, edge_time_emb), dim=1)
# 		else:
# 			msg = torch.cat((msg, edge_attr), dim=1)
# 		# msg has shape [num_edges, emb_dim+edge_attr_size+edge_time_emb_size]
# 		msg = F.relu(self.msg_function(msg))

# 		msg_aug = torch.cat([msg, boundary_condition])

# 		self_loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
# 		self_loop_index = self_loop_index.unsqueeze(0).repeat(2, 1)
# 		idx_aug = torch.cat([edge_index[1], self_loop_index[1]])

# 		out = scatter(msg_aug, idx_aug, dim=0, reduce=self.reduce, dim_size=hidden.size(0))

# 		out = self.linear(out)
# 		out = F.dropout(F.relu(self.layer_norm(out)), p=0.1, training=True)

# 		# out has shape [num_nodes, out_dim]

# 		return out

class TSARNet(torch.nn.Module):
    def __init__(self, emb_dim, edge_attr_size, edge_time_emb_dim=128, num_layers=3, use_fourier_features=True, use_id_label=True, device=None):
        super(TSARNet, self).__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.edge_attr_size = edge_attr_size
        self.num_layers = num_layers
        self.use_fourier_features = use_fourier_features
        self.use_id_label = use_id_label
        print("bool using_label_diffusion:", self.use_id_label)
        self.nbf_layers = torch.nn.ModuleList()

        if edge_time_emb_dim > 0:
            self.edge_time_emb_dim = edge_time_emb_dim
            self.time_encoder = TimeEncoder(dimension=self.edge_time_emb_dim, use_fourier_features=self.use_fourier_features)
        else:
            self.edge_time_emb_dim = 0
            self.time_encoder = None

        self.indicator_embedding = torch.nn.Parameter(
            torch.rand(1, emb_dim), requires_grad=True)

        for layer in range(num_layers):
            self.nbf_layers.append(TSARLayer(self.emb_dim, self.edge_attr_size, self.edge_time_emb_dim, reduce="sum"))

        self.mlp = MLP(emb_dim)

    def forward(self, node_feature_view, augmented_view, edge_index, edge_attr, boundary_condition, edge_time=None):
        hs = []
        batch_bc = torch.zeros(augmented_view.size(0), self.emb_dim, device=self.device)

        if edge_index.nelement() == 0 and edge_attr.nelement() == 0:
            return batch_bc
        
        if self.edge_time_emb_dim > 0 and edge_time is not None:
            edge_time_embeddings = self.time_encoder(edge_time.to(self.device))
        else:
            edge_time_embeddings = None

        h = augmented_view  # Start with the augmented view as the initial hidden state

        for layer in range(self.num_layers):
            h = self.nbf_layers[layer](node_feature_view, h, edge_index.to(self.device), edge_attr.to(self.device),
                                       edge_time_embeddings, boundary_condition)
            hs.append(h)
        
        return h, hs

# class TSARNet(torch.nn.Module):
# 	def __init__(self, emb_dim, edge_attr_size, edge_time_emb_dim=128, num_layers=3, use_fourier_features=True, use_id_label=True, device=None):
# 		super(TSARNet, self).__init__()
# 		self.device = device
# 		self.emb_dim = emb_dim
# 		self.edge_attr_size = edge_attr_size
# 		self.num_layers = num_layers
# 		self.use_fourier_features = use_fourier_features
# 		self.use_id_label = use_id_label
# 		print("bool using_label_diffusion:", self.use_id_label)
# 		self.nbf_layers = torch.nn.ModuleList()

# 		if edge_time_emb_dim > 0:
# 			self.edge_time_emb_dim = edge_time_emb_dim
# 			self.time_encoder = TimeEncoder(dimension=self.edge_time_emb_dim, use_fourier_features=self.use_fourier_features)
# 		else:
# 			self.edge_time_emb_dim = 0
# 			self.time_encoder = None

# 		self.indicator_embedding = torch.nn.Parameter(
# 			torch.rand(1, emb_dim), requires_grad=True)

# 		for layer in range(num_layers):
# 			self.nbf_layers.append(TSARLayer(self.emb_dim, self.edge_attr_size, self.edge_time_emb_dim, reduce="sum"))

# 		self.mlp = MLP(emb_dim)

# 	def forward(self, batch):
# 		hs = []
# 		#batch_sources_idx = batch.source + batch.ptr[:-1]

# 		batch_bc = torch.zeros(batch.num_nodes, self.emb_dim, device=self.device)

# 		# if self.use_id_label:
# 		# 	# this makes sure source gets different label from that of all destinations
# 		# 	batch_bc[batch_sources_idx] = self.indicator_embedding

# 		if batch.edge_index.nelement() == 0 and batch.edge_attr.nelement() == 0:
# 			return batch_bc
# 		if self.edge_time_emb_dim > 0:
# 			#print("edge time:", batch.time.shape)
# 			edge_time_embeddings = self.time_encoder(batch.time.to(self.device))
# 		else:
# 			edge_time_embeddings = None

# 		for layer in range(self.num_layers):
# 			if layer == 0:
# 				#print(batch.edge_attr.shape)
# 				h = self.nbf_layers[layer](batch_bc, batch.edge_index.to(self.device), batch.edge_attr.to(self.device),
# 				                           edge_time_embeddings, batch_bc)
# 			else:
# 				h = self.nbf_layers[layer](h, batch.edge_index.to(self.device), batch.edge_attr.to(self.device),
# 				                           edge_time_embeddings, batch_bc)
# 			hs.append(h)
# 		return h, hs

# 	def predict_proba(self, edge_repr):
# 		prob = self.mlp(edge_repr)
# 		return prob.squeeze()