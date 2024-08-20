import math
import time
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from data.tgEncoder_data import init_structural_encoding



#from models.loss_functions import recon_loss
from utils.sampler import temporal_sampling
import torch.nn.functional as F

def encoder_train_per_iter(net, optimizer, train_data, full_data, node_features, edge_features, train_neighbor_finder, 
               batch_size, num_temporal_hops, n_neighbors, verbose=False, coalesce_edges_and_time=False, train_randomize_timestamps=False):
			num_instance = len(train_data["sources"])
			num_batch = math.ceil(num_instance / batch_size)

			net.train()
			batch_loss = []
			train_start_time = time.time()
			for batch_idx in (tqdm(range(0, num_batch)) if verbose else range(0, num_batch)):
				optimizer.zero_grad()
				start_idx = batch_idx * batch_size
				end_idx = min(num_instance, start_idx + batch_size)
				sources_batch, destinations_batch = train_data["u"][start_idx:end_idx], \
													train_data["i"][start_idx:end_idx]
				edge_idxs_batch = train_data["idx"][start_idx: end_idx]
				timestamps_batch = train_data["ts"][start_idx:end_idx]
				size = len(sources_batch)
				#_, negatives_batch = train_rand_sampler.sample(size)

				n_unique_nodes = len(node_features)

				# sample positive subgraphs
				# enclosing_subgraphs_pos = temporal_sampling(sources_batch, destinations_batch,
				# 											timestamps_batch, train_neighbor_finder, train_data,
				# 											node_features, edge_features, n_unique_nodes,
				# 											num_temporal_hops, n_neighbors,coalesce_edges_and_time, train_randomize_timestamps)

				# enclosing_subgraphs_neg = temporal_sampling(sources_batch, negatives_batch, timestamps_batch,
		        #                                     train_neighbor_finder, train_data,
		        #                                     node_features, edge_features, n_unique_nodes,
		        #                                     num_temporal_hops, n_neighbors)
				# enclosing_subgraphs_pos = init_structural_encoding(enclosing_subgraphs_pos)
				# enclosing_subgraphs_neg = init_structural_encoding(enclosing_subgraphs_neg)
				batch_pos = Batch.from_data_list(enclosing_subgraphs_pos)
				pos_h, pos_hs = net(batch_pos)
				pos_g_h = torch.cat(pos_hs,1) 
				# prob = net.predict_proba(h)
				# prob_g = net.predict_proba(pos_g_h)
				# splits = torch.tensor_split(prob, batch_pos.ptr)
				# loss = []
		
				# for sp in splits[1:-1]:
				# 	y = torch.zeros(sp.shape[0], device=net.device)
				# 	# make destination as label 1
				# 	y[1] = 1.0

				# 	loss.append(-torch.sum(y * F.log_softmax(sp, dim=0)))

				# loss = sum(loss) / len(loss)
				# loss.backward()
				# optimizer.step()
				return pos_h, pos_g_h