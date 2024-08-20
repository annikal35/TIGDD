from model.dsdd import DriftDetector
from train.properties import Experiment
from train.properties import RULSIF
from model.HCL import *
from utils.sampler import get_neighbor_finder
import torch.nn as nn
import argparse
from torch_geometric.loader import DataLoader
from utils.sampler import csv_to_pyg_elec
from utils.sampler import split_train_val_test
from torch_geometric.data import Batch
from utils.sampler import alter_label_ratio
from utils.sampler import alter_val_label_ratio
import os
from tqdm import tqdm
import time

import os

def arg_parse():
    parser = argparse.ArgumentParser('TGRank Encoder Training')
    parser.add_argument('--data', type=str, default='wikipedia',
                        help='Dataset name (eg. reddit, wikipedia, mooc, lastfm, enron, uci)')
    parser.add_argument('--prefix', type=str, default='tgrank-listwise',
                        help='Prefix to name the checkpoints and models')
    parser.add_argument('--train_batch_size', default=15,
                        type=int, help="Train batch size")
    parser.add_argument('--eval_batch_size', default=256, type=int,
                        help="Evaluation batch size (should experiment to make it as big as possible (based on available GPU memory))")
    parser.add_argument('--num_epochs', default=25, type=int,
                        help="Number of training epochs")
    parser.add_argument('--num_layer', default=5, type=int, help="Number of layers")
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('--time_dim', type=int, default=128,
                        help="Time Embedding dimension size. Give 0 if no time encoding is not to be used")
    parser.add_argument('--num_temporal_hops', type=int, default=3,
                        help="No. of temporal hops for sampling candidates during training.")
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help="No. of neighbors to sample for each candidate node at each temporal hop. This is also the same parameter that samples edges.")
    parser.add_argument('--uniform_sampling', action='store_true',
                        help='Whether to use uniform sampling for temporal neighbors. Default is most recent sampling.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Patience for early stopping')
    parser.add_argument('--log_dir', type=str, default="logs/",
                        help="directory for storing logs.")
    parser.add_argument('--saved_models_dir', type=str, default="outputs/saved_models/",
                        help="directory for saved models.")
    parser.add_argument('--saved_checkpoints_dir', type=str, default="outputs/saved_checkpoints/",
                        help="directory for saved checkpoints.")
    parser.add_argument('--verbose', type=int, default=0, help="Verbosity 0/1 for tqdm")
    parser.add_argument('--seed', type=int, default=0, help="deterministic seed for training. this is different from that by used neighbor finder which uses a local random state")
    parser.add_argument('--num_temporal_hops_eval', type=int, default=3,
                        help="No. of temporal hops for sampling candidates during evaluation.")
    parser.add_argument('--num_neighbors_eval', type=int, default=20,
                        help="No. of neighbors to sample for each candidate node at each temporal hop during evaluation. This is also the same parameter that samples edges.")
    parser.add_argument('--no_fourier_time_encoding', action='store_true',
                        help='Whether to not use fourier time encoding')
    parser.add_argument('--coalesce_edges_and_time', action='store_true',
                        help='Whether to coalesce edges and time. make sure no_fourier_time_encoding is set and time_dim is 1. else will raise error')
    parser.add_argument('--train_randomize_timestamps', action='store_true',
                        help='Whether to randomize train timestamps i.e. after sampling  and before going into TSAR')
    parser.add_argument('--no_id_label', action='store_true',
                        help='Whether to not use identity label to distinguish source from destinations. Value used to set label diffusion')
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-eval_freq', type=int, default=10)                    
    args = parser.parse_args()
    return args


def main():
    results = {}
    #ds = Dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()
    prev_loss = []
    param_w = Experiment.param_w
    print("\n\n----- Experimenting on Dataset[ %s ]----"%("Dataset Name"))
    graph_df, edge_feat, node_feat, n_total_unique_nodes = to_pd()
    data_list = pd_to_list(graph_df,node_feat,edge_feat,n_total_unique_nodes)
    train_list, val_list, test_list, train_df, val_df, test_df = split_list(data_list,graph_df)


    net = TSARNet(emb_dim=args.emb_dim, edge_attr_size=edge_feat.shape[1], device = device).to(device)
    model = HCL(args.hidden_dim, args.num_layer, len(node_feat), args.dg_dim+args.rw_dim,net).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    train_dataloader = DataLoader(train_list, batch_size=args.train_batch_size, shuffle=True)



    test_detected_points = 0
    train_corr_t_pt = 0
    train_corr_cnt = 0
    train_fa_pt = 0
    train_time_diff = float("inf")
    best_train_points = 0
    best_train_t = 0
    train_t_list = []
    run_time = float("inf")
    best_dod = float("inf")
    best_fa1000 = float("inf")
    best_dcr = 0
    dcr_list = []

    print("\n\n----- Start Iterating----")
   #exit()
    for j in range(1, Experiment.iterations+1):
        train_detected_points = []
        loss_all = 0
        a = 1
        t = 0
        loss_g_all, loss_n_all = [], []
        PE = []
        t_list = []
        val_t_list= []
        print("\n----- Iteration [ ", j," ] -------\n")
        if j <= 2 or (std_g == 0 and std_n == 0):
            weight_g, weight_n = 1, 1
        else:
            print(mean_g, mean_n)
            weight_g, weight_n = std_g ** args.alpha, std_n ** args.alpha
            weight_sum = (weight_g  + weight_n) / 2
            print("weight sum: ", weight_sum)
            weight_g, weight_n = weight_g/weight_sum, weight_n/weight_sum
       
        print("-----train---------")
        for data in tqdm(train_dataloader):
            # time.sleep(0.01)
            data = data.to(device)
            optimizer.zero_grad()
            n_x, g_x, n_xs, g_xs = model(data.x, data.x_s, data)
            loss_g = model.calc_loss_g(g_x, g_xs, args.train_batch_size)
            #exit()
            loss_n = model.calc_loss_n(n_x, n_xs, args.train_batch_size)
            loss =  weight_g * loss_g.mean() + weight_n * loss_n.mean()
            loss_g = weight_g * loss_g.mean()
            loss_n = weight_n * loss_n.mean()

            loss_g_all = loss_g_all + [loss_g.detach().cpu()]
            loss_n_all = loss_n_all + [loss_n.detach().cpu()]

            loss_all += loss.item() * len(train_list)

            loss.backward()
            optimizer.step()

            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)


        loss_g_all, loss_n_all = [], []
        PE = []
        t_list = []
        val_t_list= []
        model.eval()
        window_cnt =0
        for i,data in tqdm(enumerate(val_list)):
            # time.sleep(0.01)
            data = data.to(device)
            optimizer.zero_grad()
            n_x, g_x, n_xs, g_xs = model(data.x, data.x_s, data)
            loss_g = model.calc_loss_g(g_x, g_xs, 1)
            # exit()
            loss_n = model.calc_loss_n(n_x, n_xs, data.batch, 1)
            loss =  weight_g * loss_g.mean() + weight_n * loss_n.mean()
            loss_g = weight_g * loss_g.mean()
            loss_n = weight_n * loss_n.mean()
            loss_g_all = loss_g_all + [loss_g.detach().cpu()]
            loss_n_all = loss_n_all + [loss_n.detach().cpu()]

            loss_all += loss.item() * len(train_list)


            mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
            mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)


            t += 1
            window_cnt +=1
            if window_cnt >= param_w:
                if window_cnt == param_w:
                    y_score = (loss_g - mean_g)/std_g  + (loss_n - mean_n)/std_n
                    PE.append(y_score)
                    th = DriftDetector.set_dynamic_threshold(PE, param_w)
                    if y_score >= th:
                        print("Drift Detected at: ", t)
                        t_list.append(t)
                window_cnt = 0

        train_t_list.append(t_list)


    print(train_t_list)

main()
