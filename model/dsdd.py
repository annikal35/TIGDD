# ******************************************************************************
# drift_detector.py
#
# Drift Detector, Detect drift point in Graph Stream
#
#
# Date      Name       Description
# ========  =========  ========================================================
# 03/20/2018  Paudel     Initial version,
# ******************************************************************************
#

#from pylab import *

import os
import networkx as nx
import math
from random import shuffle
#from rulsif.change_detection import ChangeDetection
from train.properties import RULSIF, Experiment, GBAD
from model.HCL import HCL
from TGRankEncoder.TGEncoder import TSARNet
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser('TGRank Interaction Ranking Listwise Training')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Dataset directory')
    parser.add_argument('--data', type=str, default='wikipedia',
                        help='Dataset name (eg. reddit, wikipedia, mooc, lastfm, enron, uci)')
    parser.add_argument('--prefix', type=str, default='tgrank-listwise',
                        help='Prefix to name the checkpoints and models')
    parser.add_argument('--train_batch_size', default=64,
                        type=int, help="Train batch size")
    parser.add_argument('--eval_batch_size', default=256, type=int,
                        help="Evaluation batch size (should experiment to make it as big as possible (based on available GPU memory))")
    parser.add_argument('--num_epochs', default=25, type=int,
                        help="Number of training epochs")
    parser.add_argument('--num_layers', default=3, type=int, help="Number of layers")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--emb_dim', type=int, default=128,
                        help="Embedding dimension size")
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
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

class DriftDetector:
    is_isomorphic = False
    subgraph_id = 1
    S_w = {}

    def __init__(self):
        # remove graph file if exist
        #print("Subgraph_id: ", self.subgraph_id)
        #print("S_w: ", self.S_w)
        self.S_w.clear()
        self.is_isomorphic = False
        self.subgraph_id = 1
        #print("\n\n\nClear S_w: ", self.S_w, self.subgraph_id, self.is_isomorphic)
        self.subgraph_id = 1
        print("Starting Drift Detection-----")


    @staticmethod
    def set_dynamic_threshold(PE, param_w):
        if len(PE) >= param_w:
            pe_w =PE[len(PE) - param_w:len(PE)]
            pe_ww= [pe.detach().cpu().numpy() for pe in pe_w]
            pe_www = np.vstack(pe_ww)
            mean = np.mean(pe_www)
            std = np.std(pe_www)
            return (mean + std)
        else:
            return RULSIF.th
        


