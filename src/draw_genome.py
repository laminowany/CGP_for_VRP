import os
import json
import time
import torch
import torch.optim as optim
import random

from legacy import GenomeFactory
from utils.graph import export_cgp_to_graphviz
from utils.process import get_options
from learning.attention_model import AttentionModel
from learning.cgp import CGP_Net
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.reinforce_baselines import RolloutBaseline, WarmupBaseline
from learning.problem_vrp import CVRP
from utils.training import evaluate 
from utils.logger import Logger

def initial_setup(opts):
    os.makedirs(opts.save_dir)
    reset_seeds(opts)
    
def reset_seeds(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    
def run(opts):
    initial_setup(opts)
    x_dim = 24
    outputs = [48]
    genome = [(5, 0), (3, 49, 1), (1, 50), (1, 50), (5, 50), (2, 1), (5, 1), (5, 4), (7, 26), (6, 2), (2, 49), (7, 25), (2, 51), (2, 57), (4, 12), (3, 29, -1), (4, 12), (2, 27), (7, 31), (5, 15), (1, 13), (2, 15), (6, 59), (6, 16), (3, 0, 0), (4, 1), (3, 1, -1), (1, 49), (5, 4), (4, 27), (5, (0, 6)), (5, 55), (5, 51), (5, 5), (4, 1), (2, 56), (7, 32), (3, 12, 1), (4, 10), (6, 52), (1, 32), (1, 40), (7, 13), (4, 9), (1, 65), (6, 39), (2, 18), (2, 1), (5, 0), (2, 0), (6, 2), (1, 26), (7, 0), (2, 1), (1, 3), (5, (26, 25, 28)), (4, 53), (2, 6), (1, 49), (6, 59), (4, 9), (6, 53), (4, 36), (5, 54), (1, 52), (5, 31), (4, 37), (4, 19), (3, 28, 0), (3, 33, -1), (5, (43, 32)), (6, 8)]
    model = AttentionModel(opts, CGP_Net(opts, genome, outputs))
    export_cgp_to_graphviz(model.get_encoder().genes, "childus")


if __name__ == "__main__":
    run(get_options())
 