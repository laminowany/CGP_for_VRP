import os
import json
import time
import torch
import torch.optim as optim
import random

from legacy import GenomeFactory
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
    
def produce_reference(row, times):
    idx = (row - 1) * 8 * times
    res = []
    init = 0
    for t in range(times):
        res.append((4, init))
        idx += 1
        res.append((5, (init, idx)))
        idx += 1
        res.append((2, idx))
        idx += 1
        res.append((3, idx, 1))
        idx += 1
        res.append((7, idx))
        idx += 1
        res.append((3, idx, -1))
        idx += 1
        res.append((5, (idx - 3, idx)))
        idx += 1
        res.append((2, idx))
        idx += 1
        init = idx
    return res    
    

def verify_sanity(opts, logger: Logger):
    opts.n_epochs = 5
    opts.epoch_size = 12800

    reset_seeds(opts)
    baseline = GenomeFactory().produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3)
    my_encoder = baseline.build_nn(opts)
    model1 = AttentionModel(opts, my_encoder)
    score_orig_encoder = evaluate(opts, model1, logger, osobnik_id=0)
    
    # reset_seeds(opts)
    # original_encoder = GraphAttentionEncoder(
    #     n_heads=opts.n_heads,
    #     embed_dim=opts.embedding_dim,
    #     n_layers=opts.n_encode_layers,
    #     normalization=opts.normalization
    # )
    # model2 = AttentionModel(opts, original_encoder)
    # score_genetic_encoder = evaluate(opts, model2, logger, osobnik_id=1)
    
    reset_seeds(opts)
    baseline = produce_reference(2, 3)
    x_dim = len(baseline)
    genome = [*[None]*x_dim,
             *baseline,
             *[None]*x_dim,]
    outputs = [48]
    encoderCGP = CGP_Net(opts.embedding_dim, x_dim, 3, outputs, genome=genome)
    model3 = AttentionModel(opts, encoderCGP)
    score_cgp = evaluate(opts, model3, logger, osobnik_id=2)
    
    # if score_orig_encoder != score_genetic_encoder:
    #     print("CARAMBA!")
    # else:
    #     print("All good boss")


def run(opts):
    initial_setup(opts)
    logger = Logger(opts)
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    verify_sanity(opts, logger)
    return

    opts.reproducible_seed = False
    #opts.graph_size = 50
    #opts.n_epochs = 5
    #opts.epoch_size = 128000
    #baseline = [ (4,0), (5,(0,9),), (2,10), (3,11,1), (7,12), (3,13,-1), (5,(10, 14),), (2,15)]*3
    baseline = produce_reference(2, 3)
    x_dim = len(baseline)
    genome = [*[None]*x_dim,
             *baseline,
             *[None]*x_dim,]
    outputs = [48]
    encoder = CGP_Net(opts.embedding_dim, x_dim, 3, outputs, genome=genome)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    model = AttentionModel(opts, original_encoder)
    
    #opts.n_epochs = 5
    #model.load_weights('/home/piotr/repos/magisterka/outputs/run_20260507T142017/snapshot_osobnik0_epoch5.pth')
    #evaluate(opts, model, logger, 0,)
    evaluate(opts, model, logger, 0, snapshots_epochs=list(range(5, 101, 5)))


if __name__ == "__main__":
    run(get_options())
 