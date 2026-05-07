import os
import json
import time
import torch
import torch.optim as optim
import random

from utils.process import get_options
from learning.attention_model import AttentionModel
from learning.cgp import GenomeFactory, CGP_Net
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.reinforce_baselines import RolloutBaseline, WarmupBaseline
from learning.problem_vrp import CVRP
from utils.training import evaluate 
from utils.logger import Logger

def cgp(opts, logger: Logger, parent=None, generations=None):
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    opts.n_epochs = 5
    opts.epoch_size = 128000
    factory = GenomeFactory()
    lambda_ = 4
    osobnik_id = 0
    if not parent:
        parent = factory.produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3) # baseline
    # parent = factory.get_random_genome(25, 0)
    parent.score = evaluate(opts, parent, logger=logger, osobnik_id=osobnik_id)
    print(f'genome:  {parent.genes}')
    print(f"gen -1 best score: {parent.score}")
    generation = 0
    evaluated_cache = {}
    while generation is None or generation <= generations:
        best = parent
        child = factory.mutate(parent)

        for _ in range(lambda_):
            osobnik_id += 1
            child = factory.mutate(parent)
            if tuple(child.genes) in evaluated_cache:
                child.score = evaluated_cache[tuple(child.genes)]
            else:
                child.score = evaluate(opts, child, logger=logger, osobnik_id=osobnik_id)
                evaluated_cache[tuple(child.genes)] = child.score
            logger.record(key="children", osobnik_id=osobnik_id, genome=child.genes, score=child.score)
            if child.score and child.score <= best.score:
                best = child
                parent = best
        parent = best
        print(f"gen {generation} | best score: {parent.score}")    
        logger.record(key="evolution", generation=generation, genome=parent.genes, score=parent.score)
        generation += 1

def initial_setup(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    os.makedirs(opts.save_dir)

def run(opts):
    initial_setup(opts)
    logger = Logger(opts)

    opts.reproducible_seed = False
    opts.n_epochs = 5
    opts.epoch_size = 12800
    baseline = [ (4,0), (5,(0,9),), (2,10), (3,11,1), (7,12), (3,13,-1), (5,(10, 14),), (2,15)]
    x_dim = len(baseline)
    genome = [*baseline,
             *baseline,
             *baseline,]
    outputs = [16]
    encoder = CGP_Net(opts.embedding_dim, x_dim, 3, outputs, genome=genome)
    model = AttentionModel(opts, encoder)
    
    opts.n_epochs = 5
    #model.load_weights('/home/piotr/repos/magisterka/outputs/run_20260507T142017/snapshot_osobnik0_epoch5.pth')
    evaluate(opts, model, logger, 0,)
    #evaluate(opts, model, logger, 0, snapshots_epochs=[5, 10, 15])


if __name__ == "__main__":
    run(get_options())
 