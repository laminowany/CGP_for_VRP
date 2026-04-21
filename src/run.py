import os
import json
import time
import torch
import torch.optim as optim
import random

from utils.process import get_options
from learning.attention_model import AttentionModel
from learning.cgp import GenomeFactory, Genome
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.reinforce_baselines import RolloutBaseline, WarmupBaseline
from learning.problem_vrp import CVRP
from utils.training import evaluate, evalaute_with_encoder
from utils.logger import Logger

def benchmark_execution_time(opts, logger: Logger, vrp_sizes = [10, 20, 50, 100]):
    baseline = GenomeFactory().produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3)
    osobnik_id = 0
    for vrp_size in vrp_sizes:
        opts.graph_size = vrp_size
        evaluate(opts, baseline, osobnik_id=osobnik_id)
        osobnik_id += 1
    logger.save_csv("execution_times.csv")

def verify_sanity(opts, logger: Logger):
    opts.n_epochs = 1
    opts.epoch_size = 12800
    baseline = GenomeFactory().produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    my_encoder = baseline.build_nn(opts)
    score_orig_encoder = evalaute_with_encoder(opts, my_encoder, logger, osobnik_id=0)
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    score_genetic_encoder = evalaute_with_encoder(opts, original_encoder, logger, osobnik_id=1)
    if score_orig_encoder != score_genetic_encoder:
        print("CARAMBA!")
    else:
        print("All good boss")

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

def test_single_chromosome(opts, logger, genes, osobnik_id=None):
    genome = GenomeFactory().produce_genome(genes)
    evaluate(opts, genome, logger=logger, osobnik_id=osobnik_id)

def test_chromosomes(opts, logger, genes):
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    factory = GenomeFactory()
    osobnik_id = 0
    for gene in genes:
        evaluate(opts, factory.produce_genome(gene), logger=logger, osobnik_id=osobnik_id)
        osobnik_id += 1

def test_random_chromosomes(opts, logger):
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    factory = GenomeFactory()
    genome = genome = factory.get_random_genome(50, 10)
    osobnik_id = 0
    while True:
        score = evaluate(opts, genome, logger=logger, osobnik_id=osobnik_id)
        logger.record(key="randomization", osobnik_id=osobnik_id, genome=genome.genes, score=score)
        osobnik_id += 1
        genome = factory.get_random_genome(50, 10)
        print(genome.genes)

def benchmark_random_chromosomes(opts, logger, length, n = 10):
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    factory = GenomeFactory()
    #baseline = factory.produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3) # baseline
    #logger.record(key="children", osobnik_id=0, genome=chromosomes[-1])
    osobnik_id = 0
    while osobnik_id < n:
        genome = factory.get_random_genome(length)
        score = evaluate(opts, genome, logger=logger, osobnik_id=osobnik_id)
        if not score:
            continue
        logger.record(key="genomes", osobnik_id=osobnik_id, score=score, genome=genome.genes)
        osobnik_id += 1

def initial_setup(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    os.makedirs(opts.save_dir)

def run(opts):
    initial_setup(opts)
    logger = Logger(opts)
    # opts.n_epochs = 10
    # opts.epoch_size = 128000
    # opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    # genome = [(2,), (1,), (9,), (9,), (9,), (8,), (5, -8), (1,), (3,), (5, -1), (5, -8), (9,), (9,), (9,), (1,), (1,), (6, 0), (6, 0), (6, -1), (9,), (8,), (6, 1), (2,), (1,), (1,), (5, -18)]
    # for i in range(10):
    #     test_single_chromosome(opts, logger, genome, i)

    #benchmark_random_chromosomes(opts, logger, length=25, n=10)
    # genomes = load_genomes("./genomes/initialversion.txt")
    # test_chromosomes(opts, logger, genomes)
    # return

    cgp(opts, logger=logger, generations=64, parent=GenomeFactory().produce_genome([(4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4), (1,), (4,), (5, -2), (9,), (6, 1), (7,), (6, -1), (5, -4), (1,), (4,), (5, -2), (1,), (6, 1), (6, 0), (6, -1), (5, -4), (1,)]))
    return

import ast
def load_genomes(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                parsed = ast.literal_eval(line)
                result.append(parsed)
    
    return result

if __name__ == "__main__":
    run(get_options())
 