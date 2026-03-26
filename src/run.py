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
from utils.training import train_epoch
from utils.logger import Logger

def evaluate(opts, genome: Genome, logger: Logger, osobnik_id = None, validation_set = None):
    encoder = genome.build_nn(opts)
    return evalaute_with_encoder(opts, encoder, logger=logger, osobnik_id=osobnik_id, validation_set=validation_set)

def evalaute_with_encoder(opts, encoder, logger: Logger, osobnik_id = None, validation_set = None):
    if not osobnik_id:
        osobnik_id = 0

    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        encoder=encoder,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    baseline = RolloutBaseline(model, opts)
    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    if not validation_set:
        validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        start = time.perf_counter()
        try:
            score = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                validation_set,
                opts
            )
        except TimeoutError:
            return 10e6
        end = time.perf_counter()
        logger.record(
                epoch=epoch,
                osobnik=osobnik_id,
                score=score,
                time=end-start
        )
    return score

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

def cgp(opts, parent, logger: Logger, generations = None):
    validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    opts.n_epochs = 5
    opts.epoch_size = 128000
    factory = GenomeFactory()
    lambda_ = 4
    osobnik_id = 0   
    #parent = factory.produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3) # baseline
    parent =  factory.produce_genome([(4,), (5, -2), (1,), (6, 1), (2,), (3,), (6, 0), (7,), (1,), (1,), (6, 1), (7,), (6, -1), (5, -5), (8,), (7,), (5, -2), (1,), (6, 0), (7,), (6, -1), (9,), (5, -12), (1,), (6, 1), (6, -1)])
    parent = factory.get_random_genome(25, 0)
    parent.score = evaluate(opts, parent, logger=logger, osobnik_id=osobnik_id, validation_set=validation_set)
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
                child.score = evaluate(opts, child, logger=logger, osobnik_id=osobnik_id, validation_set=validation_set)
                evaluated_cache[tuple(child.genes)] = child.score
            logger.record(key="children", osobnik_id=osobnik_id, genome=child.genes, score=child.score)
            # print(f'child score:  {child.score}')
            if child.score <= best.score:
                best = child
                parent = best
        parent = best
        print(f"gen {generation} | best score: {parent.score}")    
        logger.record(key="evolution", generation=generation, genome=parent.genes, score=parent.score)
        generation += 1

def test_single_chromosome(opts, logger, genes):
    opts.n_epochs = 1
    opts.epoch_size = 12800
    genome = GenomeFactory().produce_genome(genes)
    evaluate(opts, genome, logger=logger)

def test_random_chromosomes(opts, logger):
    opts.n_epochs = 1
    opts.epoch_size = 12800
    validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    factory = GenomeFactory()
    genome = genome = factory.get_random_genome(50, 10)
    osobnik_id = 0
    while True:
        score = evaluate(opts, genome, logger=logger, osobnik_id=osobnik_id, validation_set=validation_set)
        logger.record(key="randomization", osobnik_id=osobnik_id, genome=genome.genes, score=score)
        osobnik_id += 1
        genome = factory.get_random_genome(50, 10)
        print(genome.genes)

def benchmark_random_chromosomes(opts, logger):
    opts.n_epochs = 100
    opts.epoch_size = 1280000
    validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    factory = GenomeFactory()
    baseline = factory.produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3) # baseline
    chromosomes = [baseline]
    logger.record(key="children", osobnik_id=0, genome=chromosomes[-1])
    for i in range(10):
        chromosomes.append(factory.get_random_genome(len(baseline.genes), len(baseline.genes)//5))
        logger.record(key="children", osobnik_id=i+1, genome=chromosomes[-1])
    osobnik_id = 0
    for chrom in chromosomes:
        score = evaluate(opts, chrom, logger=logger, osobnik_id=osobnik_id, validation_set=validation_set)
        logger.record(key="performance", osobnik_id=osobnik_id, score=score)
        osobnik_id += 1

def initial_setup(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    os.makedirs(opts.save_dir)

def run(opts):
    initial_setup(opts)
    logger = Logger(opts)

    
    best_known_so_far = GenomeFactory().produce_genome([(4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4), (1,), (4,), (5, -2), (9,), (6, 1), (7,), (2,), (6, -1), (1,), (1,), (8,), (5, -2), (1,), (6, 1), (7,), (1,), (3,), (1,), (6, -1)])
    cgp(opts, parent=best_known_so_far, logger=logger, generations=100)
    return
    #print(scores)
    #opts.n_epochs = 100
    # opts.graph_size = 10
    # print(f"EPOCHS: {opts.n_epochs}")
    # print(f"GRAPH SIZE: {opts.graph_size}")
    # scores = []
    # genomes = []
    # baseline_genom = GenomeEncoder(opts.emb, [(4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4), (1,)]*3)
    # genomes.append(baseline_genom)
    # scores = calculate_scores(opts, tb_logger, genomes[0], osobnik_id=0)
    # print(scores)
    # for i in range(4):
    #     genome = GenomeEncoder.spawn_random_genome()
    #     genomes.append(genome)

    #score = calculate_score(opts, tb_logger, genome, logger, osobnik_id=i)
    # for i, genome in enumerate(genomes):
    #     score = calculate_score(opts, tb_logger, genome, logger, osobnik_id=i)

    # logger.save_csv("vrp10_20_epochs_epoch_128000.csv")



if __name__ == "__main__":
    run(get_options())
 