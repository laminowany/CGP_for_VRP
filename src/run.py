import ast
import csv
import gc
import os
from pathlib import Path
import numpy as np
import torch
import random

from utils.graph import export_cgp_to_graphviz
from utils.misc import compare_floats
from utils.process import get_options
from learning.attention_model import AttentionModel
from learning.cgp import CGP_Net
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.problem_vrp import CVRP
from utils.training import evaluate, validate 
from utils.logger import Logger

def initial_setup(opts):
    os.makedirs(opts.save_dir)
    if opts.mode == "cgp" or opts.mode == "random_search" or opts.mode == "evolve_transformer":    
        os.makedirs(os.path.join(opts.save_dir,"genomes_full"))
        opts.genomes_full_out_dir = os.path.join(opts.save_dir, "genomes_full")
        os.makedirs(os.path.join(opts.save_dir,"genomes_active"))
        opts.genomes_active_out_dir = os.path.join(opts.save_dir, "genomes_active")
        os.makedirs(os.path.join(opts.save_dir,"parents"))
        opts.parents_out_dir = os.path.join(opts.save_dir, "parents")
    reset_seeds(opts)
    
def reset_seeds(opts):
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    
def produce_transformer_genome(opts):
    if opts.x_dim < 8:
        raise Exception("For transformer to fit it genotype the x_dim must have at least 8 length")
    to_global_idx = lambda x, y, opts: opts.x_dim * opts.y_dim + 1 if x == opts.x_dim else y * opts.x_dim + x + 1
    length = opts.x_dim * opts.y_dim
    genome = [None] * (length + 2)
    for y in range(opts.y_dim):
        genome[to_global_idx(0, y, opts)] = ((1, 0))
        for x in range(1, opts.x_dim):
            pos = to_global_idx(x, y, opts)
            genome[pos] = ((1, to_global_idx(x - 1, y, opts)))
    main_row = opts.y_dim // 2
    genome[-1] = (5, (to_global_idx(opts.x_dim - 1, main_row, opts)))    
    
    prev_pos = 0
    x = 0
    while x <= opts.x_dim - 8:
        pos = to_global_idx(x, main_row, opts)
        genome[pos] = ((4, prev_pos))
        genome[to_global_idx(x, main_row - 1, opts)] = ((1, prev_pos))
        genome[pos + 1] = ((5, (to_global_idx(x, main_row - 1, opts), pos)))   
        genome[pos + 2] = ((2, pos + 1)) 
        genome[pos + 3] = ((3, pos + 2, 1))
        genome[to_global_idx(x + 3, main_row - 1, opts)] = ((1, pos + 2))
        genome[pos + 4] = ((7, pos + 3))
        genome[pos + 5] = ((3, pos + 4, -1))
        genome[pos + 6] = ((5, (to_global_idx(x + 5, main_row - 1, opts), pos + 5)))  
        genome[pos + 7] = ((2, pos + 6))
        prev_pos = pos + 7
        x = x + 8
    
    return genome

def verify_sanity(opts, logger: Logger):
    orig_epochs = opts.n_epochs
    orig_epoch_size = opts.epoch_size
    orig_graph_size = opts.graph_size
    orig_y_dim = opts.y_dim
    orig_x_dim = opts.x_dim
    orig_no_progress_bar = opts.no_progress_bar
    
    opts.n_epochs = 1
    opts.epoch_size = 1
    opts.graph_size = 10
    opts.no_progress_bar = True
    reset_seeds(opts)
    score_original_encoder = evaluate(opts, logger, candidate_id=-2)
    reset_seeds(opts)
    opts.x_dim = 24
    encoder = CGP_Net(opts, produce_transformer_genome(opts))
    logger.record(key="candidates", id=0, genome=encoder.genome)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"TRANSFORMER"), only_active=False)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"TRANSFORMER_ACTIVE"), only_active=True)
    score_cgp = evaluate(opts, logger, encoder, candidate_id=-1)
    if score_original_encoder != score_cgp:
        raise Exception("CARAMBA!")
    print("ALL GOOD BOSS")
    opts.n_epochs = orig_epochs
    opts.epoch_size = orig_epoch_size
    opts.graph_size = orig_graph_size
    opts.y_dim = orig_y_dim
    opts.x_dim = orig_x_dim
    opts.no_progress_bar = orig_no_progress_bar

def run_transformer_evolution(opts, logger):
    parent_encoder = CGP_Net(opts, produce_transformer_genome(opts))
    osobnik_id = 1
    result = evaluate(opts, logger, parent_encoder, osobnik_id)  
    train_and_validate_encoder(opts, logger, parent_encoder, result)
        
def train_and_validate_encoder(opts, logger, encoder, encoder_result):
    children_limit = 4
    scores = {}
    result_cache = {}
    osobnik_id = 0
    generation = 1
    
    hashkey = hash(encoder)   
    scores[osobnik_id] =  encoder_result.scores[-1]
    result_cache[hashkey] = scores[osobnik_id]      
    best_score = scores[osobnik_id]
    best_encoder = encoder
    best_id = osobnik_id
    best_weights = encoder.save_snapshot()
    osobnik_id += 1
    
    parent_weights = best_weights
    parent_score = best_score
    parent_encoder = best_encoder

    logger.record(key="candidates", id=best_id, genome=encoder.genome)
    logger.record(key="candidates_scores", id=best_id, final_score=scores[best_id])
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"_PARENT_{best_id}"), only_active=False)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"_PARENT_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}_FULL"), only_active=False)
    
    budget = opts.budget
    while budget > 0:
        encoders = parent_encoder.produce_offspring(children_limit, opts, budget)
        export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}__PARENT"), only_active=False)
        export_cgp_to_graphviz(parent_encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}__PARENT"), only_active=True)
        
        for encoder in encoders:
            logger.record(key="candidates", id=osobnik_id, genome=encoder.genome)
            hashkey = hash(encoder) 
            parent_equivalent = CGP_Net.are_equivalent(parent_encoder, encoder)
            
            if parent_equivalent:
                scores[osobnik_id] = result_cache[hashkey]
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=False)
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=True)
                if compare_floats(parent_score, best_score) == 0: # neutral drift
                    best_encoder = encoder
                    best_id = osobnik_id
            else:
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=False)
                export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=True)
                encoder.load_snapshot(parent_weights)
                if hashkey in result_cache:
                    scores[osobnik_id] = result_cache[hashkey]
                else:
                    result = evaluate(opts, logger, encoder, osobnik_id)
                    if result:
                        scores[osobnik_id] = result.scores[-1]
                        result_cache[hashkey] = scores[osobnik_id]
                        if compare_floats(scores[osobnik_id], best_score) == -1:
                            best_encoder = encoder
                            best_score = scores[osobnik_id]
                            best_weights = encoder.save_snapshot()
                            best_id = osobnik_id
                            print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}.')
                            export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.parents_out_dir, f"generation_{generation}_genome_{osobnik_id}"), only_active=True)
                    else:
                        scores[osobnik_id] = None
                    budget -= 1
                    if budget == 0:
                        break
            logger.record(key="candidates_scores", id=osobnik_id, final_score=scores[osobnik_id])
            osobnik_id += 1
        parent_encoder = best_encoder
        parent_score = best_score
        parent_weights = best_weights
            
        print(f'Generacja {generation}, najlepszy {best_score}')    
        logger.record(key="evolution", generation=generation, best_id=best_id, score=best_score)
        logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
        generation += 1

def run_random_search(opts, logger):
    result_cache = {}
    best_score = None
    best_model = None
    candidate_id = 1
    budget = opts.budget
    while budget > 0:
        encoder = CGP_Net.random_genome(opts)
        model = AttentionModel(opts, CGP_Net.random_genome(opts))
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"CANDIDATE_{candidate_id}"), only_active=False)
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"CANDIDATE_{candidate_id}"), only_active=True)
        logger.record(key="candidates", id=candidate_id, genome=model.get_encoder().genome)
        hashkey = hash(model.get_encoder()) 
        if hashkey in result_cache: # duplicate
            continue
        result = evaluate(opts, logger, encoder, candidate_id)
        if result:    
            score =  result.scores[-1]
            logger.record(key="scores", id=candidate_id, final_score=score)
            result_cache[hashkey] = score
            
            if not best_score or compare_floats(score, best_score) == -1:
                best_score = score
                del best_model
                best_model = model
                best_id = candidate_id
                print(f'Found better candidate {candidate_id} with cost: {best_score}.')
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"genome_{candidate_id}"), only_active=True)
            else:
                del model
        logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
        print(f'Iteration {opts.budget - budget + 1}, best so far {best_score}') 
        candidate_id += 1
        budget -= 1
    logger.record(key="winner", id=candidate_id, x_dim = opts.x_dim, y_dim = opts.y_dim, genome=best_model.get_encoder().genome)
    
def run_full_evaluation(opts, logger):

    csv_path = os.path.join(opts.genome_path, "candidates.csv")
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["id"]) == opts.id:
                genome = ast.literal_eval(row["genome"])
                model = AttentionModel(opts, CGP_Net(opts, genome))
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.save_dir, f"CANDIDATE_{opts.id}_ACTIVE"), only_active=True)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.save_dir, f"CANDIDATE_{opts.id}"), only_active=False)
                break
            
    
    evaluate(opts, model, logger, opts.id, snapshots_epochs=[opts.n_epochs])

def run_genome_evaluation(opts, logger):
    if not opts.genome:
        raise Exception("Please specify genome with --genome")
    genome = opts.genome
    encoder = CGP_Net(opts, genome)
    logger.record(key="genomes", name = opts.genome_name, genome = genome)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"candidate"), only_active=True)
    
    evaluate(opts, logger, encoder)

def run_cgp(opts, logger):
    osobnik_id = 0
    result = None
    while not result:
        encoder = CGP_Net.random_genome(opts)
        result = evaluate(opts, logger, encoder, osobnik_id)
        if result:    
            train_and_validate_encoder(opts, logger, encoder, result)

def run_generate_validation_data(opts):
    validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_test_size)
    torch.save(validation_set.data, os.path.join(opts.save_dir, f'dataset_{opts.graph_size}CVRP_seed_{opts.seed}.pt'))
    
def plot_best_genome(run_dir):
    run_dir = Path(run_dir)

    budget_path = run_dir / "budget.csv"
    candidates_path = run_dir / "candidates.csv"

    # Get best_id from the last row of budget.csv
    with budget_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{budget_path} is empty")

    best_id = int(rows[-1]["best_id"])

    # Find corresponding genome in candidate.csv
    with candidates_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["id"]) == best_id:
                genome = ast.literal_eval(row["genome"])
                net = CGP_Net(opts, genome)
                print(f'saving to {os.path.join(run_dir, f"cgp10")}')
                export_cgp_to_graphviz(net.genes, opts, os.path.join(run_dir, f"cgp10"), only_active=True)
                return best_id, genome
    print( os.path.join(run_dir, f"parent"))
  
    raise ValueError(f"Candidate with id={best_id} not found in {candidates_path}")

def run_evaluation(opts, logger):
    # if not opts.genome_name:
    #     raise Exception("Please specify name of architecture with --genome_name")
    # genomes = {
    #     "TRANS1" : [None, (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 0), (1, 9), (1, 10), (1, 19), (1, 12), (1, 13), (1, 14), (1, 15), (4, 0), (5, (9, 17)), (2, 18), (3, 19, 1), (7, 20), (3, 21, -1), (5, (14, 22)), (2, 23), (1, 0), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 30), (1, 31), (1, 0), (1, 33), (1, 34), (1, 35), (1, 36), (1, 37), (1, 38), (1, 39), (5, 24)],
    #     "EVO1" : [None, (7, 0), (2, 25), (4, 34), (3, 19, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31), (4, 0), (4, 9), (4, 2), (1, 19), (3, 28, -1), (1, 37), (7, 14), (6, 39), (5, 0), (5, (25, 33, 1)), (2, 18), (3, 19, 1), (2, 28), (7, 29), (1, 38), (5, 31), (5, 0), (4, 1), (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15), (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39), (5, 32)],
    #     "EVO2": [None, (1, 0), (6, 25), (3, 26, 0), (2, 35), (4, 28), (5, 21), (6, 38), (7, 15), (5, 0), (4, 25), (4, 18), (2, 19), (7, 12), (1, 37), (6, 14), (1, 15), (4, 0), (5, (9, 17)), (2, 18), (1, 19), (7, 20), (4, 21), (6, 38), (4, 7), (2, 0), (4, 1), (3, 26, 1), (3, 35, 1), (4, 4), (1, 21), (2, 30), (5, 31), (1, 0), (4, 25), (5, 26), (2, 3), (6, 28), (3, 13, 0), (5, 22), (1, 39), (5, 32)],
    #     "EVO10": [None, (5, 0), (3, 1, 1), (7, 10), (4, 35), (3, 36, 1), (2, 37), (5, (22, 30, 14)), (6, 7), (2, 0), (7, 1), (7, 26), (7, 19), (3, 12, 1), (3, 37, 0), (6, 38), (3, 39, 1), (4, 0), (5, (9, 17)), (3, 18, 0), (4, 27), (7, 12), (7, 13), (7, 30), (2, 23), (2, 0), (1, 25), (6, 2), (3, 27, 1), (1, 12), (1, 21), (2, 22), (4, 7), (1, 0), (6, 33), (7, 26), (1, 3), (2, 4), (6, 37), (2, 38), (7, 15), (5, 24)],
    # }
    # if opts.genome_name not in genomes:
    #     raise Exception(f"Architecture {opts.genome_name} not found")
    if not opts.test_set_path:
            raise Exception("Please specify name of test set with --test_set_path")
    
    encoder = CGP_Net(opts, (opts.genome))
    model = AttentionModel(opts, encoder)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"candidate"), only_active=True)
    checkpoint = torch.load(opts.checkpoint_path, map_location=opts.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    
    score = validate(model, opts.test_set, opts)
    
    logger.record(key="scores", score=score)
    print(f"Final score {score}")
    # encoder.print_active_parameters()

    # print(f"Active encoder parameters: {encoder.count_active_parameters():,}")

   

def verify_sanity2(opts, logger: Logger):
    #reset_seeds(opts)
    
    torch.manual_seed(opts.seed)
    # evo1 =  CGP_Net(opts, genome=[None, (7, 0), (2, 25), (4, 34), (3, 19, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31), (4, 0), (4, 9), (4, 2), 
    #     (1, 19), (3, 28, -1), (1, 37), (7, 14), (6, 39), (5, 0), (5, (25, 33, 1)), (2, 18), (3, 19, 1), (2, 28), 
    #     (7, 29), (1, 38), (5, 31), (5, 0), (4, 1), (4, 26), (1, 19), (6, 28), 
    #     (5, 29), (2, 22), (2, 15), (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39), (5, 32)])

    # original_encoder = GraphAttentionEncoder(
    #     n_heads=opts.n_heads,
    #     embed_dim=opts.embedding_dim,
    #     n_layers=opts.n_encode_layers,
    #     normalization=opts.normalization
    # )
   # model_original = AttentionModel(opts, evo1)
    model = AttentionModel(opts).to(opts.device)
    score_original_encoder = evaluate(opts, logger)
   
def percentage_reduction(original, new):
    return (new - original) / original * 100

if __name__ == "__main__":
    # transformer_params = 16.7780 
    # evo_params =16.8359
    # reduction = percentage_reduction(transformer_params, evo_params)
    # print(f"Reduction: {reduction:.2f}%")
    # exit()
    
    
    opts = get_options()
    initial_setup(opts)
    logger = Logger(opts)
    
    #evaluate(opts, logger)
    # verify_sanity(opts, logger)
    # exit()
    if opts.mode == "cgp":    
        run_cgp(opts, logger)
    elif opts.mode == "random_search":
        run_random_search(opts, logger)
    elif opts.mode == "full_evaluation":
        run_full_evaluation(opts, logger)
    elif opts.mode == "evolve_transformer":
        run_transformer_evolution(opts, logger)
    elif opts.mode == "generate_validation_data":
        run_generate_validation_data(opts)
    elif opts.mode == "genome_evaluation":
        run_genome_evaluation(opts, logger)
    elif opts.mode == "evaluate":
        run_evaluation(opts, logger)