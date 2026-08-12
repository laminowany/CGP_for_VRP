import ast
import csv
import gc
import os
import torch
import random

from utils.graph import export_cgp_to_graphviz
from utils.misc import compare_floats
from utils.process import get_options
from learning.attention_model import AttentionModel
from learning.cgp import CGP_Net
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.problem_vrp import CVRP
from utils.training import evaluate 
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

def produce_transformer_encoder(opts):
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
    
    model = AttentionModel(opts, CGP_Net(opts, genome))
    export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.save_dir, f"TRANSFORMER"), only_active=False)
    return model

def verify_sanity(opts, logger: Logger):
    orig_epochs = opts.n_epochs
    orig_epoch_size = opts.epoch_size
    orig_graph_size = opts.graph_size
    orig_y_dim = opts.y_dim
    orig_x_dim = opts.x_dim
    opts.n_epochs = 1
    opts.epoch_size = 1
    opts.graph_size = 10   
    reset_seeds(opts)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    model_original = AttentionModel(opts, original_encoder)
    score_original_encoder = evaluate(opts, model_original, logger, candidate_id=-1)
    reset_seeds(opts)
    opts.x_dim = 24
    modelCGP = produce_transformer_encoder(opts)
    export_cgp_to_graphviz(modelCGP.get_encoder().genes, opts, os.path.join(opts.save_dir, f"TRANSFORMER_OLD"), only_active=False)
    score_cgp = evaluate(opts, modelCGP, logger, candidate_id=-2)
    if score_original_encoder != score_cgp:
        raise Exception("CARAMBA!")
    print("ALL GOOD BOSS")
    opts.n_epochs = orig_epochs
    opts.epoch_size = orig_epoch_size
    opts.graph_size = orig_graph_size
    opts.y_dim = orig_y_dim
    opts.x_dim = orig_x_dim

def run_transformer_evolution(opts, logger):
    first_parent = produce_transformer_encoder(opts)
    
    children_limit = 4
    osobnik_id = 1
    scores = {}
    generation = 1
    result_cache = {}
    best_score = None
    best_model = None
    best_score = None

    export_cgp_to_graphviz(first_parent.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"_PARENT_{osobnik_id}"), only_active=False)
    export_cgp_to_graphviz(first_parent.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"_PARENT_{osobnik_id}"), only_active=True)
    hashkey = hash(first_parent.get_encoder()) 
    result = evaluate(opts, first_parent, logger, osobnik_id)  
    scores[osobnik_id] =  result.scores[-1]
    result_cache[hashkey] = scores[osobnik_id]      
    best_score = scores[osobnik_id]
    best_model = first_parent
    best_id = osobnik_id
    best_weights = first_parent.get_encoder().save_snapshot()
    osobnik_id += 1
            
    parent_weights = best_weights
    parent_score = best_score
    parent = best_model
    
    budget = opts.budget
    while budget > 0:
        children = parent.get_encoder().produce_offspring(children_limit, opts, budget)
        export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}__PARENT"), only_active=False)
        export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}__PARENT"), only_active=True)
        
        for child in children:
            logger.record(key="children", osobnik_id=osobnik_id, genome=child.genome)
            model = AttentionModel(opts, child)
            hashkey = hash(child) 
            parent_equivalent = CGP_Net.are_equivalent(parent.get_encoder(), model.get_encoder())
            
            if parent_equivalent:
                scores[osobnik_id] = result_cache[hashkey]
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=False)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=True)
                if compare_floats(parent_score, best_score) == 0: # neutral drift
                    best_model = model
                    best_id = osobnik_id
            else:
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=False)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=True)
                model.load_weights(parent_weights)
                if hashkey in result_cache:
                    scores[osobnik_id] = result_cache[hashkey]
                else:
                    result = evaluate(opts, model, logger, osobnik_id)
                    if result:
                        scores[osobnik_id] = result.scores[-1]
                        result_cache[hashkey] = scores[osobnik_id]
                        if compare_floats(scores[osobnik_id], best_score) == -1:
                            best_model = model
                            best_score = scores[osobnik_id]
                            best_weights = model.get_encoder().save_snapshot()
                            best_id = osobnik_id
                            print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}.')
                            export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{generation}_genome_{osobnik_id}"), only_active=True)
                    else:
                        scores[osobnik_id] = None
                    budget -= 1
                    if budget == 0:
                        break
            logger.record(key="children_scores", osobnik_id=osobnik_id, final_score=scores[osobnik_id])
            osobnik_id += 1
        parent = best_model
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
        model = AttentionModel(opts, CGP_Net.random_genome(opts))
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"CANDIDATE_{candidate_id}"), only_active=False)
        export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"CANDIDATE_{candidate_id}"), only_active=True)
        logger.record(key="candidates", id=candidate_id, genome=model.get_encoder().genome)
        hashkey = hash(model.get_encoder()) 
        if hashkey in result_cache: # duplicate
            continue
        result = evaluate(opts, model, logger, candidate_id)
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
    
    opts.epoch_size = 1280000
    opts.n_epochs = 100
    opts.seed = 23 # fix seed so evaluations are stable (actually this sets the seed too late, but it must stay now for reproducibility :D)
    reset_seeds(opts)
    evaluate(opts, model, logger, opts.id)


def run_cgp(opts, logger):
    opts.debug = False
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    
    children_limit = 4
    osobnik_id = 1
    scores = {}
    generation = 1
    result_cache = {}
    best_score = None
    best_model = None
    best_score = None
    while not best_score:
        first_parent = AttentionModel(opts, CGP_Net.random_genome(opts))
        export_cgp_to_graphviz(first_parent.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"_PARENT_{osobnik_id}"), only_active=False)
        export_cgp_to_graphviz(first_parent.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"_PARENT_{osobnik_id}"), only_active=True)
        hashkey = hash(first_parent.get_encoder()) 
        result = evaluate(opts, first_parent, logger, osobnik_id)
        if result:    
            scores[osobnik_id] =  result.scores[-1]
            result_cache[hashkey] = scores[osobnik_id]      
            best_score = scores[osobnik_id]
            best_model = first_parent
            best_id = osobnik_id
            best_weights = first_parent.get_encoder().save_snapshot()
            osobnik_id += 1
            
    parent_weights = best_weights
    parent_score = best_score
    parent = best_model
    export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}_FULL"), only_active=False)
    
    budget = opts.budget
    while budget > 0:
        children = parent.get_encoder().produce_offspring(children_limit, opts, budget)
        export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}__PARENT"), only_active=False)
        export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}__PARENT"), only_active=True)
        
        for child in children:
            logger.record(key="children", osobnik_id=osobnik_id, genome=child.genome)
            model = AttentionModel(opts, child)
            hashkey = hash(child) 
            parent_equivalent = CGP_Net.are_equivalent(parent.get_encoder(), model.get_encoder())
            
            if parent_equivalent:
                scores[osobnik_id] = result_cache[hashkey]
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=False)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=True)
                if compare_floats(parent_score, best_score) == 0: # neutral drift
                    best_model = model
                    best_id = osobnik_id
            else:
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=False)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_NEW"), only_active=True)
                model.load_weights(parent_weights)
                if hashkey in result_cache:
                    scores[osobnik_id] = result_cache[hashkey]
                else:
                    result = evaluate(opts, model, logger, osobnik_id)
                    if result:
                        scores[osobnik_id] = result.scores[-1]
                        result_cache[hashkey] = scores[osobnik_id]
                        if compare_floats(scores[osobnik_id], best_score) == -1:
                            best_model = model
                            best_score = scores[osobnik_id]
                            best_weights = model.get_encoder().save_snapshot()
                            best_id = osobnik_id
                            print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}.')
                            export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{generation}_genome_{osobnik_id}"), only_active=True)
                    else:
                        scores[osobnik_id] = None
                    budget -= 1
                    if budget == 0:
                        break
            logger.record(key="children_scores", osobnik_id=osobnik_id, final_score=scores[osobnik_id])
            osobnik_id += 1
        parent = best_model
        parent_score = best_score
        parent_weights = best_weights
            
        print(f'Generacja {generation}, najlepszy {best_score}')    
        logger.record(key="evolution", generation=generation, best_id=best_id, score=best_score)
        logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
        generation += 1
    
if __name__ == "__main__":
    opts = get_options()
    initial_setup(opts)
    logger = Logger(opts)
    verify_sanity(opts, logger)
    
    if opts.mode == "cgp":    
        run_cgp(opts, logger)
    elif opts.mode == "random_search":
        run_random_search(opts, logger)
    elif opts.mode == "full_evaluation":
        run_full_evaluation(opts, logger)
    elif opts.mode == "evolve_transformer":
        run_transformer_evolution(opts, logger)