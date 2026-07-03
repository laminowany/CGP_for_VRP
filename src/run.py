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
    os.makedirs(os.path.join(
        opts.save_dir,
        "genomes_full"
    ))
    opts.genomes_full_out_dir = os.path.join(opts.save_dir, "genomes_full")
    os.makedirs(os.path.join(
        opts.save_dir,
        "genomes_active"
    ))
    opts.genomes_active_out_dir = os.path.join(opts.save_dir, "genomes_active")
    os.makedirs(os.path.join(
        opts.save_dir,
        "parents"
    ))
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

def verify_sanity(opts, logger: Logger):
    orig_epochs = opts.n_epochs
    orig_epoch_size = opts.epoch_size
    opts.n_epochs = 1
    opts.epoch_size = 1

    reset_seeds(opts)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    model_original = AttentionModel(opts, original_encoder)
    score_original_encoder = evaluate(opts, model_original, logger, osobnik_id=-1)
    
    reset_seeds(opts)
    baseline = produce_reference(2, 3)
    x_dim = len(baseline)
    genome = [*[None]*(x_dim + 1),
             *baseline,
             *[None]*x_dim, (5, 48)]
    encoderCGP = CGP_Net(opts, x_dim, 3, genome)
    model3 = AttentionModel(opts, encoderCGP)
    score_cgp = evaluate(opts, model3, logger, osobnik_id=-2)
    if score_original_encoder != score_cgp:
        raise Exception("CARAMBA!")
    print("ALL GOOD BOSS")
    opts.n_epochs = orig_epochs
    opts.epoch_size = orig_epoch_size
    
    
def test_single_chromosome(output_dir_path, opts, chromosome_id, logger):
    csv_path = os.path.join(output_dir_path, "children.csv")
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["osobnik_id"]) == chromosome_id:
                genome = ast.literal_eval(row["genome"])
                model = AttentionModel(opts, CGP_Net(opts, 10, 3, genome))
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.save_dir, f"CHROMOSOME_{chromosome_id}"), only_active=True)
                break

    evaluate(opts, model, logger, chromosome_id)


def run(opts):
    initial_setup(opts)
    logger = Logger(opts)
    verify_sanity(opts, logger)
    opts.debug = False
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    
    children_limit = 4
    osobnik_id = 1
    scores = {}
    generation = 1
    result_cache = {}
    snapshots = {}
    best_score = None
    best_model = None
    
    # opts.n_epochs = 1
    # opts.epoch_size = 1
    # best_score = 10000
    
    # opts.x_lim = 2
    # opts.y_lim = 1
    # opts.n_epochs = 1
    # opts.epoch_size = 1
    # model = AttentionModel(opts,  CGP_Net(
    #         opts = opts,
    #         x_dim = opts.x_dim,
    #         y_dim = opts.y_dim,
    #         genome = [None, (4, 0), (2, 16), (1, 62), (6, 48), (7, 19), (1, 50), (1, 51), 
    #                   (4, 52), (4, 23), (4, 24), (5, (40, 55)), (3, 41, 1), (6, 72), (2, 28),
    #                   (4, 74), (7, 0), (6, 16), (3, 62, 1), (7, 3), (3, 64, 1), (4, 65),
    #                   (3, 36, 0), (2, 67), (1, 68), (2, 39), (3, 10, 1), (3, 71, 1), (6, 27),
    #                   (7, 73), (2, 44), (3, 0, 1), (1, 31), (1, 32), (7, 33), (1, 4), (1, 65), 
    #                   (4, 51), (6, 37), (5, 38), (7, 9), (7, 40), (2, 71), (4, 12), (4, 43), 
    #                   (3, 14, -1), (6, 0), (7, 1), (7, 32), (6, 48), (5, 4), (3, 35, 1), (6, 21),
    #                   (5, (37, 7)), (7, 8), (7, 54), (2, 25), (7, 71), (1, 12), (6, 28), (6, 44),
    #                   (2, 0), (7, 31), (7, 17), (4, 18), (2, 64), (2, 50), (2, 36), (4, 67), 
    #                   (1, 8), (1, 9), (5, (40, 55)), (1, 26), (6, 57), (7, 58), (5, 59), (5, (45, 60, 75, 15))]
    #     ))
    # result = evaluate(opts, model, logger, osobnik_id)
    # return
    # while True:
    #     export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"_PARENT_{osobnik_id}"), only_active=True)
    #     logger.record(key="children", osobnik_id=osobnik_id, genome=parent.get_encoder().genome)
    #     evaluate(opts, parent, logger, osobnik_id)
    #     parent = AttentionModel(opts, parent.get_encoder().produce_offspring(1, opts)[0])
    #     osobnik_id += 1

    # test_single_chromosome("/home/piotr/repos/magisterka/outputs/run_20260616T232231", opts, 66, logger)
    # return
    if opts.random_search:
        budget = opts.budget
        while budget > 0:
            model = AttentionModel(opts, CGP_Net.random_genome(opts, opts.x_dim, opts.y_dim))
            export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"CHILD_{osobnik_id}"), only_active=False)
            export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"CHILD_{osobnik_id}"), only_active=True)
            logger.record(key="children", osobnik_id=osobnik_id, genome=model.get_encoder().genome)
            hashkey = hash(model.get_encoder()) 
            if hashkey in result_cache: # duplicate
                continue
            result = evaluate(opts, model, logger, osobnik_id)
            if result:    
                score =  result.scores[-1]
                logger.record(key="children_scores", osobnik_id=osobnik_id, final_score=score)
                result_cache[hashkey] = score
                
                if not best_score or  compare_floats(score, best_score) == -1:
                    best_score = score
                    del best_model
                    best_model = model
                    best_id = osobnik_id
                    print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}.')
                    export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{generation}_genome_{osobnik_id}"), only_active=True)
                else:
                    del model
            logger.record(key="budget", budget=(opts.budget - budget), best_id=best_id, score=best_score)
            print(f'Iteracja {opts.budget - budget + 1}, najlepszy {best_score}') 
            osobnik_id += 1
            budget -= 1
            #gc.collect()
        return
    elif opts.start_from_transformer:
        baseline = produce_reference(2, 3)
        x_dim = len(baseline)
        genome = [*[None]*x_dim,
                *baseline,
                *[None]*x_dim,]
        outputs = [48]
        baseline = AttentionModel(opts, CGP_Net(opts, x_dim, 3, genome, outputs))
        hashkey = hash(baseline.get_encoder()) 
        best_weights = torch.load('/home/piotr/repos/magisterka/outputs/BASE_GENOME_CGP2D/snapshot_osobnik0_epoch10.pth')
        best_score = 4.8383 #4.785
        result_cache[hashkey] = best_score
        best_model = baseline
        best_id = 0
        export_cgp_to_graphviz(baseline.get_encoder().genes, os.path.join(opts.genomes_full_out_dir, f"TRANSFORMER{osobnik_id}"), only_active=False)
        export_cgp_to_graphviz(baseline.get_encoder().genes, os.path.join(opts.genomes_active_out_dir, f"TRANSFORMER{osobnik_id}"), only_active=True)
    else:
        best_score = None
        while not best_score:
            first_parent = AttentionModel(opts, CGP_Net.random_genome(opts, opts.x_dim, opts.y_dim))
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
        
        
        # orig_epochs = opts.n_epochs
        # opts.n_epochs =  opts.n_epochs * 2
        # best_score = None
        # for _ in range(children_limit + 1):
        #     candidate = AttentionModel(opts, CGP_Net.random_genome(opts, opts.x_dim, opts.y_dim))
        #     export_cgp_to_graphviz(candidate.get_encoder().genes, os.path.join(opts.genomes_full_out_dir, f"_CANDIDATE_{osobnik_id}"), only_active=False)
        #     export_cgp_to_graphviz(candidate.get_encoder().genes, os.path.join(opts.genomes_active_out_dir, f"_CANDIDATE_{osobnik_id}"), only_active=True)
        #     hashkey = hash(candidate.get_encoder()) 

        #     if hashkey in result_cache: # duplicate
        #         continue
            
        #     result = evaluate(opts, candidate, logger, osobnik_id, snapshots_epochs=[orig_epochs])
        #     if result:    
        #         scores[osobnik_id] =  result.scores[-1]
        #         snapshots[osobnik_id] = result.snapshots[0]
        #         result_cache[hashkey] = scores[osobnik_id]
        #         if not best_score or compare_floats(scores[osobnik_id], best_score) == -1:
        #             best_score = scores[osobnik_id]
        #             best_model = candidate
        #             best_id = osobnik_id
        #     osobnik_id += 1
        # opts.n_epochs = orig_epochs
        # best_weights = snapshots[best_id]
        
    parent_weights = best_weights
    parent_score = best_score
    parent = best_model
    export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}"), only_active=True)
    export_cgp_to_graphviz(parent.get_encoder().genes, opts, os.path.join(opts.parents_out_dir, f"generation_{0}_genome_{best_id}_FULL"), only_active=False)
    
    
    budget = opts.budget
    #while generation <= opts.generations:  
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
                #print(f'Osobnik {osobnik_id} dryfuje.')
                scores[osobnik_id] = result_cache[hashkey]
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_full_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=False)
                export_cgp_to_graphviz(model.get_encoder().genes, opts, os.path.join(opts.genomes_active_out_dir, f"{generation}_genome_{osobnik_id}_EQ"), only_active=True)
                if compare_floats(parent_score, best_score) == 0: # neutral drift
                    best_model = model
                    best_id = osobnik_id
                    #print(f'Osobnik {osobnik_id} nowym najlepszym genomem.')
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
                        #print(f'Osobnik {osobnik_id} o koszcie: {scores[osobnik_id]}.')
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
      
    # reset_seeds(opts)
    # opts.epoch_size = 1280000
    # opts.n_epochs = 100
    # reset_seeds(opts)
    # print(f'Ewaluacja pierwszego rodzica.')    
    # result = evaluate(opts, first_parent, logger, osobnik_id)
    # if result:
    #     print(f"Wynik: {result.scores[-1]}")
    # osobnik_id += 1
    # print(f'Ewaluacja ostatniego dziecka.')    
    # reset_seeds(opts)
    # result = evaluate(opts, parent, logger, osobnik_id)
    # if result:
    #     print(f"Wynik: {result.scores[-1]}")
    
if __name__ == "__main__":
    run(get_options())
 