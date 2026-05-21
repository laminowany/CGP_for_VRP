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
    os.makedirs(os.path.join(
        opts.save_dir,
        "children"
    ))
    #reset_seeds(opts)
    
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
    genome = [*[None]*x_dim,
             *baseline,
             *[None]*x_dim,]
    outputs = [48]
    encoderCGP = CGP_Net(opts, x_dim, 3, genome, outputs)
    model3 = AttentionModel(opts, encoderCGP)
    score_cgp = evaluate(opts, model3, logger, osobnik_id=-2)
    if score_original_encoder != score_cgp:
        raise Exception("CARAMBA!")
    print("ALL GOOD BOSS")
    opts.n_epochs = orig_epochs
    opts.epoch_size = orig_epoch_size


def run(opts):
    initial_setup(opts)
    logger = Logger(opts)
    #verify_sanity(opts, logger)
    children_out_dir = os.path.join(opts.save_dir, "children")
    opts.debug = False
    opts.validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    

    opts.n_epochs = 5
    opts.epoch_size = 128000
    
    children_limit = 4
    osobnik_id = 1
    models = {}
    scores = {}
    best_so_far = 0

    generation = 1
    
    
    baseline = produce_reference(2, 3)
    x_dim = len(baseline)
    
    genome = [*[None]*x_dim,
             *baseline,
             *[None]*x_dim,]
    outputs = [48]
    baseline = AttentionModel(opts, CGP_Net(opts, x_dim, 3, genome, outputs))
    best_weights = torch.load('/home/piotr/repos/magisterka/outputs/BASE_GENOME_CGP2D/snapshot_osobnik0_epoch50.pth')
    best_score = 4.8 #4.785
    parent = baseline
    #models[0] = parent
    
    opts.n_epochs = 10
    opts.epoch_size = 128000
    # best_score = 10000
    # for i in range(children_limit + 1):
    #     models[osobnik_id] = AttentionModel(opts, CGP_Net.random_genome(opts, x_dim, 3))
    #     export_cgp_to_graphviz(models[osobnik_id].get_encoder().genes, os.path.join(children_out_dir, f"children{osobnik_id}"))
    #     scores[osobnik_id]  = evaluate(opts, models[osobnik_id], logger, osobnik_id).score
    #     if scores[osobnik_id]  < best_score:
    #         best_score = scores[osobnik_id] 
    #         best_so_far = osobnik_id
    #         best_weights = models[osobnik_id].get_encoder().save_snapshot()
    #         parent = models[osobnik_id]
    #     osobnik_id += 1
    
    # opts.n_epochs = 5
    # opts.epoch_size = 128000
    
    
    
    
    while True:  
        children = parent.get_encoder().produce_offspring2(children_limit, opts)
        export_cgp_to_graphviz(parent.get_encoder().genes, os.path.join(children_out_dir, f"children{osobnik_id}_BASE"))
        parent_weights = best_weights
        for child in children:
            logger.record(key="children", osobnik_id=osobnik_id, genome=child.genome, outputs=child.outputs)
            model = AttentionModel(opts, child)
            is_old= CGP_Net.are_equivalent(parent.get_encoder(), model.get_encoder())
            # print(f'{"OLD" if is_old else "NOWY"}')
            # continue
            export_cgp_to_graphviz(model.get_encoder().genes, os.path.join(children_out_dir, f"children{osobnik_id}"))
            if is_old:
                #export_cgp_to_graphviz(model.get_encoder().genes, os.path.join(children_out_dir, f"children{osobnik_id}"))
                scores[osobnik_id] = best_score       
                # osobnik_id += 1
                # continue
            else:
                # export_cgp_to_graphviz(model.get_encoder().genes, os.path.join(children_out_dir, f"children{osobnik_id}"))
                # osobnik_id += 1
                # continue
                model.load_weights(parent_weights)
                result = evaluate(opts, model, logger, osobnik_id)
                if result:       
                    scores[osobnik_id] = sum(result.scores[-3:])/3.0
                    if scores[osobnik_id] < best_score:
                        best_score = scores[osobnik_id]
                        best_weights = model.get_encoder().save_snapshot()
                        print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}. {"" if is_old else "NOWY"}')
                else:
                    scores[osobnik_id] = None
            if scores[osobnik_id] and scores[osobnik_id] <= best_score:
                best_score = scores[osobnik_id]
                best_weights = model.get_encoder().save_snapshot()
                print(f'Znaleziono lepszego osobnika {osobnik_id} o koszcie: {best_score}. {"" if is_old else "NOWY"}')
            logger.record(key="children_scores", osobnik_id=osobnik_id, final_score=scores[osobnik_id])
            osobnik_id += 1
            
        parent = model
        print(f'Generacja {generation}, najlepszy {best_score}')    
        logger.record(key="evolution", generation=generation, genome=parent.get_encoder().genome,  outputs=child.outputs, score=best_score)
        generation += 1
      
        
        
    for i, child in enumerate(children):
        model = AttentionModel(opts, child)
        evaluate(opts, model, logger, i)
        
    for i, child in enumerate(children):
        model = AttentionModel(opts, child)
        model.load_weights('/home/piotr/repos/magisterka/outputs/BASE_GENOME_CGP2D/snapshot_osobnik0_epoch50.pth')
        evaluate(opts, model, logger, i+10)
    
    # x_dim = 24
    # outputs = [48]
    # genome6 = [(7, 0), (3, 1, -1), (5, (49, 0, 1)), (3, 26, 0), (4, 3), (5, (51, 52)), (5, (54, 28)), (3, 7, 1), (2, 54), (4, 26), (1, 29), (1, 34), (2, 32), (2, 51), (4, 36), (2, 36), (5, (3, 60)), (3, 41, 1), (2, 40), (6, 16), (1, 49), (5, (14, 18)), (4, 35), (7, 40), (4, 0), (5, (0, 25)), (2, 26), (3, 27, 1), (7, 28), (3, 29, -1), (5, (27, 30)), (2, 31), (4, 32), (5, (32, 33)), (2, 34), (3, 35, 1), (7, 36), (3, 37, -1), (5, (35, 38)), (2, 39), (4, 40), (5, (40, 41)), (2, 42), (3, 43, 1), (7, 44), (3, 45, -1), (5, (43, 46)), (2, 47), (3, 0, -1), (1, 49), (6, 1), (6, 51), (4, 25), (1, 1), (2, 1), (7, 3), (4, 1), (6, 3), (3, 9, -1), (7, 54), (5, (6, 4)), (5, (33, 53)), (1, 50), (4, 62), (7, 63), (7, 41), (1, 30), (5, 6), (4, 1), (3, 26, 1), (6, 57), (4, 36)]
    # specie6 = AttentionModel(opts, CGP_Net(opts, x_dim, 3, outputs, genome=genome6))
    #export_cgp_to_graphviz(specie6.get_encoder().genes, "specie6")
    #evaluate(opts, specie6, logger, 0)
    return
    # opts.n_epochs = 10
    # opts.epoch_size = 12800
    export_cgp_to_graphviz(children[6].genes, "my_cgp")




if __name__ == "__main__":
    run(get_options())
 