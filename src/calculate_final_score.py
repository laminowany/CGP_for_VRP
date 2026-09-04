



import ast
import csv
import os
from pathlib import Path

import torch

from learning.attention_model import AttentionModel
from learning.cgp import CGP_Net
from utils.graph import export_cgp_to_graphviz
from utils.logger import Logger
from utils.process import get_options, initial_setup
from utils.training import produce_transformer_genome, reset_seeds, validate


if __name__ == "__main__":
    opts = get_options()
    initial_setup(opts)
    logger = Logger(opts)
    
    # models = [("CGP5", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260825T183307_FULL_EVAL_CGP5_CVRP10"),
    #           ("CGP7", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260825T221244_FULL_EVAL_CGP7_CVRP10"),
    #           ("CGP8", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260826T014718_FULL_EVAL_CGP8_CVRP10"),
    #           ("RND2", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260825T085729_FULL_EVAL_RND2_CVRP10"),
    #           ("RND4", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260825T124528_FULL_EVAL_RND4_CVRP10"),
    #           ("RND6", "/home/piotr/outpusNEW/EXP1_CVRP10/run_20260825T162939_FULL_EVAL_RND6_CVRP10"),
    #           ]

    
    # opts.x_dim = 15
    # for models_metadata in models:
    #     path = Path(models_metadata[1])
    #     checkpoint = path / "epoch-99.pt"
    #     with open(path / "genomes.csv", newline="") as f:
    #         genome = ast.literal_eval(next(csv.DictReader(f))["genome"])
            
    #     reset_seeds(opts)
    #     encoder = CGP_Net(opts, (genome))
    #     model = AttentionModel(opts, encoder)
    #     export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"candidate_{models_metadata[0]}"), only_active=True, paper_style=True)
    #     checkpoint = torch.load(checkpoint, map_location=opts.device, weights_only=False)
    #     model.load_state_dict(checkpoint["model"])
    #     score = validate(model, opts.test_set, opts)
    #     logger.record(key="scores", score=score)
    #     print(f"Final score {models_metadata[0]} {score}")
    #     # encoder.print_active_parameters()

    # print(f"Active encoder parameters: {encoder.count_active_parameters():,}")
    
    
    
    
    
    # models = [("RND1", "/home/piotr/outpusNEW/EVO2/run_20260824T155242_RND1"),
    #           ("RND2", "/home/piotr/outpusNEW/EVO2/run_20260824T174625_RND2"),
    #           ("RND3", "/home/piotr/outpusNEW/EVO2/run_20260824T194216_RND3"),
    #           ("RND4", "/home/piotr/outpusNEW/TRANS1/run_20260824T155159_RND4"),
    #           ("RND5", "/home/piotr/outpusNEW/TRANS1/run_20260824T174616_RND5"),
    #           ("RND6", "/home/piotr/outpusNEW/TRANS1/run_20260824T193517_RND6"),
    #           ("RND7", "/home/piotr/outpusNEW/TRANS1/run_20260824T212841_RND7"),
    #           ("RND8", "/home/piotr/outpusNEW/TRANS1/run_20260824T232339_RND8"),
    #           ("RND9", "/home/piotr/outpusNEW/TRANS1/run_20260825T011520_RND9"),
    #           ("RND10", "/home/piotr/outpusNEW/TRANS1/run_20260825T031031_RND10"),
              
    #           ]
    # models = [("CGP1", "/home/piotr/outpusNEW/EVO2/run_20260824T220550_CGP1"),
    #           ("CGP2", "/home/piotr/outpusNEW/EVO2/run_20260825T000344_CGP2"),
    #           ("CGP3", "/home/piotr/outpusNEW/EVO2/run_20260825T015503_CGP3"),
    #           ("CGP4", "/home/piotr/outpusNEW/EVO2/run_20260825T035906_CGP4"),
    #           ("CGP5", "/home/piotr/outpusNEW/EVO2/run_20260825T060638_CGP5"),
    #           ("CGP6", "/home/piotr/outpusNEW/EVO2/run_20260825T080442_CGP6"),
    #           ("CGP7", "/home/piotr/outpusNEW/EVO2/run_20260825T095815_CGP7"),
    #           ("CGP8", "/home/piotr/outpusNEW/EVO2/run_20260825T115414_CGP8"),
    #           ("CGP9", "/home/piotr/outpusNEW/EVO2/run_20260825T134552_CGP9"),
    #           ("CGP10", "/home/piotr/outpusNEW/EVO2/run_20260825T153955_CGP10"),
              
    #           ]
    
    models = [("EVO1", "/home/piotr/outpusNEW/EXP2/run_20260826T151741_EVOLVE_TRANS_1"),
              ("EVO2", "/home/piotr/outpusNEW/EXP2/run_20260826T151900_EVOLVE_TRANS"),
              ("EVO3", "/home/piotr/outpusNEW/EXP2/run_20260826T170019_EVOLVE_TRANS"),
              ("EVO4", "/home/piotr/outpusNEW/EXP2/run_20260826T171315_EVOLVE_TRANS_4"),
              ("EVO5", "/home/piotr/outpusNEW/EXP2/run_20260826T184440_EVOLVE_TRANS"),
              ("EVO6", "/home/piotr/outpusNEW/EXP2/run_20260826T191111_EVOLVE_TRANS_6"),
              ("EVO7", "/home/piotr/outpusNEW/EXP2/run_20260826T203028_EVOLVE_TRANS"),
              ("EVO8", "/home/piotr/outpusNEW/EXP2/run_20260826T210836_EVOLVE_TRANS"),
              ("EVO9", "/home/piotr/outpusNEW/EXP2/run_20260826T220918_EVOLVE_TRANS"),
              ("EVO10", "/home/piotr/outpusNEW/EXP2/run_20260826T234938_EVOLVE_TRANS_10"),
              
              ]
    opts.x_dim = 24
    trasnformer = produce_transformer_genome(opts)
    print(trasnformer)
    encoder = CGP_Net(opts, trasnformer)
    export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"trans1"), only_active=True, paper_style=False)
    exit()
    for model_metadata in models:
         # 1. Find the last best_id from budget.csv
        with open(Path(model_metadata[1]) / "budget.csv", newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise ValueError(f"No rows found in {model_metadata[1]  / 'budget.csv'}")

        best_id = int(rows[-1]["best_id"])

        # 2. Find this id in candidates.csv
        with open(Path(model_metadata[1])  / "candidates.csv", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if int(row["id"]) == best_id:
                    genome = ast.literal_eval(row["genome"])
        encoder = CGP_Net(opts, (genome))
        export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, f"{model_metadata[0]}"), only_active=True, paper_style=False)