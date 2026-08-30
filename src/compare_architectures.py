


from dataclasses import dataclass
import os

from learning.cgp import CGP_Net
from utils.graph import export_cgp_to_graphviz
from utils.logger import Logger
from utils.process import get_options, initial_setup
from utils.training import evaluate, reset_seeds, verify_sanity



@dataclass
class Architecture:
    name: str
    x_dim: int
    y_dim: int
    genome: list
    
    
    
def compare_architectures(opts, logger):
    architectures = [ 
                     Architecture("EVO1_orig", 8, 5, [None,
                        (7, 0), (2, 25), (4, 34), (3, 35, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31), 
                        (4, 0), (4, 9), (4, 2), (3, 19, -1), (3, 28, -1), (1, 37), (7, 14), (6, 39), 
                        (5, 0),  (5, (25, 33, 1)), (2, 18), (3, 19, 1), (2, 28), (2, 29), (1, 38), (5, 31), 
                        (5, 0), (4, 1),  (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15), 
                        (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39), (5, 32)]),
                    # Architecture("EVO1_simplified", 8, 5, [None,
                    #     (7, 0), (2, 25), (4, 34), (3, 35, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31),     # 1-8
                    #     (4, 0), (4, 9), (4, 2), (3, 19, -1), (3, 28, -1), (1, 37), (7, 14), (6, 39),        # 9 - 16
                    #     (1, 0),  (5, (25, 33, 1)), (2, 18), (3, 18, 1), (2, 28), (2, 29), (1, 38), (5, 31), # 17 - 24
                    #     (1, 0), (4, 1),  (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15),              # 25 - 32
                    #     (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39),           # 33 - 41
                    #     (5, 32)]),
                    Architecture("EVO1", 5, 3, [None,
                        (7, 0), None, None, None, None,     
                        (1, 0), (5, (1, 0, 11)), (2, 7), (7, 8), (2, 9),
                        (4, 0), None, None, None, None,
                        (2, 9)]),
                    Architecture("EVO2", 5, 2, [None,
                        None, None, None, None, None,     
                        (4, 0), (5, (0, 6)), (2, 7), (7, 8), (2, 9),
                        (2, 9)]),
                    Architecture("EVO10", 5, 2, [None,
                        (2, 0), None, None, None, None,     
                        (4, 0), (5, (1, 6)), (3, 7 ,0), (7, 8), (2, 9),
                        (2, 9)]) 
                    ]

    opts.n_epochs = 3
    opts.epoch_size = 1280000
    opts.no_progress_bar = True
    opts.no_save_model = True
    
    for arch in architectures:
        reset_seeds(opts)
        opts.x_dim = arch.x_dim
        opts.y_dim = arch.y_dim
        genome = arch.genome
        encoder = CGP_Net(opts, genome)
        logger.record(key="genomes", name = opts.genome_name, genome = genome)
        export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, arch.name), only_active=True, paper_style=True)
        #result = evaluate(opts, logger, encoder)
        #print(result.scores)
        #encoder.print_active_parameters()
        print(f"Active encoder parameters: {encoder.count_active_parameters():,}")

if __name__ == "__main__":
    opts = get_options()
    initial_setup(opts)
    logger = Logger(opts)
    #verify_sanity(opts, logger)
    
    compare_architectures(opts, logger)
        
        
    