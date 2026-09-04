


from dataclasses import dataclass
import os

from learning.cgp import CGP_Net
from utils.graph import export_cgp_to_graphviz
from utils.logger import Logger
from utils.process import get_options, initial_setup
from utils.training import evaluate, produce_transformer_genome, reset_seeds, verify_sanity



@dataclass
class Architecture:
    name: str
    x_dim: int
    y_dim: int
    genome: list
    
    
    
def compare_architectures(opts, logger):
    architectures = [ 
                    #  Architecture("EVO1_orig", 8, 5, [None,
                    #     (7, 0), (2, 25), (4, 34), (3, 35, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31), 
                    #     (4, 0), (4, 9), (4, 2), (3, 19, -1), (3, 28, -1), (1, 37), (7, 14), (6, 39), 
                    #     (5, 0),  (5, (25, 33, 1)), (2, 18), (3, 19, 1), (2, 28), (2, 29), (1, 38), (5, 31), 
                    #     (5, 0), (4, 1),  (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15), 
                    #     (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39), (5, 32)]),
                    # Architecture("EVO1_simplified", 8, 5, [None,
                    #     (7, 0), (2, 25), (4, 34), (3, 35, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31),     # 1-8
                    #     (4, 0), (4, 9), (4, 2), (3, 19, -1), (3, 28, -1), (1, 37), (7, 14), (6, 39),        # 9 - 16
                    #     (1, 0),  (5, (25, 33, 1)), (2, 18), (3, 18, 1), (2, 28), (2, 29), (1, 38), (5, 31), # 17 - 24
                    #     (1, 0), (4, 1),  (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15),              # 25 - 32
                    #     (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39),           # 33 - 41
                    #     (5, 32)]),
                    # Architecture("EVO1", 5, 3, [None,
                    #     (7, 0), None, None, None, None,     
                    #     (1, 0), (5, (1, 0, 11)), (2, 7), (7, 8), (2, 9),
                    #     (4, 0), None, None, None, None,
                    #     (2, 9)]),
                    # Architecture("EVO2", 5, 2, [None,
                    #     None, None, None, None, None,     
                    #     (4, 0), (5, (0, 6)), (2, 7), (7, 8), (2, 9),
                    #     (2, 9)]),
                    # Architecture("EVO10", 5, 2, [None,
                    #     (2, 0), None, None, None, None,     
                    #     (4, 0), (5, (1, 6)), (3, 7 ,0), (7, 8), (2, 9),
                    #     (2, 9)]),
                    # Architecture("TRANS1_orig", 8, 2, [None,
                    #     (1, 0), (1, 1), (1, 2), (1, 11), (1, 4), (1, 5), (1, 6), (1, 7),
                    #     (4, 0), (5, (1, 9)), (2, 10), (3, 11, 1), (7, 12), (3, 13, -1), (5, (6, 14)), (2, 15), 
                    #     (5, 16)]),
                    # Architecture("TRANS1", 8, 2, [None,
                    #     (1, 0), (1, 1), (1, 2), (1, 11), (1, 4), (1, 5), (1, 6), (1, 7),
                    #     (4, 0), (5, (0, 9)), (2, 10), (3, 11, 1), (7, 12), (3, 13, -1), (5, (11, 14)), (2, 15), 
                    #     (2, 15)]),
                    # Architecture("EVO_II_3_orig", 8, 5,
                    #     [None, 
                    #     (5, 0), (1, 17), (7, 18), (5, 27), (7, 20), (1, 21), (5, 22), (7, 7), (4, 0), (3, 33, 1), 
                    #     (4, 2), (6, 19), (1, 4), (7, 5), (2, 38), (6, 15), (2, 0), (5, (1, 9)), (5, 26), (3, 3, 1), 
                    #     (1, 12), (3, 5, -1), (4, 30), (2, 7), (7, 0), (6, 25), (3, 34, -1), (2, 35), (1, 28), (2, 21),
                    #     (7, 38), (7, 23), (2, 0), (7, 25), (1, 34), (5, 3), (4, 28), (4, 13), (2, 30), (7, 23),
                    #     (5, 24)]
                    #     ),
                #     Architecture("EVO3", 8, 5,
                #         [None, 
                #         (5, 0), (1, 17), (7, 18), (5, 27), (7, 20), (1, 21), (5, 22), (2, 22), 
                #         (4, 0), (3, 33, 1), (4, 2), (6, 19), (1, 4), (7, 5), (2, 38), (6, 15), 
                #         (2, 0), (5, (0, 9)), (5, 26), (3, 3, 1),  (1, 12), (3, 5, -1), (4, 30), (2, 7), 
                #         (7, 0), (6, 25), (3, 34, -1), (2, 35), (1, 28), (2, 21), (7, 38), (7, 23), 
                #         (2, 0), (7, 25), (1, 34), (5, 3), (4, 28), (4, 13), (2, 30), (7, 23),
                #         (2, 22)]
                #         ),
                #      Architecture("EVO4", 8, 5,
                #         [None, 
                #             (5, 0), (3, 1, 1), (1, 2), (6, 19), (5, 36), (4, 5), (3, 14, 0), (1, 39), 
                #             (1, 0), (3, 1, 1), (7, 2), (2, 35), (2, 12), (7, 13), (6, 14), (7, 7), 
                #             (3, 0, 0), (2, 0), (4, 18), (1, 19), (6, 19), (6, 37), (2, 38), (7, 15),
                #             (5, 0), (3, 9, 0), (7, 34), (7, 27), (1, 28), (6, 37), (4, 6), (3, 7, -1), 
                #             (1, 0), (2, 0), (6, 34), (4, 3), (3, 28, -1), (2, 21), (2, 38), (5, 39), 
                #             (5, (39, 24))]
                #         ),
                #  Architecture("EVO6", 8, 5,
                #         [None, 
                #          (2, 0), (5, (33, 25)), (5, 18), (6, 19), (5, 36), (2, 35), (1, 6), (3, 7, 0), 
                #          (2, 0), (4, 17), (7, 18), (3, 27, -1), (5, 36), (7, 35), (7, 6), (3, 15, 0),
                #          (4, 0), (5, (9, 17)), (5, 18), (3, 18, 1), (7, 20), (3, 21, -1), (5, (35, 22)), (2, 23),
                #          (1, 0), (4, 17), (2, 26), (3, 27, -1), (1, 20), (7, 29), (1, 38), (6, 15), 
                #          (3, 0, 1), (3, 25, 0), (7, 26), (1, 35), (2, 28), (6, 5), (2, 6), (6, 23), 
                #          (5, (24, 16))]
                #         ),
                                  # Architecture("EVO1_simplified", 8, 5, [None,
                                     #     (7, 0), (2, 25), (4, 34), (3, 35, -1), (5, (4, 28)), (5, 21), (1, 22), (1, 31),     # 1-8
                                     #     (4, 0), (4, 9), (4, 2), (3, 19, -1), (3, 28, -1), (1, 37), (7, 14), (6, 39),        # 9 - 16
                                     #     (1, 0),  (5, (25, 33, 1)), (2, 18), (3, 18, 1), (2, 28), (2, 29), (1, 38), (5, 31), # 17 - 24
                                     #     (1, 0), (4, 1),  (4, 26), (1, 19), (6, 28), (5, 29), (2, 22), (2, 15),              # 25 - 32
                                     #     (4, 0), (5, 25), (3, 34, 0), (7, 35), (5, 28), (4, 21), (2, 14), (6, 39),           # 33 - 41
                                     #     (5, 32)]),
                        Architecture("TEST", 2,2,
                                        [None, 
                                         (2, 0), (2, 1), (2, 0), (3, 1, 1), (5, 4)]
                                        ),
                    ]
    opts.x_dim = 8
    opts.y_dim = 5
    architectures.append(Architecture("TRANSFORMER", 8, 5, produce_transformer_genome(opts)))
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
        export_cgp_to_graphviz(encoder.genes, opts, os.path.join(opts.save_dir, arch.name), only_active=False, paper_style=False)
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
        
        
    