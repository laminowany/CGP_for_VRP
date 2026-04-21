

from utils.process import get_options
from learning.cgp import GenomeFactory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from run import evaluate
from src.run import evaluate, get_options

class DummyLogger:
    def __init__(self):
        pass
     
    def record(self, key="epochs", **kwargs):
        pass

    def to_dataframe(self, key):
        pass

CHROMOSOMES = [[(5, -1), (1,), (5, -2), (7,), (9,), (6, -1), (4,), (5, -8), (1,), (4,)],]

def test_validity_of_chromosomes():
    opts = get_options()
    opts.n_epochs = 1
    opts.epoch_size = 12800
    for chromosome in CHROMOSOMES:
        genome = GenomeFactory().produce_genome(chromosome)
        evaluate(opts, genome, logger=DummyLogger())