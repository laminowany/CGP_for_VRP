#!/bin/bash

for i in {1..10}
do
    python3 src/run.py --mode evolve_transformer --x_dim 8 --y_dim 5 --n_epochs 10 --epoch_size 12800 \
    --validation_set_path data/dataset_10CVRP_seed_3232.pt --seed -1  --no_progress_bar --graph_size 10 --no_save_model --run_name EVOLVE_TRANS  
done
