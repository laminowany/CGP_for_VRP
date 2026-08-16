#!/bin/bash
python3 src/run.py --mode full_evaluation --genome_path ./exp2_genes/arch1 --id 435 --graph_size 20 -x_dim 8 --y_dim 5 
python3 src/run.py --mode full_evaluation --genome_path ./exp2_genes/arch2 --id 560 --graph_size 20 -x_dim 8 --y_dim 5 
python3 src/run.py --mode full_evaluation --genome_path ./exp2_genes/arch3 --id 511 --graph_size 20 -x_dim 8 --y_dim 5 