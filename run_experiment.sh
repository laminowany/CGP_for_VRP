#!/bin/bash

for i in {1..10}
do
    echo "Run $i"
    python3 src/run.py --mode evolve_transformer --x_dim 8 --y_dim 5 
done