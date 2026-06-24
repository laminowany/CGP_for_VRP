#!/bin/bash

for i in {1..10}
do
    echo "Run $i"
    python3 src/run.py --random_search
done