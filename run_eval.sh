#!/bin/bash

GENOME_NAME=$1

python src/run.py \
    --run_name "${GENOME_NAME}" \
    --x_dim 8 \
    --mode genome_evaluation \
    --genome_name "$GENOME_NAME" \
    --graph_size 20

python src/run.py \
    --run_name "${GENOME_NAME}" \
    --x_dim 8 \
    --mode genome_evaluation \
    --genome_name "$GENOME_NAME" \
    --graph_size 50

python src/run.py \
    --run_name "${GENOME_NAME}" \
    --x_dim 8 \
    --mode genome_evaluation \
    --genome_name "$GENOME_NAME" \
    --graph_size 100