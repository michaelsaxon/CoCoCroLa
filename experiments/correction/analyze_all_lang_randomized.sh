#!/bin/bash

# first argument is GPU number
# second argument is folder name
# ../altdiffusion/ or something

device="${1:-0}"
model="$2"

for lang in es de zh ja he id; do
    CUDA_VISIBLE_DEVICES=$device python analyze_randomized.py \
        --random_csv randomized_"$lang".csv \
        --analysis_dir $model &
done