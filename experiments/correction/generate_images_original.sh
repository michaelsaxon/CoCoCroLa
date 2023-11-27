#!/bin/bash

# first argument is GPU number
# second argument is model name
# SD1-4 SD2 SD2-1 AD DE2

device="${1:-0}"
model="${2:-SD2}"
sb="${3:-1}"

cd ../../scripts
CUDA_VISIBLE_DEVICES="$device" python generate_images_to_dir.py \
    --output_dir ../results/correction_zh_jp_revised/orig_"$model"/ \
    --model "$model" \
    --split_batch "$sb" \
    --eval_samples_file ../experiments/correction/lines_zh_jp_revised.csv

