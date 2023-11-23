#!/bin/bash

# first argument is GPU number
# second argument is model name
# SD1-4 SD2 SD2-1 AD DE2

device="${1:-0}"
model="${2:-SD2}"
bs="${3:-32}"
sb="${4:-1}"

cd ../../scripts
CUDA_VISIBLE_DEVICES="$device" python generate_images_to_dir.py \
    --input_csv ../experiments/correction/concepts_zh_jp_revised.csv \
    --output_dir ../results/correction_zh_jp_revised/"$model"/ \
    --model "$model" \
    --batch_size "$bs" \
    --split_batch "$sb"
