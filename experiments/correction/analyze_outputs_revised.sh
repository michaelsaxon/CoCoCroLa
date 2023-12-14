#!/bin/bash

# first argument is GPU number
# second argument is model name
# SD1-4 SD2 SD2-1 AD DE2

device="${1:-0}"
model="${2:-SD2}"
langs="${3:-zh_jp}"

cd ../../cococrola/analyze
CUDA_VISIBLE_DEVICES="$device" python evaluate_model.py \
    --analysis_dir ../../results/correction_"$langs"_revised/orig_"$model"/ \
    --fingerprint_selection_count 25 \
    --eval_samples_file ../../experiments/correction/lines_"$langs"_revised.csv


# test and see