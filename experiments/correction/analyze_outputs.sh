#!/bin/bash

# first argument is GPU number
# second argument is model name
# SD1-4 SD2 SD2-1 AD DE2

model="${1:-SD2}"

cd ../../cococrola/analyze
CUDA_VISIBLE_DEVICES="$device" python evaluate_model.py \
    --input_csv ../../experiments/correction/concepts_zh_jp_revised.csv \
    --analysis_dir ../../results/correction_zh_jp_revised/"$model"/ \
    --eval_samples_file ../../experiments/correction/lines_zh_jp_revised.csv
