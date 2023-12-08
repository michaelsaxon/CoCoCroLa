"""
CUDA_VISIBLE_DEVICES="$device" python evaluate_model.py \
    --analysis_dir ../../results/correction_zh_jp_revised/orig_"$model"/ \
    --fingerprint_selection_count 25 \
    --eval_samples_file ../../experiments/correction/lines_zh_jp_revised.csv    
"""
import click
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPVisionModel
from collections import defaultdict
import torch.nn.functional as F
import random
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

from cococrola.analyze.evaluate_model import get_image_embeddings, compare_by_lang


# fname format is idx(line_no - 1)-lang-word-img_num.png
def gen_fnames(img_idx, lang, word_in_english, num_samples):
    return [f"{img_idx}-{lang}-{word_in_english}-{i}.png" for i in range(num_samples)]

def get_fnames_from_lang_word(lines, lang, word_in_lang, num_samples):
    index = lines[0].strip().split(",")
    lang_idx = index.index(lang)
    line_no = [line.strip().split(",")[lang_idx] for line in lines[1:]].index(word_in_lang) + 1
    word_english = lines[line_no].strip().split(",")[0]
    return [f"{line_no - 1}-{lang}-{word_english}-{i}.png" for i in range(num_samples)]

@click.command()
@click.option('--analysis_dir', default='../results/correction_zh_jp_revised/')
@click.option('--num_samples', default=9)
@click.option('--input_csv', type=str, default="randomized_es.csv")
def main(analysis_dir, num_samples, input_csv):
    # HACK get language from fname, model from analysis_dir only works if you ran the randomize language script first.
    lang = input_csv.split(".")[0].split("_")[-1]
    model_name = analysis_dir.strip("/").split("/")[-1].strip("samples_")
    # unlike the main analysis script, in this case we process each language serpately (they are randomized separately after all)

    if not os.path.exists(input_csv):
        raise Exception(f"Input CSV {input_csv} does not exist. Generate it first!")

    device = "cuda"
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    
    lines = open(input_csv, "r").readlines()

    outlines = [lines[0].strip() + f",{lang}_before,{lang}_after\n"]    

    # for line_no, line in enumerate(prompts_base[1:]):
    for img_idx, line in enumerate(lines[1:]):
        results_dict_before = defaultdict(list)
        results_dict_after = defaultdict(list)
        # line_no is a line in the csv, needs to be decremented by 1 for use as an fname
        line = line.strip().split(",")
        

        fnames_en = gen_fnames(img_idx, "en", line[0], num_samples)
        fnames_lang_before = get_fnames_from_lang_word(lines, lang, line[1], num_samples)
        fnames_lang_after = gen_fnames(img_idx, lang, line[0], num_samples)

        en_embeddings = get_image_embeddings(processor, model, fnames_en)

        results_dict_before["en"] =  en_embeddings
        results_dict_after["en"] =  en_embeddings
        results_dict_before[lang] = get_image_embeddings(processor, model, fnames_lang_before)
        results_dict_after[lang] = get_image_embeddings(processor, model, fnames_lang_after)

        cross_sim_before = compare_by_lang(results_dict_before, main_lang="en")[lang]
        cross_sim_after = compare_by_lang(results_dict_before, main_lang="en")[lang]

        outlines.append(f"{line.strip()},{cross_sim_before},{cross_sim_after}\n")

        
    with open(f"cccl_score_{model_name}_{lang}.csv", "w") as f:
        f.writelines(outlines)
