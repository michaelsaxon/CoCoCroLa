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

from cococrola.analyze.evaluate_model import get_image_embeddings, lang_cross_sim

@click.command()
@click.option('--analysis_dir', default='../results/correction_zh_jp_revised/')
@click.option('--num_samples', default=9)
@click.option('--main_language', default="en")
@click.option('--input_csv', type=str, default="../../benchmark/v0-1/concepts.csv")
def main(analysis_dir, num_samples, main_language, input_csv):

    if not os.path.exists(input_csv):
        raise Exception(f"Input CSV {input_csv} does not exist. Generate it first!")

    device = "cuda"
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    
    prompts_base = open(input_csv, "r").readlines()
    index = prompts_base[0].strip().split(",")

    out_lines_main_sim = [prompts_base[0]]
    out_lines_self_sim = [prompts_base[0]]
    out_lines_main_spec = [prompts_base[0]]
    

    # for line_no, line in enumerate(prompts_base[1:]):
    for line_no, line in enumerate(prompts_base):
        # line_no is a line in the csv, needs to be decremented by 1 for use as an fname
        results_dict = defaultdict(list)
        line = line.strip().split(",")
        
        # collect this languages embeddings
        for idx in range(len(index)):
            # build a prompt based on the above templates from the 
            fnames = [f"{analysis_dir}/{line_no - 1}-{index[idx]}-{line[0]}-{i}.png" for i in range(num_samples)]
            image_embedding = get_image_embeddings(processor, model, fnames)
            results_dict[index[idx]] = image_embedding
        
        # zero out if there's an error log for each word
        for language in index:
            if os.path.isfile(f"{analysis_dir}/{line_no}-{language}-{line[0]}-failure.log"):
                language_similarities[language] = "---"
                self_sims[language] = "---"
                inverse_specificity[language] = "---"
        

        print(f"{main_language} SIM " + line[0] + " " + str(language_similarities))
        print("self SIM " + line[0] + " " + str(self_sims))
        print("specific " + line[0] + " " + str(inverse_specificity))

        out_lines_main_sim.append(f"{line[0]}," + ",".join([str(language_similarities[language]) for language in index]) + "\n")
        out_lines_self_sim.append(f"{line[0]}," + ",".join([str(self_sims[language]) for language in index]) + "\n")
        out_lines_main_spec.append(f"{line[0]}," + ",".join([str(inverse_specificity[language]) for language in index]) + "\n")
        
    with open(f"{analysis_dir}/results_{main_language}.csv", "w") as f:
        f.writelines(out_lines_main_sim)

    with open(f"{analysis_dir}/results_self.csv", "w") as f:
        f.writelines(out_lines_self_sim)

    with open(f"{analysis_dir}/results_specific.csv", "w") as f:
        f.writelines(out_lines_main_spec)
