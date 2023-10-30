#CUDA_VISIBLE_DEVICES=2 python genimg_polyglot_decode.py --output_dir ../results/polyglot/SD2_en_900 --switch_prompt_step 900 --randomize
#CUDA_VISIBLE_DEVICES=3 python genimg_polyglot_decode.py --output_dir ../results/polyglot/SD2_en_800 --switch_prompt_step 800 --randomize
#CUDA_VISIBLE_DEVICES=4 python genimg_polyglot_decode.py --output_dir ../results/polyglot/SD2_en_700 --switch_prompt_step 700 --randomize
#CUDA_VISIBLE_DEVICES=5 python genimg_polyglot_decode.py --output_dir ../results/polyglot/SD2_en_600 --switch_prompt_step 600 --randomize

import os
import json
import random

import click

from cococrola.generate import models
from cococrola.generate.models.patches.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineSwapPromptSteps
from cococrola.generate.models.huggingface_diffusers import DiffusersImageGenerator



#from cococrola.utils.click_config import CommandWithConfigFile
from cococrola.utils.simple_csv import csv_to_index_elem_iterator
from cococrola.utils.save_imgs import save_imgs_to_dir

#@click.command(cls=CommandWithConfigFile('../config/generate.yaml'))
@click.command()
@click.option('--model', type=click.Choice(models.SUPPORTED_MODELS, case_sensitive=False), required=True, default="SD2")
@click.option('--output_dir', type=str, required=True)
@click.option('--num_img', type=int, default=9)
@click.option('--input_csv', type=str, default="../benchmark/v0-1/concepts.csv")
@click.option('--prompts_base', type=str, default="../benchmark/v0-1/prompts.json")
@click.option('--start_line', type=int, default=1)
@click.option('--device', type=str, default="cuda")
@click.option('--global_seed_fudge', type=int, default=0)
@click.option('--switch_prompt_step', type=int, default=25)
@click.option('--randomize', is_flag=True, help="Prevent the default behavior of using identical seeds across languages for each example")
@click.option('--ref_lang', type=str, default="en")
def main(model, output_dir, num_img, input_csv, prompts_base, start_line, device, global_seed_fudge, switch_prompt_step, randomize, ref_lang):
    pipeline_type = StableDiffusionPipelineSwapPromptSteps
    generator = DiffusersImageGenerator(models.MODEL_MAP_DIFFUSERS[model], device, pipeline_type)

    os.makedirs(output_dir, exist_ok=True)

    lang_prompt_templates = json.load(open(prompts_base, "r"))

    print("This version exclusively uses gold translations produced in the pipeline. Updated version will use translation api in the loop")

    # putting this in a different file would be overabstracting!
    for concept_number, concept_reflang, lang_code, concept_lang in csv_to_index_elem_iterator(input_csv, start_line, ref_lang=ref_lang):
        # 1. build a prompt based on the language-specific prompt templates
        starting_prompt = lang_prompt_templates[ref_lang].replace("$$$", concept_reflang)
        ending_prompt = lang_prompt_templates[lang_code].replace("$$$", concept_lang)

        # 2. do whatever state modification we want to do to the generator
        # fix the concept-level seed based on the csv line we're on (same starting seed for each lang)
        if randomize:
            # add the language code to the seed to make it different for each language
            generator.update_noise_generator(seed = concept_number + global_seed_fudge + int(lang_code, 36))
        else:
            generator.update_noise_generator(seed = concept_number + global_seed_fudge)

        # 3. generate the images
        images = generator.generate_steps_change(starting_prompt, changes_list = (ending_prompt, switch_prompt_step), num_img = num_img)
 
        # 4. save_the images
        fname_base = f"{output_dir}/{concept_number}-{lang_code}-{concept_reflang}"
        save_imgs_to_dir(images, fname_base)


if __name__ == "__main__":
    main()