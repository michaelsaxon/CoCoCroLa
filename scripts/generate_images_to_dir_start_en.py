import os
import json

import click

from cococrola.generate import models, generate_split_batch
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
@click.option('--switch_lang_step', type=int, default=950)
@click.option('--start_lang', type=str, default="en")
@click.option('--split_batch', type=int, default=1)
def main(model, output_dir, num_img, input_csv, prompts_base, start_line, device, global_seed_fudge, switch_lang_step, start_lang, split_batch):
    generator = models.get_generator(model, device)

    os.makedirs(output_dir, exist_ok=True)

    lang_prompt_templates = json.load(open(prompts_base, "r"))



    # putting this in a different file would be overabstracting!
    for concept_number, concept_reflang, lang_code, concept_lang in csv_to_index_elem_iterator(input_csv, start_line, ref_lang=start_lang):
        # 1. build a prompt based on the language-specific prompt templates
        first_prompt = lang_prompt_templates[start_lang].replace("$$$", concept_reflang)
        second_prompt = lang_prompt_templates[lang_code].replace("$$$", concept_lang)

        # 2. do whatever state modification we want to do to the generator
        # fix the concept-level seed based on the csv line we're on (same starting seed for each lang)
        generator.update_noise_generator(seed = concept_number + global_seed_fudge)

        # 3. generate the images
        images = generate_split_batch(
            generate_function = generator.generate_prompt_change, 
            num_img = num_img, 
            split_batch = split_batch,
            prompt = first_prompt, 
            second_prompt = second_prompt,
            prompt_reset_step = switch_lang_step
        )
 
        # 4. save_the images
        fname_base = f"{output_dir}/{concept_number}-{lang_code}-{concept_reflang}"
        save_imgs_to_dir(images, fname_base)


if __name__ == "__main__":
    main()