from cococrola.generate.models.huggingface_diffusers import DiffusersImageGenerator
from cococrola.utils.click_config import CommandWithConfigFile
import click
import os
import json

import diffusers

#@click.command(cls=CommandWithConfigFile('../config/generate.yaml'))
@click.command()
@click.option('--output_dir')
@click.option('--n_predictions', default=9)
@click.option('--split_batch', default=1)
@click.option('--model_id', default="CompVis/stable-diffusion-v1-4")
@click.option('--input_csv', default="../benchmark/v0-1/concepts.csv")
@click.option('--prompts_base', default="../benchmark/v0-1/prompts.json")
@click.option('--start_line', default=1)
@click.option('--device', default="cuda")
@click.option('--global_seed_fudge', default=0)
def main(output_dir, n_predictions, split_batch, model_id, input_csv, prompts_base, start_line, device, global_seed_fudge):
    assert n_predictions % split_batch == 0

    if model_id == "BAAI/AltDiffusion-m9":
        generator = DiffusersImageGenerator(model_id, device, pipeline_type = diffusers.AltDiffusionPipeline)
    else:    
        generator = DiffusersImageGenerator(model_id, device, pipeline_type = diffusers.StableDiffusionPipeline)

    os.makedirs(output_dir, exist_ok=True)

    # put this into cococrola.utils.simple_csv
    words = open(input_csv, "r").readlines()
    index = words[0].strip().split(",")
    lang_prompt_templates = json.load(open(prompts_base, "r"))

    # this could be put into a new generate_save_images function
    # inputs prompts_base, lang_prompt_templates, generator, output_dir, start_line
    for line_idx in range(start_line, len(words)):
        line = words[line_idx]
        line_no = line_idx - 1
        line = line.strip().split(",")
        for lang_idx in range(len(index)):
            # build a prompt based on the language-specific prompt templates
            prompt = lang_prompt_templates[index[lang_idx]].replace("$$$", line[lang_idx])
            print(f"generating {index[lang_idx]}:{line[0]}, '{line[lang_idx]}'")
            # fix the concept-level seed based on the csv line we're on (same starting seed for each lang)
            generator.update_noise_generator(seed = line_no + global_seed_fudge)
            images = generator.generate(prompt)
            for i, im in enumerate(images):
                fname = f"{line_no}-{index[lang_idx]}-{line[0]}-{i}.png"
                print(f"saving image {fname}...")
                im.save(f"{output_dir}/{fname}")


if __name__ == "__main__":
    main()