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
def main(output_dir, n_predictions, split_batch, model_id, input_csv, prompts_base, start_line, device):
    assert n_predictions % split_batch == 0

    if model_id == "BAAI/AltDiffusion-m9":
        generator = DiffusersImageGenerator(model_id, device, pipeline_type = diffusers.AltDiffusionPipeline)
    else:    
        generator = DiffusersImageGenerator(model_id, device, pipeline_type = diffusers.StableDiffusionPipeline)

    os.makedirs(output_dir, exist_ok=True)

    # put this into cococrola.utils.simple_csv
    prompts_base = open(input_csv, "r").readlines()
    index = prompts_base[0].strip().split(",")
    lang_prompt_templates = json.load(open(prompts_base, "r"))

    # this could be put into a new generate_save_images function
    # inputs prompts_base, lang_prompt_templates, generator, output_dir, start_line
    for line_idx in range(start_line, len(prompts_base)):
        line = prompts_base[line_idx]
        line_no = line_idx - 1
        line = line.strip().split(",")
        for idx in range(len(index)):
            # build a prompt based on the above templates from the 
            prompt = lang_prompt_templates[index[idx]].replace("$$$", line[idx])
            print(f"generating {index[idx]}:{line[0]}, '{line[idx]}'")
            images = generator.generate(prompt)
            for i, im in enumerate(images):
                fname = f"{line_no}-{index[idx]}-{line[0]}-{i}.png"
                print(f"saving image {fname}...")
                im.save(f"{output_dir}/{fname}")


if __name__ == "__main__":
    main()