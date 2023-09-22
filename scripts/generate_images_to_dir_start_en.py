import os
import json

import click

from cococrola.generate import models
from cococrola.utils.click_config import CommandWithConfigFile


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
@click.option('--switch_off_english_step', type=int, default=25)
def main(model, output_dir, num_img, input_csv, prompts_base, start_line, device, global_seed_fudge, switch_off_english_step):
    generator = models.get_generator(model, device)

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
        english_prompt = lang_prompt_templates["en"].replace("$$$", line[0])
        for lang_idx in range(len(index)):
            # build a prompt based on the language-specific prompt templates
            prompt = lang_prompt_templates[index[lang_idx]].replace("$$$", line[lang_idx])
            print(f"generating {index[lang_idx]}:{line[0]}, '{line[lang_idx]}'")
            # fix the concept-level seed based on the csv line we're on (same starting seed for each lang)
            generator.update_noise_generator(seed = line_no + global_seed_fudge)
            images = generator.generate_prompt_change(english_prompt, prompt, prompt_reset_step = switch_off_english_step, num_img = num_img)
            for i, im in enumerate(images):
                fname = f"{line_no}-{index[lang_idx]}-{line[0]}-{i}.png"
                print(f"saving image {fname}...")
                im.save(f"{output_dir}/{fname}")


if __name__ == "__main__":
    main()