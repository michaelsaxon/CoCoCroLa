import click
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, AltDiffusionPipeline
from typing import Callable, List, Optional, Union
import os

# CUDA_VISIBLE_DEVICES=3 python generate_inspect_utils.py --output_dir samples_translated/altdiffusion --n_predictions 12 --model_id BAAI/AltDiffusion-m9 --split_batch 3

# external implementation of the first two steps of the generation pipeline for stable diffusion
# STEP 1: GET EMBEDDINGS
def get_text_embs(
    model: StableDiffusionPipeline,         
    prompt: Union[str, List[str]],
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
):

    if not isinstance(prompt, str) and not isinstance(prompt, list):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = model._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    return model._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

# external implementation of the first two steps of the generation pipeline for stable diffusion
# STEP 2: GET LATENTS
def get_latents(
    model: StableDiffusionPipeline,         
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    callback_steps: Optional[int] = 1,
):

    # 0. Default height and width to unet
    height = height or model.unet.config.sample_size * model.vae_scale_factor
    width = width or model.unet.config.sample_size * model.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(prompt, height, width, callback_steps)

    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = model._execution_device

    text_embeddings = get_text_embs(model, prompt, guidance_scale, negative_prompt, num_images_per_prompt)
    # 4. Prepare timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = model.unet.in_channels
    latents = model.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        text_embeddings.dtype,
        device,
        generator,
        latents,
    )

    return latents


'''
def get_generate():
    prompt = input("Prompt:\n").strip().lower()
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=7.5, num_images_per_prompt=9).images
    for i, im in enumerate(image):
        print(f"done generating {prompt.replace(' ','')}_{i}.png")
        im.save(f"playground/{prompt.replace(' ','')}_{i}.png")
'''

# some of these are hacks. I believe german and indonesian require some inflections depending on the word
# english, spanish, and german additionally probably need an indefinite article
# unclear if this is the correct colloquial chinese
LANG_PROMPT_BITS = {
    'en' : "a photograph of $$$",
    'es' : "una fotografía de $$$",
    'de' : "ein Foto von $$$",
    'zh' : "$$$照片",
    'ja' : "$$$の写真",
    #'kr' : Language.KR,
    'he' : " צילום של$$$",
    'id' : "foto $$$"
}


# BAAI/AltDiffusion-m9
# stabilityai/stable-diffusion-2
@click.command()
@click.option('--output_dir', default='samples_sd1-4')
@click.option('--n_predictions', default=9)
@click.option('--split_batch', default=1)
@click.option('--model_id', default="CompVis/stable-diffusion-v1-4")
@click.option('--input_csv', default="freq_lists_translated.csv")
@click.option('--start_line', default=1)
def main(output_dir, n_predictions, split_batch, model_id, input_csv, start_line):
    assert n_predictions % split_batch == 0
    model_id = model_id
    device = "cuda"

    if model_id == "BAAI/AltDiffusion-m9":
        pipe = AltDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    else:    
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe = pipe.to(device)

    os.makedirs(output_dir, exist_ok=True)

    prompts_base = open(f"frequencylist/{input_csv}", "r").readlines()
    index = prompts_base[0].strip().split(",")
    for line_idx in range(start_line, len(prompts_base)):
        line = prompts_base[line_idx]
        line_no = line_idx - 1
        line = line.strip().split(",")
        for idx in range(len(index)):
            # build a prompt based on the above templates from the 
            prompt = LANG_PROMPT_BITS[index[idx]].replace("$$$", line[idx])
            print(f"generating {index[idx]}:{line[0]}, '{line[idx]}'")
            images = []
            for _ in range(split_batch):
                with autocast("cuda"):
                    images += pipe(prompt, guidance_scale=7.5, num_images_per_prompt=int(n_predictions / split_batch)).images
            for i, im in enumerate(images):
                fname = f"{line_no}-{index[idx]}-{line[0]}-{i}.png"
                print(f"saving image {fname}...")
                im.save(f"{output_dir}/{fname}")


if __name__ == "__main__":
    main()