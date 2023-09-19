import click
import jax
import jax.numpy as jnp

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel


from flax.jax_utils import replicate

from functools import partial


import random

from dalle_mini import DalleBartProcessor



from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
#from tqdm.notebook import trange

import os

# dalle-mega
DALLE_MODELS = {
    "mega" : "dalle-mini/dalle-mini/mega-1-fp16:latest",
    "mini" : "dalle-mini/dalle-mini/mini-1:v0"
}
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"




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


@click.command()
@click.option('--output_dir', default='samples_demega')
@click.option('--n_predictions', default=9)
@click.option('--model_size', default="mega")
@click.option('--input_csv', default="freq_lists_translated.csv")
def main(output_dir, n_predictions, model_size, input_csv):

    # check how many devices are available
    jax.local_device_count()


    # Load dalle-mini
    model, params = DalleBart.from_pretrained(
        DALLE_MODELS[model_size], revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )

    # Load VQGAN
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )


    params = replicate(params)
    vqgan_params = replicate(vqgan_params)


    # create a random key
    seed = random.randint(0, 2**32 - 1)
    key = jax.random.PRNGKey(seed)


    processor = DalleBartProcessor.from_pretrained(DALLE_MODELS[model_size], revision=DALLE_COMMIT_ID)

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0


    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
    ):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def p_decode(indices, params):
        return vqgan.decode_code(indices, params=params)

    os.makedirs(output_dir, exist_ok=True)



    prompts_base = open(f"frequencylist/{input_csv}", "r").readlines()
    index = prompts_base[0].strip().split(",")
    for line_no, line in enumerate(prompts_base[1:]):
        line = line.strip().split(",")
        for idx in range(len(index)):

            print(f"generating {index[idx]}:{line[0]}, '{line[idx]}'")

            prompt = LANG_PROMPT_BITS[index[idx]].replace("$$$", line[idx])

            prompts = [prompt] * n_predictions
            tokenized_prompts = processor(prompts)
            tokenized_prompt = replicate(tokenized_prompts)

            # get a new key
            key, subkey = jax.random.split(key)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for i, decoded_img in enumerate(decoded_images):
                fname = f"{line_no}-{index[idx]}-{line[0]}-{i}.png"
                print(f"saving image {fname}...")
                im = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                im.save(f"{output_dir}/{fname}")


if __name__ == "__main__":
    main()