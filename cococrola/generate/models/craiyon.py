import click
import jax
import jax.numpy as jnp

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel

from flax.jax_utils import replicate

from functools import partial

from typing import List, Optional, Union

from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
#from tqdm.notebook import trange

from cococrola.generate.models.image_generator import ImageGenerator



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

# DALLE mini/mega has been renamed to Craiyon
class CraiyonImageGenerator(ImageGenerator):
    def __init__(
            self, 
            device : str, 
            model_size : str = "mega", 
            seed : int = 0
            ) -> None:
        jax.local_device_count()

        # Load dalle-mini
        self.model, params = DalleBart.from_pretrained(
            DALLE_MODELS[model_size], revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        self.vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        self.params = replicate(params)
        self.vqgan_params = replicate(vqgan_params)

        # create a random key
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODELS[model_size], revision=DALLE_COMMIT_ID)

        # model inference
        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
            tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.model.generate(
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
            return self.vqgan.decode_code(indices, params=params)

        self.p_generate = p_generate
        self.p_decode = p_decode


    def update_noise_generator(self, seed : int = 0) -> None:
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)


    def generate(
            self, 
            prompt: str, 
            cond_scale : float = 10.0, 
            num_img: int = 9, 
            gen_top_k : Optional[int] = None,
            gen_top_p : Optional[int] = None,
            temperature : Optional[int] = None
        ) -> List[Image.Image]:
        prompts = [prompt] * num_img
        tokenized_prompts = self.processor(prompts)
        tokenized_prompt = replicate(tokenized_prompts)

        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = self.p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            self.params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        # decode images
        decoded_images = self.p_decode(encoded_images, self.vqgan_params)
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        return [Image.fromarray(np.asarray(decoded_image * 255, dtype=np.uint8)) for decoded_image in decoded_images]
