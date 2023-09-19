import torch
from PIL import Image
from torch import autocast
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from typing import List, Type

from tqdm import tqdm

from cococrola.generate.image_generator import ImageGenerator


class DiffusersImageGenerator(ImageGenerator):
    def __init__(self, model_id, device, pipeline_type : Type[DiffusionPipeline] = StableDiffusionPipeline) -> None:
        self.pipe = pipeline_type.from_pretrained(model_id)
        self.pipe.to(device)

    def generate(self, prompt: str, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        with autocast(self.args.device):
            images += self.pipe(prompt, guidance_scale = guidance_scale, num_images_per_prompt = num_img).images
        return images