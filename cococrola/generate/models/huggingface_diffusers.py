import warnings

import torch
from PIL import Image
from torch import autocast
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from typing import List, Type, Optional, Union, Tuple

from tqdm import tqdm

from cococrola.generate.models.image_generator import ImageGenerator
from cococrola.generate.models.patches.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMidwayPatch
from cococrola.generate.models.patches.diffusers.pipeline_alt_diffusion import AltDiffusionPipelineMidwayPatch


class DiffusersImageGenerator(ImageGenerator):
    def __init__(
            self, 
            model_id : str, 
            device : str, 
            pipeline_type : Type[DiffusionPipeline] = StableDiffusionPipeline,
            seed :Optional[int] = None
            ) -> None:
        self.pipe = pipeline_type.from_pretrained(model_id)
        self.device = device
        self.pipe.to(device)
        self.noise_generator = torch.Generator(device = device)
        if seed != None:
            self.noise_generator = self.noise_generator.manual_seed(seed)

    def update_noise_generator(self, seed : int = 0) -> None:
        if self.noise_generator == None:
            raise UserWarning("Updating noise generator that hasn't been instantiated. Nothing happened.")
        else:
            self.noise_generator = self.noise_generator.manual_seed(seed)

    def generate(self, prompt: str, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        images = []
        with autocast(self.device):
            images += self.pipe(
                prompt, 
                guidance_scale = guidance_scale, 
                num_images_per_prompt = num_img, 
                generator = self.noise_generator
            ).images
        return images
    
    def generate_seed_change(self, prompt: str, seed_reset_step : int = 25, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        #if self.pipe is not Union[Type[StableDiffusionPipelineMidwayPatch], Type[AltDiffusionPipelineMidwayPatch]]:
        #    raise ValueError("Seed change only supported for the MidwayPatch patched pipelines in models.patches")
        # warn that this is going to get deprecated
        warnings.warn("generate_seed_change is gonna get deprecated. Use generate_steps_change instead.")
        images = []
        with autocast(self.device):
            images += self.pipe(
                prompt, 
                reset_step = seed_reset_step, 
                guidance_scale = guidance_scale, 
                num_images_per_prompt = num_img, 
                generator = self.noise_generator
            ).images
        return images
    
    def generate_prompt_change(self, prompt: str, second_prompt: str, prompt_reset_step : int = 25, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        #print(type(self.pipe))
        #if self.pipe is not Union[Type[StableDiffusionPipelineMidwayPatch], Type[AltDiffusionPipelineMidwayPatch]]:
        #    raise ValueError("Seed change only supported for the MidwayPatch patched pipelines in models.patches")
        # warn that this is going to get deprecated
        warnings.warn("generate_seed_change is gonna get deprecated. Use generate_steps_change instead.")
        images = []
        with autocast(self.device):
            images += self.pipe(
                prompt, 
                second_prompt = second_prompt, 
                reset_step = prompt_reset_step, 
                guidance_scale = guidance_scale, 
                num_images_per_prompt = num_img, 
                generator = self.noise_generator
            ).images
        return images

    def generate_steps_change(self, prompt: str, changes_list : Optional[Union[Tuple[Optional[str], int], List[Tuple[Optional[str],int]]]], guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        images = []
        with autocast(self.device):
            images += self.pipe(
                prompt, 
                changes_list = changes_list, 
                guidance_scale = guidance_scale, 
                num_images_per_prompt = num_img, 
                generator = self.noise_generator
            ).images
        return images
