import torch
from PIL import Image
from torch import autocast
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from typing import List, Type, Optional, Union

from tqdm import tqdm

from cococrola.generate.models.image_generator import ImageGenerator
from cococrola.generate.models.patches.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMidwayPatch
from cococrola.generate.models.patches.diffusers.pipeline_alt_diffusion import AltDiffusionPipelineMidwayPatch

'''
Functionality to add:
- [x] set beginning of run seed to be the same for each language 
- [ ] switch seed midway through generation (requires modification to level Kexun dev was on)
    - [ ] write a general pipeline function in a patch part
- [ ] switch conditioning prompt midway through generation
'''


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
    
    def generate_seed_change(self, prompt: str, seed_reset_step : int = 500, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        if self.pipe is not Union[Type[StableDiffusionPipelineMidwayPatch], Type[AltDiffusionPipelineMidwayPatch]]:
            raise ValueError("Seed change only supported for the MidwayPatch patched pipelines in models.patches")
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
    
    def generate_prompt_change(self, prompt: str, second_prompt: str, prompt_reset_step : int = 500, guidance_scale : float = 7.5, num_img: int = 9) -> List[Image.Image]:
        print(type(self.pipe))
        if self.pipe is not Union[Type[StableDiffusionPipelineMidwayPatch], Type[AltDiffusionPipelineMidwayPatch]]:
            raise ValueError("Seed change only supported for the MidwayPatch patched pipelines in models.patches")
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
