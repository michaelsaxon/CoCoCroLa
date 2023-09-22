from PIL import Image
from typing import List
import os
import requests

import backoff
import openai

from cococrola.generate.models.image_generator import ImageGenerator


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def gen_img_with_backoff(prompt, num_img, size, model, api_key):
    return openai.Image.create(
                        prompt=prompt,
                        n=num_img,
                        size=size,
                        api_key=api_key
                    )


class OpenAIImageGenerator(ImageGenerator):
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model = "DE2"
        self.size = "256x256"

    def generate(self, prompt: str, num_img: int = 9) -> List[Image.Image]:
        response = gen_img_with_backoff(prompt, num_img, self.size, "DE2", self.openai_api_key)
        images = []
        for i in range(num_img):
            images.append(Image.open(requests.get(response['data'][i]['url'], stream=True).raw))
        return images

    def generate_split_batch(self, prompt: str, num_img: int = 9, split_batch: int = 3) -> List[Image.Image]:
        if split_batch < 1 or split_batch > num_img:
            raise ValueError("split_batch must be an integer between 1 and num_img")
        if num_img % split_batch != 0:
            raise ValueError("num_img must be divisible by split_batch")
        images = []
        for _ in range(num_img // split_batch):
            images += self.generate(prompt, split_batch)
        return images

    def update_noise_generator(self, seed : int = 0) -> None:
        raise AttributeError("The seed cannot be set for DALL-E 2 within the OpenAI API")

    def generate_seed_change(self, prompt: str, seed_reset_step : int = 500, num_img: int = 9) -> List[Image.Image]:
        raise AttributeError("The seed cannot be set for DALL-E 2 within the OpenAI API")
    
    def generate_prompt_change(self, prompt: str, second_prompt: str, prompt_reset_step : int = 500, num_img: int = 9) -> List[Image.Image]:
        raise AttributeError("The diffusion pipeline for DALL-E 2 cannot be accessed within the OpenAI API")
