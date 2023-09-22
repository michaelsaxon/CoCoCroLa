from PIL import Image
from typing import List


class ImageGenerator():
    def __init__(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, num_img: int = 9) -> List[Image.Image]:
        raise NotImplementedError

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
        raise NotImplementedError

    def generate_seed_change(self, prompt: str, seed_reset_step : int = 500, num_img: int = 9) -> List[Image.Image]:
        raise NotImplementedError
    
    def generate_prompt_change(self, prompt: str, second_prompt: str, prompt_reset_step : int = 500, num_img: int = 9) -> List[Image.Image]:
        raise NotImplementedError