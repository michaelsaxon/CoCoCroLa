from PIL import Image
from typing import List

class ImageGenerator():
    def __init__(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, num_img: int = 9) -> List[Image.Image]:
        raise NotImplementedError

    def update_noise_generator(self, seed : int = 0) -> None:
        raise NotImplementedError