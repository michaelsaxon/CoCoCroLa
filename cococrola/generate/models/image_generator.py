import PIL
from typing import List

class ImageGenerator():
    def __init__(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, num_img: int = 9) -> List[PIL.Image]:
        raise NotImplementedError