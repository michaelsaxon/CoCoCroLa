from typing import List
import torch
from PIL import Image
import argparse
from tqdm import tqdm

from wimbd.es import *

from image_generator import ImageGenerator


def get_laion_wimbd_extractor_func(set_size : int = 50):

    # set up the es in whatever way is necessary

    def extract(prompt, num_img):
        return count_documents_containing_phrases("re_laion2b-en-*", prompt, set_size)

    return extract


class LAIONImageCollector(ImageGenerator):
    def __init__(self):
        self.laion_wimbd_extractor_func = get_laion_wimbd_extractor_func()
    
    def generate(self, prompt: str, num_img: int = 9) -> List:
        with torch.no_grad():
            return self.get_laion_wimbd_extractor_func(prompt, num_img)