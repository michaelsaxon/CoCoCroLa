import os
from typing import List, Union
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPVisionModel


def open_image_if_exists(fname):
    if os.path.isfile(fname):
        return Image.open(fname, "r")
    else:
        return Image.new('RGB', (50, 50), (0, 0, 0))


class ImageProcessor():
    """Given images or fnames, output vectors for those images on device (if app)."""

    def __init__(self, device : str = "cpu"):
        raise NotImplementedError

    # Given a list of images, return the expected vectors
    def process(self, images : List[Image.Image]) -> List[torch.Tensor]:
        raise NotImplementedError

    # Given a list of fnames or images, return the expected vectors
    def process_from_file(self, fnames_or_images : Union[List[str], List[Image.Image]]):
        if fnames_or_images is List[str]:
            images = [open_image_if_exists(fname) for fname in fnames_or_images]
        else:
            images = fnames_or_images
        return self.process(images)


# This is how you write a wrapper for some image extractor using CLIP
class CLIPImageProcessor(ImageProcessor):
    def __init__(self, model_name : str = "openai/clip-vit-base-patch32", device : str = "cpu"):
        self.processor = CLIPProcessor(device = device)
        self.model = CLIPVisionModel.from_pretrained(model_name, device = device)
    
    def process(self, images: List[Image.Image]):
        return self.model(**self.processor(
                images, 
                return_tensors="pt"
            ).to(self.model.device)).pooler_output.squeeze()