from typing import List
from PIL import Image

def save_imgs_to_dir(images : List[Image.Image], fname_base : str):
    for image_number, im in enumerate(images):
        fname = f"{fname_base}-{image_number}.png"
        print(f"saving image {fname}...")
        im.save(fname)
