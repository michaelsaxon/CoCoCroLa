from PIL import Image
import os

def convert_dir_to_dir(input_path, output_path, in_fmt = "png", out_fmt = "jpg"):
    format_match_files = [_fname for _fname in os.listdir(path) if _fname[-len(in_fmt):] == in_fmt]
    for fname in format_match_files:
        im = Image.open(input_path + "/" + fname)
        im.save(f"{output_path}/{fname[:-len(in_fmt)]}{out_fmt}")
        print(f"saved {fname} to {output_path}/{fname[:-len(in_fmt)]}{out_fmt}")