from typing import List
import torch
from PIL import Image
import argparse
from tqdm import tqdm

from cogview2_text2image import *

from image_generator import ImageGenerator

def get_cogview2_gen_func(img_size : int = 160, first_stage : bool = False, inverse_prompt : bool = False):
    # stupid hack for argparse dependency in cogview booooooo
    args = argparse.ArgumentParser().parse_args(args=['--img-size' , img_size, '--only-first-stage' , first_stage, '--inverse-prompt' , inverse_prompt, '--style' , 'mainbody'])
    model, args = InferenceModel.from_pretrained(args, 'coglm')
    
    invalid_slices = [slice(tokenizer.num_image_tokens, None)]
    strategy = CoglmStrategy(invalid_slices,
                            temperature=args.temp_all_gen, top_k=args.topk_gen, top_k_cluster=args.temp_cluster_gen)
    
    from sr_pipeline import SRGroup 
    if not args.only_first_stage:
        srg = SRGroup(args)
        
    def process(text, num_img_per_prompt = 5):
        seq = tokenizer.encode(text)
        txt_len = len(seq) - 1
        seq = torch.tensor(seq + [-1]*400, device=args.device)
        # calibrate text length
        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=args.device, dtype=torch.half)
        log_attention_weights[:, :txt_len] = args.attn_plus
        # generation
        get_func = partial(get_masks_and_position_ids_coglm, context_length=txt_len)
        output_list, score_list = [], []

        for _ in tqdm(range(2)):
            strategy.start_pos = txt_len + 1
            coarse_samples = filling_sequence(model, seq.clone(),
                    batch_size=num_img_per_prompt,
                    strategy=strategy,
                    log_attention_weights=log_attention_weights,
                    get_masks_and_position_ids=get_func
                    )[0]
                        
            output_list.append(
                    coarse_samples
                )
            output_tokens = torch.cat(output_list, dim=0)
                    
        imgs = []
        iter_tokens = srg.sr_base(output_tokens[:, -400:], seq[:txt_len])
        for seq in tqdm(iter_tokens):
            decoded_img = tokenizer.decode(image_ids=seq[-3600:])
            decoded_img = torch.nn.functional.interpolate(decoded_img, size=(480, 480)).squeeze()
            imgs.append(decoded_img) # only the last image (target)

        return list(map(
            lambda img: Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()), 
            imgs
        ))

    return process


class CogviewImageGenerator(ImageGenerator):
    def __init__(self):
        self.cogview2_gen_func = get_cogview2_gen_func()
    
    def generate(self, prompt: str, num_img: int = 9) -> List:
        with torch.no_grad():
            return self.cogview2_gen_func(prompt, num_img)