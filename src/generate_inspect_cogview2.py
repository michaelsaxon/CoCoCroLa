import torch
from PIL import Image
import argparse
from tqdm import tqdm

print("This requires installation of CogView2")

from cogview2_text2image import *


# some of these are hacks. I believe german and indonesian require some inflections depending on the word
# english, spanish, and german additionally probably need an indefinite article
# unclear if this is the correct colloquial chinese
LANG_PROMPT_BITS = {
    'en' : "a photograph of $$$",
    'es' : "una fotografía de $$$",
    'de' : "ein Foto von $$$",
    'zh' : "$$$照片",
    'ja' : "$$$の写真",
    #'kr' : Language.KR,
    'he' : " צילום של$$$",
    'id' : "foto $$$"
}


def get_cogview2_gen_func(args):
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


def main():
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--img-size', type=int, default=160)
    py_parser.add_argument('--only-first-stage', action='store_true')
    py_parser.add_argument('--inverse-prompt', action='store_true')
    py_parser.add_argument('--style', type=str, default='mainbody', 
        choices=['none', 'mainbody', 'photo', 'flat', 'comics', 'oil', 'sketch', 'isometric', 'chinese', 'watercolor'])
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known), **get_recipe(known.style))

    with torch.no_grad():
        cogview2_img_from_prompt = get_cogview2_gen_func(args)
        print("collected model successfully")
        prompts_base = open("frequencylist/freq_lists_gold.csv", "r").readlines()
        index = prompts_base[0].strip().split(",")
        for line_no, line in enumerate(prompts_base[1:]):
            line = line.strip().split(",")
            for idx in range(len(index)):
                # build a prompt based on the above templates from the 
                prompt = LANG_PROMPT_BITS[index[idx]].replace("$$$", line[idx])
                print(f"generating {index[idx]}:{line[0]}, '{line[idx]}'")
                image = cogview2_img_from_prompt(prompt)
                for i, im in enumerate(image):
                    fname = f"{line_no}-{index[idx]}-{line[0]}-{i}.png"
                    print(f"saving image {fname}...")
                    im.save(f"samples_cv2/{fname}")


if __name__ == "__main__":
    main()