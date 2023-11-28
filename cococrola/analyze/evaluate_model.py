# python evaluate_model.py --input_csv ../../benchmark/v0-1/concepts.csv --analysis_dir ../../results/polyglot/SD2_en_600/
import click
from PIL import Image, ImageFile
from transformers import CLIPProcessor, CLIPVisionModel
from collections import defaultdict
import torch.nn.functional as F
import random
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

"samples-11-30-7_5-flg1/0-en-dog-0.png"

# if image exists, open it. Else, generate 50x50 black
def open_image_if_exists(fname):
    if os.path.isfile(fname):
        return Image.open(fname, "r")
    else:
        print(f"Failing to open {fname}")
        return Image.new('RGB', (50, 50), (0, 0, 0))

def get_image_embeddings(processor, model, fnames):
    images = [open_image_if_exists(fname) for fname in fnames]
    inputs = processor(images=images, return_tensors="pt")
    inputs.to(model.device)
    outputs = model(**inputs)
    return outputs.pooler_output.squeeze()

def avg_cos_sim(vec_list_1, vec_list_2, is_self_sim = False):
    # this is O(n^2) lmao
    sims_sum = 0
    sims_num = vec_list_1.shape[0] * vec_list_2.shape[0] - is_self_sim * vec_list_1.shape[0]
    for i in range(vec_list_2.shape[0]):
        sims_sum += float(F.cosine_similarity(vec_list_1, vec_list_2[i].unsqueeze(0)).sum() - 1 * is_self_sim)
    return sims_sum / sims_num

# query cross-consistency
def compare_by_lang(results_dict, main_lang = "en", similarity_func = avg_cos_sim):
    langs = results_dict.keys()
    # evaluate pairwise similarity
    output_dict = {}
    for lang in langs:
        output_dict[lang] = similarity_func(results_dict[main_lang], results_dict[lang], lang == main_lang)
    return output_dict

# query self-sim by lang
def lang_self_sim(results_dict, similarity_func = avg_cos_sim):
    langs = results_dict.keys()
    # evaluate pairwise similarity
    output_dict = {}
    for lang in langs:
        output_dict[lang] = similarity_func(results_dict[lang], results_dict[lang], True)
    return output_dict

# query self-sim by lang
def lang_cross_sim(results_dict_1, results_dict_2, similarity_func = avg_cos_sim):
    # suggest using larger vector as results_dict_1
    langs = results_dict_1.keys()
    # evaluate pairwise similarity
    output_dict = {}
    for lang in langs:
        output_dict[lang] = similarity_func(results_dict_1[lang], results_dict_2[lang], False)
    return output_dict


# results dict is a language-indexed dictionary of the gpu-placed word matrices
# the fingerprint dict is identical in structure but not tied to a word.

# produce a language-level index by precomputing the n=0 image for every word, language pair
def precompute_fingerprint_matrix(processor, model, prompts_base, analysis_dir, selection_count, number_spec_range = 12):
    fingerprints = {}
    index = prompts_base[0].strip().split(",")
    if selection_count >= len(prompts_base) - 1 or selection_count == -1:
        use_lines = list(range(len(prompts_base) - 1))
    else:
        use_lines = random.sample(range(len(prompts_base) - 1), selection_count)
    # extract a fingerprint for each language
    for idx in range(len(index)):
        fnames = []
        for line_no in use_lines:
            line = prompts_base[line_no + 1]
            line = line.strip().split(",")
            # sample a number in the correct range
            img_idx = random.randrange(number_spec_range)
            fnames.append(f"{analysis_dir}/{line_no}-{index[idx]}-{line[0]}-{img_idx}.png")
        fingerprints[index[idx]] = get_image_embeddings(processor, model, fnames)
    return fingerprints


@click.command()
@click.option('--analysis_dir', default='samples_sd2')
@click.option('--num_samples', default=9)
@click.option('--fingerprint_selection_count', default=100)
@click.option('--main_language', default="en")
@click.option('--input_csv', type=str, default="../../benchmark/v0-1/concepts.csv")
@click.option('--eval_samples_file', type=str, default=None, help="If specified, only use the line numbers listed in this file to evaluate")
def main(analysis_dir, num_samples, fingerprint_selection_count, main_language, input_csv, eval_samples_file):
    device = "cuda"
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    
    prompts_base = open(input_csv, "r").readlines()
    index = prompts_base[0].strip().split(",")

    out_lines_main_sim = [prompts_base[0]]
    out_lines_self_sim = [prompts_base[0]]
    out_lines_main_spec = [prompts_base[0]]

    # collect the fingerprints for each language in this model
    fingerprints = precompute_fingerprint_matrix(processor, model, prompts_base, analysis_dir, fingerprint_selection_count)
    # language fingerprint self-similarity (negative diversity)
    inverse_diversity = lang_self_sim(fingerprints)
    print(f"INVERSE_DIVERSITY: {inverse_diversity}")
    with open(f"{analysis_dir}/language_diversity.csv", "w") as f:
        f.writelines([prompts_base[0], ",".join([str(inverse_diversity[index]) for index in index]) + "\n"])
    
    if eval_samples_file is not None:
        eval_samples = [int(line.strip()) for line in open(eval_samples_file, "r").readlines()]
    else:
        eval_samples = range(1, len(prompts_base))

    # for line_no, line in enumerate(prompts_base[1:]):
    for line_no, line in [(i, prompts_base[i]) for i in eval_samples]:
        results_dict = defaultdict(list)
        line = line.strip().split(",")
        
        # collect this languages embeddings
        for idx in range(len(index)):
            # build a prompt based on the above templates from the 
            fnames = [f"{analysis_dir}/{line_no}-{index[idx]}-{line[0]}-{i}.png" for i in range(num_samples)]
            image_embedding = get_image_embeddings(processor, model, fnames)
            results_dict[index[idx]] = image_embedding
        
        language_similarities = compare_by_lang(results_dict, main_lang=main_language)
        self_sims = lang_self_sim(results_dict)
        inverse_specificity = lang_cross_sim(fingerprints, results_dict)

        # zero out if there's an error log for each word
        for language in index:
            if os.path.isfile(f"{analysis_dir}/{line_no}-{language}-{line[0]}-failure.log"):
                language_similarities[language] = "---"
                self_sims[language] = "---"
                inverse_specificity[language] = "---"
        

        print(f"{main_language} SIM " + line[0] + " " + str(language_similarities))
        print("self SIM " + line[0] + " " + str(self_sims))
        print("specific " + line[0] + " " + str(inverse_specificity))

        out_lines_main_sim.append(",".join([str(language_similarities[language]) for language in index]) + "\n")
        out_lines_self_sim.append(",".join([str(self_sims[language]) for language in index]) + "\n")
        out_lines_main_spec.append(",".join([str(inverse_specificity[language]) for language in index]) + "\n")
        
    with open(f"{analysis_dir}/results_{main_language}.csv", "w") as f:
        f.writelines(out_lines_main_sim)

    with open(f"{analysis_dir}/results_self.csv", "w") as f:
        f.writelines(out_lines_self_sim)

    with open(f"{analysis_dir}/results_specific.csv", "w") as f:
        f.writelines(out_lines_main_spec)
    


if __name__ == "__main__":
    main()