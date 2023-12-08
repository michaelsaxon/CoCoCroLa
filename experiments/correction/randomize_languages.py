# requires pip install -U sentence-transformers

# read in the starting csv and then generate EN,testlang-before,testlang-after with 10 different randomized orderings for testlang-before
# write out the new csv with EN,testlang-before,testlang-after,sim-before,sim-after,sim-diff

import click
import random
import json
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

# takes in a sentence sim model and two sents, returns the simscores 
def get_sim_scores_string(text_model, sent_english, sent_ml_before, sent_ml_after):
    sim_before = util.cos_sim(text_model.encode(sent_english), text_model.encode(sent_ml_before))
    sim_after = util.cos_sim(text_model.encode(sent_english), text_model.encode(sent_ml_after))
    sim_diff = sim_after - sim_before
    return f"{float(sim_before)},{float(sim_after)},{float(sim_diff)}"

@click.command()
@click.option('--input_folder', default='../../benchmark/v0-1/')
def main(input_folder):
    lines = open(input_folder + "/concepts.csv", "r").readlines()
    prompt_templates = json.load(open(input_folder + "/prompts.json", "r"))
    index = lines[0].strip().split(",")
    template_en = prompt_templates["en"]
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    text_model.to("cuda")
    # for each language, generate 10 different randomized orderings
    for lang_idx in range(1, len(index)):
        lang = index[lang_idx]
        out_lines = [f"en,{lang}-before,{lang}-after,sim-before,sim-after,sim-diff\n"]
        print(f"Generating for {lang}...")
        template_other_lang = prompt_templates[lang]
        # for each language, generate 10 different randomized orderings
        for j in range(10):
            # generate a random ordering of the images
            print(f"Ordering {j} of 10...")
            random_ordering = list(range(1, len(lines)))
            random.shuffle(random_ordering)
            for i in tqdm(range(1, len(lines))):
                true_english = lines[i].strip().split(",")[0]
                true_other_lang = lines[i].strip().split(",")[lang_idx]
                random_other_lang = lines[random_ordering[i-1]].strip().split(",")[lang_idx]
                sim_scores_string = get_sim_scores_string(
                    text_model, 
                    template_en.replace("$$$",true_english), 
                    template_other_lang.replace("$$$",true_other_lang), 
                    template_other_lang.replace("$$$",random_other_lang)
                )
                out_lines.append(f"{true_english},{true_other_lang},{random_other_lang},{sim_scores_string}\n")
        with open(f"randomized_{lang}.csv", "w") as f:
            f.writelines(out_lines)
            

if __name__ == "__main__":
    main()