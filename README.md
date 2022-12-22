# CoCoCroLa: Conceptual Coverage Across Languages Benchmark

### [\[Demo Link\]](https://saxon.me/coco-crola)

### Package currently under cleaning and construction.

## Temporary non-package usage instructions:

3 critical files
```
benchmark/v0-1/concepts.csv
benchmark/v0-1/prompts.json
cococrola/evaluate/evaluate_folder.py
```

### Loading concepts and saving returned images

The CoCo-CroLa v0.1 concept list is contained in `concepts.csv`. Each row contains the concept in the language signified by its corresponding column. To convert a translated concept word into the CCCL v0.1 prompts, replace `###` in its language's prompt (located in `prompts.json`).

To use `evaluate_folder.py`, run your model on each concept,language pair n times (recommended n>=10) and name each file according to the format: `{[concept.csv line number] - 2}-{language}-{english concept name}-{}.png`

For example, the second image generated for german for the concept "mother" (concept number 15), the filename will be `14-de-mother-2.png`.


#### Minimal example

```python
import json

import YOUR_TEST_MODEL

N_IMG_PER_PROMPT = 10

OUTPUT_FOLDER = "folder"

prompts = json.load("benchmark/v0-1/prompts.json")
with open("benchmark/v0-1/concepts.csv", "r") as f:
    concepts = list(map(lambda x: x.strip().split(","), f.readlines()))
    languages = concepts[0]
    concepts = concepts[1:]

for concept_id, concept in enumerate(concepts):
    concept_en = concept[languages.index('en')]
    for lang_idx in range(len(concept)):
        language = languages[lang_idx]
        prompt = prompts[language].replace("###", concept[lang_idx])
        for image_idx in range(N_IMG_PER_PROMPT):
            image = YOUR_TEST_MODEL.text_to_image(prompt)
            image.save(f"{OUTPUT_FOLDER}/{concept_id}-{language}-{concept_en}-{i}.png")
```

For example, `cococrola/run_diffusers.py` implements this loop (requires `click`, `torch`, `diffusers`).

### Running analysis on a folder of images

Once all files are placed in folder `folder`, e.g., `folder/0-en-eye-0.png`, `folder/0-en-eye-1.png`, ..., run the command: 

```bash
evaluate_folder.py --analysis_dir folder --main_language en
```
to generate output analysis files, `results_en.csv`, `results_self.csv`, `results_specific.csv`, containing the correctness, consistency, and specificity scores for each (language, concept) pair.

## Bugs questions and corrections

For any of the above, please file an issue or email me directly: [saxon@ucsb.edu](mailto:saxon@ucsb.edu)

## Citation:

```
@unpublished{              
saxon2022multilingual,              
title={Multilingual Conceptual Coverage in Text-to-Image Models},              
author={Michael Saxon,William Yang Wang},              
journal={OpenReview Preprint},              
year={2022},              
url={https://openreview.net/forum?id=5H2m3tCEaQ},
note={preprint under review}          
}
```
