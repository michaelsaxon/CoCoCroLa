# make a working install of wimbd next to the CoCoCroLa directory
# cp zipf.py ../../wimbd/.
# cd ../../wimbd
# python zipf.py
# cd ../CoCoCroLa/experiments/wimbd-integration

from wimbd.es import count_documents_containing_phrases

from tqdm import tqdm

def split_line(line):
    return line.strip().split(",")

concepts = open("../CoCoCroLa/benchmark/v0-1/concepts.csv", "r").readlines()

langs = split_line(concepts[0])
concepts = list(map(split_line, concepts[1:]))

out_line_lists = [langs]

#with open(f"{analysis_dir}/language_diversity.csv", "w") as f:
#    f.writelines([prompts_base[0], ",".join([str(inverse_diversity[index]) for index in index]) + "\n"])

for concept_line in tqdm(concepts):
    counts = []
    for concept in concept_line:
        counts.append(count_documents_containing_phrases("re_laion2b-en-*", concept))
    out_line_lists.append(counts)

with open("../CoCoCroLa/experiments/wimbd-integration/counts_zipf.csv","w") as f:
    f.writelines([",".join(map(str, line)) + "\n" for line in out_line_lists])