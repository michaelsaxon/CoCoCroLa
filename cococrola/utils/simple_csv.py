# the first line of the csv is a header containing column labels (eg, language codes)
# return an iterator over pairs of (index value, cell contents, current line no, line[0])
def csv_to_index_elem_iterator(input_csv_path : str, start_line : int = 1, ref_lang : str = "en"):
    csv_lines = open(input_csv_path, "r").readlines()
    index = csv_lines[0].strip().split(",")
    ref_lang_idx = index.index(ref_lang)
    for line_idx in range(start_line, len(csv_lines)):
        line = csv_lines[line_idx]
        line = line.strip().split(",")
        for lang_idx, lang in enumerate(index):
            yield (line_idx - 1, line[ref_lang_idx], lang, line[lang_idx])