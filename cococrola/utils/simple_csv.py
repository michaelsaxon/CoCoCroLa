from typing import Union, List

# the first line of the csv is a header containing column labels (eg, language codes)
# return an iterator over pairs of (index value, cell contents, current line no, line[0])
def csv_to_index_elem_iterator(input_csv_path : str, start_line : Union[int, List[int]] = 1, ref_lang : str = "en"):

    # name start_line is grandfathered in from the old implementation, where it was always just a way to
    # restart dead runs. It now supports the functionality to run through an arbitrary set of lines.
    # in a refactor it should be renamed to something like "run_indices"
    # the old start_line behavior is currently preserved if it's an int (this is least invasive way)
    # but eventually it should only take a list of line nos, generated in the outer run script, and be renamed

    csv_lines = open(input_csv_path, "r").readlines()
    index = csv_lines[0].strip().split(",")
    print(index)
    ref_lang_idx = index.index(ref_lang)

    if isinstance(start_line, int):
        run_indices = range(start_line, len(csv_lines))
    else:
        run_indices = start_line

    for line_idx in run_indices:
        line = csv_lines[line_idx]
        line = line.strip().split(",")
        for lang_idx, lang in enumerate(index):
            yield (line_idx - 1, line[ref_lang_idx], lang, line[lang_idx])

