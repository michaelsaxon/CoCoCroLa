import click
import babelnet as bn
from babelnet.language import Language
from babelnet.pos import POS
from babelnet.data.source import BabelSenseSource
import json
from collections import defaultdict
import translators as ts
from multiprocessing import Pool

VALID_SOURCES = [
    BabelSenseSource.WN,
    BabelSenseSource.OMWN,
    BabelSenseSource.IWN,
    BabelSenseSource.WNTR,
    BabelSenseSource.OMWN_ID,
    BabelSenseSource.OMWN_ZH,
    BabelSenseSource.OMWN_JA,
    BabelSenseSource.MCR_CA,
    BabelSenseSource.MCR_ES,
    BabelSenseSource.OMWN_KO,
    BabelSenseSource.MCR_PT,
    BabelSenseSource.OMWN_GAE,
    BabelSenseSource.OMWN_CWN,
    BabelSenseSource.WN2020,
    BabelSenseSource.OEWN
]


LANGS = {
    'en' : Language.EN,
    'es' : Language.ES,
    'de' : Language.DE,
    'zh' : Language.ZH,
    'ja' : Language.JA,
    #'kr' : Language.KR,
    'he' : Language.HE,
    'id' : Language.ID
}


# lang code in, language frequency list out, each frequency list is keyed by words, 
def get_freq_list(lang):
    print(f"Loading language json {lang}...")
    with open(f"{lang}_2k.json", "r") as f:
        return json.loads(f.read())


def get_word_or_synsets(word, to_langs, from_lang):
    # first, test if this word has a wikipedia page title
    synset = bn.get_synset(bn.resources.WikipediaID(word, LANGS[from_lang]))
    if synset is not None:
        if synset.pos == POS.NOUN:
            return synset
        else:
            return None
    else:
        candidate_synsets = bn.get_synsets(
            word, 
            from_langs=[LANGS[from_lang]], 
            to_langs=[LANGS[to_lang] for to_lang in to_langs], 
            sources=VALID_SOURCES
        )
        # this produces a list of synsets for the word, will include garbage senses, use heuristics to pick
        # heuristic 1: is the name in one of the titles
        names = [str(word_synset).split("__")[1].split("#")[0].lower() for word_synset in candidate_synsets]
        if word.lower() in names:
            synset = candidate_synsets[names.index(word)]
        else:
            return candidate_synsets
    return synset

# [y for y in x if str(y.language).lower() in test_languages]


LANGS_GOOGLE = {
    'he' : 'iw',
    'zh' : 'zh-CN'
}

def lmap(mapdict, lang):
    return mapdict.get(lang, lang)

TLS = {
    "gg": ts.google,
    "bd" : ts.baidu,
    "bg" : ts.bing,
    "it" : ts.itranslate
}

def attempt_translation(word, from_lang, to_lang, translator_function, translator_map):
    try:
        return TLS[translator_function](word, from_language=lmap(translator_map, from_lang), to_language=lmap(translator_map, to_lang)).lower()
    except:
        return "*NOT_FOUND*"

def simplified_translator_mapping(input_elems):
    word, from_lang, to_lang, translator, lmapping = input_elems
    return attempt_translation(word, from_lang, to_lang, translator, lmapping)

def translator_heuristic(word, from_lang, to_lang):
    # could add others https://pypi.org/project/translate-api/
    translators = [
        (word, from_lang, to_lang, "gg", LANGS_GOOGLE), 
        (word, from_lang, to_lang, "bd", {}), 
        (word, from_lang, to_lang, "bg", {}), 
        (word, from_lang, to_lang, "it", {})
    ]
    with Pool(len(translators)) as p:
        hypotheses = p.map(simplified_translator_mapping, translators)
    return hypotheses

# trying to improve this stuff:
'''
elem members include 'full_lemma', 'id', 'is_automatic_translation', 'is_key_sense', 
'is_not_automatic_translation', 'language', 'lemma', 'license', 'normalized_lemma', 
'pos', 'pronunciations', 'sense_str', 'sensekey', 'source', 'synset', 'synset_id', 'to_uri' 
'''
def get_aligned_row(synset, test_languages, freq_lists_dict):
    candidates = defaultdict(dict)
    quality = 0
    for elem in synset:
        word = str(elem).split(":")[-1]
        lang = str(elem.language).lower()
        if word in freq_lists_dict.get(lang, []):
            candidates[lang][word] = freq_lists_dict[lang][word]
    for lang in test_languages:
        if candidates.get(lang, False) is False:
            print(f"GArR: word {word} not found for lang {lang}")
            return None, None
        quality += max(candidates[lang].values())
        candidates[lang] = max(candidates[lang], key=candidates[lang].get)
    return candidates, quality


def aligned_row_to_csv(source_name, aligned_row, test_languages):
    return ",".join([source_name] + list(map(lambda lang: aligned_row[lang], test_languages))) + "\n"

@click.command()
@click.option('--main_lang', default='en')
@click.option('--output_file', default='freq_lists.csv')
def main_freqlist(main_lang, output_file):
    freq_lists_dict = {}
    languages = list(LANGS.keys())
    for lang in languages:
        freq_lists_dict[lang] = get_freq_list(lang)
    # we will save the final list as a csv
    test_languages = [lang for lang in languages if lang != main_lang]

    csv_rows = [",".join([main_lang] + test_languages) + "\n"]
    for word in freq_lists_dict[main_lang].keys():
        print(word)
        synset_or_list = get_word_or_synsets(word, test_languages, main_lang)
        if synset_or_list is None:
            # can't get anything for this word
            print("main: coundn't find any synset for this word")
            continue
        if type(synset_or_list) is list:
            # we need to determine which is the best
            best_quality = 0
            for synset in synset_or_list:
                aligned_row, quality = get_aligned_row(synset, test_languages, freq_lists_dict)
                if aligned_row is None:
                    continue
                if quality >= best_quality:
                    row = aligned_row
                    best_quality = quality
            if best_quality == 0:
                print("main: no aligned row across all langs for this synset (from a list)")
                continue
        else:
            # it's a single synset. let's parse and get the crosslingual words
            aligned_row, _ = get_aligned_row(synset_or_list, test_languages, freq_lists_dict)
            if aligned_row is None:
                print("main: found a single word, no alignment across all langs")
                continue
            row = aligned_row
        # row should only ever be an aligned row
        csv_row = aligned_row_to_csv(word, row, test_languages)
        print(csv_row)
        csv_rows.append(csv_row)

    with open(output_file, "w") as f:
        f.writelines(csv_rows)


def synset_word_best(synset, word, candidate_words, test_languages):
    candidates = {}
    for elem in synset:
        word = str(elem).split(":")[-1]
        lang = str(elem.language).lower()
        if lang not in test_languages:
            continue
        if word in candidate_words[lang]:
            candidates[lang] = word
    #if len(test_languages) > len(candidates.keys()):
    #    print(f"Missing a language: {candidates}")
    return candidates


def meld_overlapping_dicts(list_of_dicts, target_key_set):
    sample = {key : None for key in target_key_set}
    #print(list_of_dicts)
    list_of_dicts.sort(key = lambda _dict: len(_dict.keys()))
    if list_of_dicts[0].keys() == sample.keys():
        return list_of_dicts[0]
    for i in range(len(list_of_dicts)):
        out_dict = {key: list_of_dicts[i][key] for key in list_of_dicts[i].keys()}
        for j in range(i+1, len(list_of_dicts)):
            for key in list_of_dicts[j].keys():
                if key in out_dict.keys():
                    if out_dict[key] != list_of_dicts[j][key]:
                        continue
                else:
                    out_dict[key] = list_of_dicts[j][key]
        if out_dict.keys() == sample.keys():
            return out_dict
    else:
        return list_of_dicts[0]



@click.command()
@click.option('--main_lang', default='en')
@click.option('--input_file', default='english_nouns.txt')
@click.option('--output_file', default='freq_lists_2_$$$.csv')
@click.option('--start_line', default=0)
@click.option('--end_line', default=-1)
def main_translation_service(main_lang, input_file, output_file, start_line, end_line):
    in_lines = open(input_file).readlines()
    if end_line == -1:
        end_line = len(in_lines)
    input_words = map(lambda x: x.strip().lower(), in_lines[start_line:end_line])

    languages = list(LANGS.keys())
    test_languages = [lang for lang in languages if lang != main_lang]

    csv_rows = [",".join([main_lang] + test_languages) + "\n"]
    for word in input_words:
        candidate_words = {language : translator_heuristic(word, main_lang, language) for language in test_languages}
        print(word)
        synset_or_list = get_word_or_synsets(word, test_languages, main_lang)
        if synset_or_list is None:
            # can't get anything for this word
            print("main: coundn't find any synset for this word")
            continue
        if type(synset_or_list) is list:
            print("testing a list of synset options")
            # we need to determine which is the best
            best_quality = 0  
            row = meld_overlapping_dicts(
                [synset_word_best(synset, word, candidate_words, test_languages) for synset in synset_or_list],
                test_languages
            )
            quality = len(row.keys())
            if quality < len(test_languages):
                print("main: no aligned row across all langs for this synset (from a list)")
                continue
        else:
            # it's a single synset. let's parse and get the crosslingual words
            row = synset_word_best(synset_or_list, word, candidate_words, test_languages)
            quality = len(row.keys())
            if row is None:
                print("main: found a single word, no alignment across all langs")
                continue
            if quality < len(test_languages):
                print("main: found a single word, no alignment across all langs")
                continue
        # row should only ever be an aligned row
        csv_row = aligned_row_to_csv(word, row, test_languages)
        print(csv_row)
        csv_rows.append(csv_row)

    with open(output_file.replace("$$$",f"{start_line}_{end_line}"), "w") as f:
        f.writelines(csv_rows)




if __name__ == "__main__":
    main_translation_service()