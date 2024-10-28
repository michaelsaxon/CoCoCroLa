import click
import json
from collections import defaultdict
import openai
from multiprocessing import Pool
from typing import List, Dict


LANGS = [
    'en',
    'es',
    'fr',
    'de',
    'da',
    'id',
    'tl',
    'vi',
    'sw',
    'hr',
    'pl',
    'ru',
    'sr',
    'zh',
    'ja',
    'ko',
    'he',
    'ar',
    'ru',

]


# lang code in, language frequency list out, each frequency list is keyed by words, 
def get_freq_list(lang):
    print(f"Loading language json {lang}...")
    with open(f"{lang}_2k.json", "r") as f:
        return json.loads(f.read())


TRANSLATION_PROMPT = """
[Tangible sense translation system] - Translates given English term, describing a tangible noun, in the most common, colloquial sense into the given target language (described by ISO lang code), in as short a phrase as possible.

For example:
rock, es: roca; piedra
airplane, ko: 비행기
bike, fr: vélo
{source_word}, {target_language}: 
"""

def get_single_translation(source_name, target_language):
    prompt = TRANSLATION_PROMPT.format(source_word=source_name, target_language=target_language)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        temperature=0.2,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
    )
    return response.choices[0].text.strip()

def get_aligned_row(source_name, target_languages):
    # right now source lang is always english
    return {lang : get_single_translation(source_name, lang) for lang in target_languages}

def aligned_row_to_csv(source_name : str, aligned_row : Dict[str], test_languages : List[str]):
    return ",".join([source_name] + list(map(lambda lang: aligned_row[lang], test_languages))) + "\n"

def is_noun(word):
    NOUN_PROMPT = """
    [Noun detection system] - Determines if the given word is a tangible noun in English. Abstract nouns should not count. Returns True if the word is a noun, False otherwise. Examples:
    rock: True
    happiness: False
    eat: False
    {word}:
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=NOUN_PROMPT.format(word=word),
        max_tokens=50,
        temperature=0.2,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None,
    )
    return response.choices[0].text.strip() == "True"

# command to get the list of best words from the frequency list
@click.command()
@click.option('--main_lang', default='en')
@click.option('--output_file', default='freq_lists.csv')
@click.option('--langsfile', default=None, help='newline separated file with list of languages to translate to')
def translation_from_freqlist(main_lang, output_file, langsfile):
    if langsfile is not None:
        with open(langsfile, "r") as f:
            languages = f.read().split("\n")
    else:
        languages = LANGS
    
    freq_lists_dict = {}
    main_lang = 'en'
    test_languages = LANGS
    for lang in languages:
        freq_lists_dict[lang] = get_freq_list(lang)
    # we will save the final list as a csv
    test_languages = [lang for lang in languages if lang != main_lang]

    csv_rows = [",".join([main_lang] + test_languages) + "\n"]
    for word in freq_lists_dict[main_lang].keys():
        print(word)
        if not is_noun(word):
            print("main: not a noun")
            continue
        else:
            aligned_row = get_aligned_row(word, test_languages)
            csv_rows.append(aligned_row_to_csv(word, aligned_row, test_languages))

    with open(output_file, "w") as f:
        f.writelines(csv_rows)

if __name__ == "__main__":
    translation_from_freqlist()