import json
import sys

import pdb

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import click

# pip3 install wordfreq
from wordfreq import word_frequency
# requires jieba, mecab-python3, unidic-lite, ipadic

LANGS = ["en","es","de","zh","ja","he","id"]

def corr_plot(df, title, output_path):
    # generate the correlation plot
    corr = df.corr(method="spearman")
    print(corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.title(title)
    plt.show()
    #plt.savefig(output_path + f"corr_{row}_{column}.png"

df_laion_counts = pd.read_csv("counts_zipf.csv")

df_words = pd.read_csv("../../benchmark/v0-1/concepts.csv")

#df = df.drop(columns=["en"])

#corr_plot(df, "Correlation of counts of concepts in different languages", "corr_counts_concepts_")

#pdb.set_trace()

# sort by en column
def sort_by(df, column):
    df = df.sort_values(by=[column],ignore_index=True)
    df.reset_index(drop=True)
    return df

def sort_plot(df, language, plot_cols=[]):
    df = sort_by(df, language)
    sns.lineplot(data=df[plot_cols])
    plt.yscale("log")
    plt.ylim(100,10e7)
    plt.show()

def label_point_thresh(x, y, val, ax, x_thresh = 0, y_thresh = 0):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if x_thresh is not None and point['x'] > x_thresh and y_thresh is not None and point['y'] > y_thresh:
            ax.text(point['x']+.02, point['y'], str(point['val']))

# load the freqlist numbers for each word
# old way using my piece of shit freqlists
"""
LANGS = ["en","es","de","zh","ja","he","id"]
FREQLISTS = {}
for lang in LANGS:
    FREQLISTS[lang] = json.load(open(f"frequencylist/{lang}_2k.json"))

def get_frequency(language, word):
    return int(FREQLISTS[language].get(word, 0))

for lang in LANGS:
    df_words[lang + "_freq"] = df_words[lang].apply(lambda x: get_frequency(lang, x))
"""

LANGS = ["en","es","de","zh","ja","he","id"]
for lang in LANGS:
    df_words[lang + "_freq"] = df_words[lang].apply(lambda x: (word_frequency(x, lang)))

# create new df that combines all info
df = df_words
for lang in LANGS:
    df[lang + "_count"] = df_laion_counts[lang]


# used to generate 2024-02-04 plots of logcount-cccl score
# generate one scatterplot with each language in a different color
def plot_count_correct_model(model, laion_count = True):
    # results are in ConceptualCoverage.github.io/{model_code}/results_en.csv
    # for example, dallemega:
    df_results_model = pd.read_csv(f"../../../ConceptualCoverage.github.io/{model}/results_en.csv")
    # replace all '---' in df_results_model with 0
    df_results_model = df_results_model.replace('---',0).astype('float')
    #pdb.set_trace()
    print(df_results_model)
    for lang in LANGS:
        df[lang + "_correct_score"] = df_results_model[lang]
    df_paired_language = pd.DataFrame()
    for lang in ["en","es","de","zh","ja"]:
        add_df = df[[f"{lang}_count",f"{lang}_correct_score",f"{lang}_freq"]].rename(
            columns={f"{lang}_count":"count",f"{lang}_correct_score":"correct_score",f"{lang}_freq":"freq"}
        )
        print(add_df)
        print(lang)
        add_df["language"] = lang
        df_paired_language = pd.concat([
            df_paired_language,
            add_df
            ])
    df_paired_language["log_count"] = np.log(df_paired_language["count"] + sys.float_info.epsilon)
    df_paired_language["log_freq"] = np.log(df_paired_language["freq"] + sys.float_info.epsilon)
    print(df_paired_language)
    if laion_count:
        ax = sns.scatterplot(data=df_paired_language, x="log_count", y="correct_score", hue="language")
        ax.set_xlabel("Log count of occurrences in LAION-2b-en for ngram")
        ax.set_xlim(0,None)
    else:
        ax = sns.scatterplot(data=df_paired_language, x="log_freq", y="correct_score", hue="language")
        ax.set_xlabel("Log frequency of ngram in language-specific corpora")
        ax.set_xlim(-20,None)
    ax.set_title(f"Correctness score vs log count for {model}")
    ax.set_ylabel("CoCoCroLa Correctness score")
    ax.set_ylim(0,1)
    plt.show()


for model in ["demini", "demega", "sd1-1", "sd1-2", "sd1-4", "sd2", "dalle2"]:
    plot_count_correct_model(model)

"""
# generate individual figures of corr between the variables
#ranges = {"ja": 1e5, "zh": 40000, "he": 2000}
for language in ["en","de","es","zh","ja", "he", "id"]:
    df[f"{language}_count_log"] = np.log(df[f"{language}_count"] + sys.float_info.epsilon)
    ax = sns.regplot(data=df, x=f"{language}_count_log", y=f"{language}_correct_score")
    #plt.xscale("log")
    plt.xlim(max(0,df[f"{language}_count_log"].min()),None)
    plt.ylim(0,1)
    #plt.yscale("log")
    #plt.xscale("log")
    #label_point_thresh(df["es_freq"], df["es_count"], df["es"], plt.gca(), 300, -1)
    plt.show()
"""

"""
# example for generating figures 2024-02-03
#ranges = {"ja": 1e5, "zh": 40000, "he": 2000}
for language in ["en","de","es","zh","ja", "he", "id"]:
    df[f"{language}_freq_log"] = np.log(df[f"{language}_freq"] + sys.float_info.epsilon)
    df[f"{language}_count_log"] = np.log(df[f"{language}_count"] + sys.float_info.epsilon)
    ax = sns.regplot(data=df, x=f"{language}_freq_log", y=f"{language}_count_log")
    #plt.xscale("log")
    plt.ylim(max(0,df[f"{language}_count_log"].min()),None)
    #plt.yscale("log")
    #plt.xscale("log")
    #label_point_thresh(df["es_freq"], df["es_count"], df["es"], plt.gca(), 300, -1)
    plt.show()"""

"""
df_laion_counts = sort_by(df_laion_counts, "en")
print(df_laion_counts)


sns.lineplot(data=df_laion_counts[['en','es','ja']])
plt.yscale("log")
plt.ylim(100,10e7)
plt.show()
"""

"""sort_plot(df, "es", ["ja"])
sort_plot(df, "ja", ["es"])
sort_plot(df, "es", ["ja"])
sort_plot(df_laion_counts, "zh", ["ja"])
sort_plot(df_laion_counts, "ja", ["zh"])"""

# combine df with the scores