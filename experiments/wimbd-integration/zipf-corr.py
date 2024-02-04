import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import click

def corr_plot(df, title, output_path):
    # generate the correlation plot
    corr = df.corr(method="spearman")
    print(corr)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.tight_layout()
    plt.title(title)
    plt.show()
    #plt.savefig(output_path + f"corr_{row}_{column}.png"

df = pd.read_csv("counts_zipf.csv")


#df = df.drop(columns=["en"])

#corr_plot(df, "Correlation of counts of concepts in different languages", "corr_counts_concepts_")

# sort by en column
def sort_plot(df, language, others=[]):
    df = df.sort_values(by=[language],ignore_index=True)    
    df.reset_index(drop=True)
    sns.lineplot(data=df[["en",language]+others])
    plt.yscale("log")
    plt.ylim(100,10e7)
    plt.show()


df = df.sort_values(by=["en"],ignore_index=True)
# sns multi-line plot log scale
df.reset_index(drop=True)
print(df)

sns.lineplot(data=df[['en','es','ja']])
plt.yscale("log")
plt.ylim(100,10e7)
plt.show()


sort_plot(df, "es", ["ja"])
sort_plot(df, "ja", ["es"])
sort_plot(df, "es", ["ja"])
sort_plot(df, "zh", ["ja"])
sort_plot(df, "ja", ["zh"])

# combine df with the scores