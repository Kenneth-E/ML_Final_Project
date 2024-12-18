from collections import defaultdict
from urllib.request import urlretrieve
import gzip
import shutil
import json
from nltk import ngrams
import csv


def bigram2(string):
    letters = [*string]
    return ngrams(letters, 2)

def download():
    url = 'https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz'
    dst = '../data/wiktionary/raw-wiktextract-data.jsonl.gz'
    urlretrieve(url, dst)

def unzip():
    with gzip.open("../data/wiktionary/raw-wiktextract-data.jsonl.gz", "rb") as f_in:
        with open("../data/wiktionary/raw-wiktextract-data.jsonl/raw-wiktextract-data.jsonl", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def count_lines():
    with open("../data/wiktionary/raw-wiktextract-data.jsonl/raw-wiktextract-data.jsonl", "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            idx += 1
    return idx


def parse():
    bigram_counts = defaultdict(lambda: defaultdict(int))
    num_lines = 9_883_876
    # num_lines = count_lines()
    # i counted it already so no need to do it again
    with open("../data/wiktionary/raw-wiktextract-data.jsonl/raw-wiktextract-data.jsonl", "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            data = json.loads(line)
            if "word" in data and "lang" in data:
                lang = data["lang"]
                word = data["word"]
                padded_word = " " + word + " "
                bigrams = set(bigram2(padded_word))
                for bigram in bigrams:
                    bigram_counts[lang][bigram] += 1
            idx += 1
            if idx % 20_000 == 0:
                print(f"parsing line {idx}/{num_lines} ({idx/num_lines * 100:.2f}%)")

    print("writing to disk")
    with open("../data/wiktionary/bigram_counts.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Bigram', 'Count'])

        for lang, bigrams in bigram_counts.items():
            for bigram, count in bigrams.items():
                writer.writerow([lang, f"({bigram[0]}, {bigram[1]})", count])
    print("done")

if __name__ == '__main__':
    parse()