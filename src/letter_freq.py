import json
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import regex
import concurrent.futures

BOOK_FILES = [
    {
        "lang": "en",
        "filepath": "../data/raw/books/en_TheWarOfTheWorlds.txt"
    },
    {
        "lang": "de",
        "filepath": "../data/raw/books/de_AusDerChronikaEinesFahrendenSchlers.txt"
    },
    {
        "lang": "de",
        "filepath": "../data/raw/books/de_WestStlicherDivan.txt"
    },
    {
        "lang": "fr",
        "filepath": "../data/raw/books/fr_DeLaTerreALaLune.txt"
    },
    {
        "lang": "fr",
        "filepath": "../data/raw/books/fr_TourDuMond80Jours.txt"
    },
    {
        "lang": "es",
        "filepath": "../data/raw/books/es_DonQuijote.txt"
    },
    {
        "lang": "qwerty_mash",
        "filepath": "../data/utterances/keyboard_mash_sentences.txt"
    }
]

def get_language_letter_freqs(languages: list[str] = []):
    """
        languages: list of languages (i.e. ["en", "fr"] or ["es"] or [], etc.). Empty list (default) means all languages
        returns list of letter frequencies associated with input languages
    """
    FREQS_JSON_FILE_PATH = "../data/letter_frequencies.json" 

    with open(FREQS_JSON_FILE_PATH, encoding="utf8") as freqs_json_file:
        all_lang_tables: list = json.load(freqs_json_file)["languages"]

    # analyze all languages by default (or given empty list of languages)
    if not languages:
        return all_lang_tables

    lang_freq_tables = []
    for lang in languages:
        for x in all_lang_tables:
            if x["lang"] == lang:
                lang_freq_tables += [{
                    "lang": lang,
                    "table": x["freq_table"]
                }]
                break
        else:
            print(f"get_language_letter_freqs() couldn't find language '{ lang }' in '{ FREQS_JSON_FILE_PATH }'")

    return lang_freq_tables


def find_letter_freqs(sample: str) -> dict:
    """
        sample: sample text to analyze

        returns letter frequency table for the given sample text
    """

    freq_table: dict = {}
    total_character_count = 0

    for i in range(len(sample)):
        letter = sample[i]

        if letter not in freq_table:
            freq_table.update({letter: 0})

        freq_table[letter] += 1
        total_character_count += 1

    # normalize
    for letter in freq_table:
        freq_table[letter] /= total_character_count
    
    return freq_table

def compare_letter_freqs(freq_table: dict, compare_to: dict):
    """
        freq_table: frequency table #1 (typically use find_letter_freqs on a sample text)
        compare_to: frequency table #2 (typically one from get_language_letter_freqs)

        performs goodness of fit test by comparing two letter frequency tables (assumed
        to be observed from sample and a language's known frequency table)

        returns chi_sq value. Smaller values mean the frequency tables are more likely correlated
    """
    # goodness of fit test
    chi_sq = 0
    for letter, obs_freq in compare_to.items():
        if letter in freq_table:
            exp_freq = freq_table[letter]
            chi_sq += ((obs_freq - exp_freq) ** 2) / exp_freq
        else:
            # if expected frequency is zero, just say its really small...
            chi_sq += (obs_freq ** 2) / 0.00001

    # smaller chi_sq is better
    return chi_sq

def compare_all_languages(sample: str) -> tuple[str, float]:
    """
        sample: sample text to be analyzed

        performs goodness of fit test with sample text on all languages, returning the
        best fitting language with its chi^2 score

        returns best fitting language and associated chi^2 value
    """
    sample_freq_table = find_letter_freqs(sample)

    lang_freq_tables = get_language_letter_freqs()

    lang_chi_sqs = [{
        "lang": language["lang"],
        "value": compare_letter_freqs(sample_freq_table, language["freq_table"])
    } for language in lang_freq_tables]

    # print(f"language Chi^2 values:")
    # print(lang_chi_sqs)

    min_chi_sq_lang = lang_chi_sqs[0]
    for i in range(1, len(lang_chi_sqs)):
        if lang_chi_sqs[i]["value"] < min_chi_sq_lang["value"]:
            min_chi_sq_lang = lang_chi_sqs[i]

    # print("min Chi^2:")
    # print(min_chi_sq_lang)

    return min_chi_sq_lang["lang"], min_chi_sq_lang["value"]

def get_sample_text_from_books(lang: str = "", sample_len: int = 0):
    """
        lang: language of book to sample (i.e. "en", "fr", etc.). "" (empty string) = random language
        sample_len: length of sample text to return. 0 = whole text

        return tuple with sample text (of length sample_len) and language of sample text
    """
    # random language by default
    if not lang:
        # get list of options (uniform, each language occurs once)
        lang_options = []
        for book_file in BOOK_FILES:
            if book_file["lang"] not in lang_options:
                lang_options += [book_file["lang"]]

        # choose language
        lang = random.choice(lang_options)

    # choose book randomly from books of correct language
    books_of_lang: list = [book["filepath"] for book in BOOK_FILES if book["lang"] == lang]
    if not books_of_lang: raise Exception(f"No book of language { lang } found.")
    book: str = random.choice(books_of_lang)

    # choose text from book
    with open(book, 'r', encoding="utf8") as f:
        text = f.read().replace('\n', '')

    if sample_len == 0:
        return text, lang

    if sample_len >= len(text):
        # print(f"oversampling by { sample_len - len(text) }")
        return text, lang

    start_index = random.randint(0, len(text) - 1 - sample_len)
    sample = text[start_index:start_index + sample_len]

    # sanitize sample text (so only characters - no numbers or punctuation)
    sample = regex.sub(r'[\W\d_]+', '', sample)
    i = 1
    while len(sample) < sample_len:
        new_index = start_index + sample_len + i
        if new_index >= len(text):
            break
        sample += regex.sub(r'[\W\d_]+', '', text[new_index])
        i += 1

    return sample.upper(), lang

def process_sample_length(sample_len, samples_count=10000):
    accuracy = 0
    langs_checked = {}

    for _ in range(samples_count):
        sample_text, exp_lang = get_sample_text_from_books(sample_len=sample_len)
        obs_lang, _ = compare_all_languages(sample_text)
        if exp_lang == obs_lang:
            accuracy += 1

        if exp_lang not in langs_checked:
            langs_checked[exp_lang] = 0

        langs_checked[exp_lang] += 1

    accuracy /= samples_count
    return sample_len, accuracy

def main():
    max_sample_len = 1001
    steps = 200
    sample_len_step = int(max_sample_len / steps) 

    sample_lens = []
    test_data = []

    sample_len_range = range(sample_len_step, max_sample_len, sample_len_step)

    # Use a ProcessPoolExecutor for parallelization
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map the process_sample_length function to the range of sample lengths
        results = list(tqdm.tqdm(
            executor.map(process_sample_length, sample_len_range),
            total=len(sample_len_range),
            desc="Processing sample lengths"
        ))

    # Unpack the results
    for sample_len, accuracy in results:
        sample_lens.append(sample_len)
        test_data.append(accuracy)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(sample_lens, test_data, marker='', linestyle='-', color='b', label='Accuracy')

    # Customizing the plot
    plt.title('Sample Length vs. Accuracy', fontsize=14)
    plt.xlabel('Sample Length', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, max_sample_len + sample_len_step)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()

    # for i in range(10):
    #     print(get_sample_text_from_books(sample_len=100))