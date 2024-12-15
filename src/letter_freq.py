import json

def get_language_letter_freqs(languages: list[str] = []):
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
    # goodness of fit test
    chi_sq = 0
    for letter, obs_freq in compare_to.items():
        # ignore letters not in dictionary (i.e. numbers, commas, diacritics in english, etc.)
        if letter in freq_table:
            exp_freq = freq_table[letter]
            chi_sq += ((obs_freq - exp_freq) ** 2) / exp_freq

    # smaller chi_sq is better
    return chi_sq

def compare_all_languages(sample: str):
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

    return min_chi_sq_lang

def main():
    text_files = [
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
        }
    ]

    for file in text_files:
        with open(file["filepath"], 'r', encoding="utf8") as f:
            text = f.read().replace('\n', '')

        print(f"Exp: { file["lang"] }, Obs: { compare_all_languages(text)["lang"] }")

if __name__ == "__main__":
    main()