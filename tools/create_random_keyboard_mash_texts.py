import numpy as np

def qwerty_to_dvorak(string):
    qwerty = r"""qwertyuiop[]\asdfghjkl;'zxcvbnm,./QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?"""
    dvorak = r"""',.pyfgcrl/=\aoeuidhtns-;qjkxbmwvz"<>PYFGCRL?+|AOEUIDHTNS_:QJKXBMWVZ""" # yes, I installed the Dvorak keyboard just to type that
    translation = str.maketrans(qwerty, dvorak)
    return string.translate(translation)

def append_string_to_file(filename, string):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(string)

def create_random_texts(num_files, input_words_filenames, output_sentences_filenames, num_samples, max_sentences, random_seed, convert_to_dvorak_list):
    for file_idx in range(num_files):
        input_filename = input_words_filenames[file_idx]
        output_filename = output_sentences_filenames[file_idx]
        convert_to_dvorak = convert_to_dvorak_list[file_idx]

        # clear output file
        with open(output_filename, "w", encoding="utf-8") as output_file:
            pass

        print(f"{input_filename} -> {output_filename}")

        words = list()

        with open(input_filename, "r", encoding="utf-8") as words_file:
            for line in words_file:
                # line = words_file.readline()
                line = line.strip()
                word = line.split(",")[0]
                words.append(word)

        # Use the "Prose" fitted distribution from
        # https://www.researchgate.net/publication/318396023_How_will_text_size_influence_the_length_of_its_linguistic_constituents
        # negative binomial distribution
        # k = 2.17, p = 0.10
        # TODO: the negative_binomial() used is probably wrong, n is not k

        rng = np.random.default_rng(seed=random_seed)

        for _ in range(num_samples):
            num_sentences = rng.integers(low=1, high=max_sentences, endpoint=True)
            for _ in range(num_sentences):
                sentence = ""
                raw_num_words = rng.negative_binomial(n=2.17, p=0.10)
                num_words = np.max([1, int(raw_num_words)])
                for _ in range(num_words):
                    random_word_idx = rng.integers(low=0, high=len(words), endpoint=False)
                    random_word = words[random_word_idx]
                    if convert_to_dvorak:
                        random_word = qwerty_to_dvorak(random_word)
                    random_word = random_word.lower() # lowercase to make it less easily distinguishable
                    sentence += random_word
                    sentence += " "
                sentence += "\n"
                append_string_to_file(output_filename, sentence)

def main():
    print("beginning create_random_keyboard_mash_texts.py")

    NUM_SAMPLES = 10000
    MAX_SENTENCES = 1
    NUM_FILES = 3
    INPUT_WORDS_FILENAMES = [
        r"..\data\raw\words\keyboard_mash_words.txt",
        r"..\data\raw\words\keyboard_mash_words.txt",
        r"..\data\raw\words\random_words.txt",
    ]
    OUTPUT_SENTENCES_FILENAMES = [
        r"..\data\utterances\keyboard_mash_sentences.txt",
        r"..\data\utterances\keyboard_mash_sentences_dvorak_simulated.txt",
        r"..\data\utterances\random_sentences.txt",
    ]
    CONVERT_TO_DVORAK_LIST = [
        False,
        True,
        False,
    ]
    RANDOM_SEED = 42

    create_random_texts(NUM_FILES, INPUT_WORDS_FILENAMES, OUTPUT_SENTENCES_FILENAMES, NUM_SAMPLES, MAX_SENTENCES, RANDOM_SEED, CONVERT_TO_DVORAK_LIST)

    print("ending create_random_keyboard_mash_texts.py")

if __name__ == "__main__":
    main()