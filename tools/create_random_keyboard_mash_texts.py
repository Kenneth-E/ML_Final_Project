import numpy as np

NUM_SAMPLES = 10000
MAX_SENTENCES = 1
NUM_FILES = 2
INPUT_WORDS_FILENAMES = [r"..\src\keyboard_mash_words.txt", r"..\src\random_words.txt"]
OUTPUT_SENTENCES_FILENAMES = [r"..\src\keyboard_mash_sentences.txt", r"..\src\random_sentences.txt"]

def append_string_to_file(filename, string):
    with open(filename, "a") as file:
        file.write(string)

for file_idx in range(NUM_FILES):
    INPUT_FILENAME = INPUT_WORDS_FILENAMES[file_idx]
    OUTPUT_FILENAME = OUTPUT_SENTENCES_FILENAMES[file_idx]

    # clear output file
    with open(OUTPUT_FILENAME, "w") as output_file:
        pass

    print(INPUT_FILENAME)

    words = list()

    with open(INPUT_FILENAME, "r") as words_file:
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

    rng = np.random.default_rng()

    for _ in range(NUM_SAMPLES):
        num_sentences = rng.integers(low=1, high=MAX_SENTENCES, endpoint=True)
        for _ in range(num_sentences):
            sentence = ""
            raw_num_words = rng.negative_binomial(n=2.17, p=0.10)
            num_words = np.max([1, int(raw_num_words)])
            for _ in range(num_words):
                random_word_idx = rng.integers(low=0, high=len(words), endpoint=False)
                sentence += words[random_word_idx]
                sentence += " "
            sentence += "\n"
            append_string_to_file(OUTPUT_FILENAME, sentence)