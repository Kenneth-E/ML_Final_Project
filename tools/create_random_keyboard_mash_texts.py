import numpy as np

def append_string_to_file(filename, string):
    with open(filename, "a") as file:
        file.write(string)

def create_random_texts(num_files, input_words_filenames, output_sentences_filenames, num_samples, max_sentences, random_seed):
    for file_idx in range(num_files):
        input_filename = input_words_filenames[file_idx]
        output_filename = output_sentences_filenames[file_idx]

        # clear output file
        with open(output_filename, "w") as output_file:
            pass

        print(input_filename)

        words = list()

        with open(input_filename, "r") as words_file:
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
                    sentence += words[random_word_idx]
                    sentence += " "
                sentence += "\n"
                append_string_to_file(output_filename, sentence)

def main():
    print("beginning create_random_keyboard_mash_texts.py")

    NUM_SAMPLES = 10000
    MAX_SENTENCES = 1
    NUM_FILES = 2
    INPUT_WORDS_FILENAMES = [r"..\data\raw\words\keyboard_mash_words.txt", r"..\data\raw\words\random_words.txt"]
    OUTPUT_SENTENCES_FILENAMES = [r"..\data\utterances\keyboard_mash_sentences.txt",
                                  r"..\data\utterances\random_sentences.txt"]
    RANDOM_SEED = 42

    create_random_texts(NUM_FILES, INPUT_WORDS_FILENAMES, OUTPUT_SENTENCES_FILENAMES, NUM_SAMPLES, MAX_SENTENCES, RANDOM_SEED)

    print("ending create_random_keyboard_mash_texts.py")

if __name__ == "__main__":
    main()