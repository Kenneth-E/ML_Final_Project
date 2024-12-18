import csv
import numpy as np
import re

NUM_FILES = 5
INPUT_SENTENCES_FILENAMES = [
    r"..\data\utterances\keyboard_mash_sentences.txt",
    r"..\data\utterances\random_sentences.txt",
    r"..\data\raw\books\de_AusDerChronikaEinesFahrendenSchlers.txt",
    r"..\data\raw\Bible_texts\ACV.txt",
    r"..\data\raw\Bible_texts\CebPinadayag.txt",
]
# temporarily use raw book sources until files can be parsed well TODO
OUTPUT_ADJACENCY_FILENAMES = [
    r"..\data\adjacency_matrices\keyboard_mash_adjacency.csv",
    r"..\data\adjacency_matrices\random_adjacency.csv",
    r"..\data\adjacency_matrices\de_AusDerChronikaEinesFahrendenSchlers.csv",
    r"..\data\adjacency_matrices\ACV.csv",
    r"..\data\adjacency_matrices\CebPinadayag.csv",
]
BIBLE_REMOVE_NUMBERING_LIST = [
    False,
    False,
    False,
    True,
    True
]

def append_string_to_file(filename, string):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(string)

def append_csv_row_to_file(filename, list):
    with open(filename, "a", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(list)

def append_csv_matrix_to_file(filename, matrix):
    with open(filename, "a", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(matrix)


for file_idx in range(NUM_FILES):
    INPUT_FILENAME = INPUT_SENTENCES_FILENAMES[file_idx]
    OUTPUT_FILENAME = OUTPUT_ADJACENCY_FILENAMES[file_idx]
    BIBLE_REMOVE_NUMBERING = BIBLE_REMOVE_NUMBERING_LIST[file_idx]

    # clear output file
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as output_file:
        pass

    letter_pair_dict = dict()
    adjacency_matrix = []

    skipped_lines = 0

    with open(INPUT_FILENAME, "r", encoding="utf-8") as words_file:
        for line_idx, line in enumerate(words_file):
            # line = words_file.readline()
            sentence = line.strip()
            # skip blank utterances
            if sentence == "":
                skipped_lines += 1
                continue
            # remove beginning brackets, ex. "[1:1] In the beginning" -> "In the beginning" with simple regex
            if BIBLE_REMOVE_NUMBERING:
                sentence = re.sub(r'^\[.*?\]\s*', '', sentence)
            sample_adjacency_list = np.array([])
            num_letter_adjacencies = len(sentence) - 1
            for idx in range(len(sentence) - 1):
                letter_1 = sentence[idx]
                letter_2 = sentence[idx + 1]
                letter_pair = (letter_1, letter_2)

                letter_pair_idx = None
                if letter_pair in letter_pair_dict:
                    letter_pair_idx = letter_pair_dict[letter_pair]
                else:
                    letter_pair_idx = len(letter_pair_dict)
                    letter_pair_dict[letter_pair] = len(letter_pair_dict)

                if letter_pair_idx + 1 >= len(sample_adjacency_list):
                    sample_adjacency_list = np.pad(sample_adjacency_list, (0, len(letter_pair_dict) - len(sample_adjacency_list)), constant_values=(0,0))
                sample_adjacency_list[letter_pair_idx] += 1 / num_letter_adjacencies
            adjacency_matrix.append(sample_adjacency_list)

    print(f"skipped_lines: {skipped_lines}")
    print(f"INPUT_FILENAME: {INPUT_FILENAME}")
    print(f"---------")

    append_csv_row_to_file(OUTPUT_FILENAME, letter_pair_dict.keys())
    append_csv_matrix_to_file(OUTPUT_FILENAME, adjacency_matrix)