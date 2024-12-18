import csv
import numpy as np
import re

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

def create_adjacency_matrix(num_files, input_sentences_filenames, output_adjacency_filenames, bible_remove_numbering_list, np_int_data_type):
    max_int = np.iinfo(np_int_data_type).max
    for file_idx in range(num_files):
        input_filename = input_sentences_filenames[file_idx]
        output_filename = output_adjacency_filenames[file_idx]
        bible_remove_numbering = bible_remove_numbering_list[file_idx]

        # clear output file
        with open(output_filename, "w", encoding="utf-8") as output_file:
            pass

        letter_pair_dict = dict()
        adjacency_matrix = []

        skipped_lines = 0

        with open(input_filename, "r", encoding="utf-8") as words_file:
            for line_idx, line in enumerate(words_file):
                # line = words_file.readline()
                sentence = line.strip()
                # skip blank utterances
                if sentence == "":
                    skipped_lines += 1
                    continue
                # remove beginning brackets, ex. "[1:1] In the beginning" -> "In the beginning" with simple regex
                if bible_remove_numbering:
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
                    sample_adjacency_list[letter_pair_idx] += 1
                # convert letter pair counts to fixed point
                for letter_pair_idx, count in enumerate(sample_adjacency_list):
                    letter_pair_proportion = count / num_letter_adjacencies
                    # use ceiling because it is rare to have a proportion of almost 1 so we can round up, but it will
                    # be common to have a proportion of almost 0, but it should not be rounded to 0
                    # yes, precision will be lost, and they won't sum to the max_int exactly
                    letter_pair_fixed_point = np.ceil(max_int * letter_pair_proportion)
                    letter_pair_fixed_point = int(letter_pair_fixed_point)
                    sample_adjacency_list[letter_pair_idx] = letter_pair_fixed_point
                adjacency_matrix.append(sample_adjacency_list)

        print(f"skipped_lines: {skipped_lines}")
        print(f"INPUT_FILENAME: {input_filename}")
        print(f"OUTPUT_FILENAME: {output_filename}")
        print(f"---------")

        append_csv_row_to_file(output_filename, letter_pair_dict.keys())
        append_csv_matrix_to_file(output_filename, adjacency_matrix)

def main():
    print("beginning create_adjacency_matrix.py")

    NUM_FILES = 18
    INPUT_SENTENCES_FILENAMES = [
        r"../data/utterances/keyboard_mash_sentences.txt",
        r"../data/utterances/keyboard_mash_sentences_dvorak_simulated.txt",
        r"../data/utterances/random_sentences.txt",
        r"../data/raw/books/de_AusDerChronikaEinesFahrendenSchlers.txt",
        r"../data/raw/Bible_texts/ACV.txt",
        r"../data/raw/Bible_texts/CebPinadayag.txt",
        r"../data/raw/Bible_texts/CzeCSP.txt",
        r"../data/raw/Bible_texts/DutSVV.txt",
        r"../data/raw/Bible_texts/Esperanto.txt",
        r"../data/raw/Bible_texts/FinBiblia.txt",
        r"../data/raw/Bible_texts/FrePGR.txt",
        r"../data/raw/Bible_texts/Haitian.txt",
        r"../data/raw/Bible_texts/LvGluck8.txt",
        r"../data/raw/Bible_texts/NorSMB.txt",
        r"../data/raw/Bible_texts/PolGdanska.txt",
        r"../data/raw/Bible_texts/PorNVA.txt",
        r"../data/raw/Bible_texts/Swe1917.txt",
        r"../data/raw/Bible_texts/nova_vulgata.txt",
    ]
    # temporarily use raw book sources until files can be parsed well TODO
    OUTPUT_ADJACENCY_FILENAMES = [
        r"../data/adjacency_matrices/keyboard_mash_adjacency.csv",
        r"../data/adjacency_matrices/keyboard_mash_sentences_dvorak_simulated.csv",
        r"../data/adjacency_matrices/random_adjacency.csv",
        r"../data/adjacency_matrices/de_AusDerChronikaEinesFahrendenSchlers.csv",
        r"../data/adjacency_matrices/ACV.csv",
        r"../data/adjacency_matrices/CebPinadayag.csv",
        r"../data/adjacency_matrices/CzeCSP.csv",
        r"../data/adjacency_matrices/DutSVV.csv",
        r"../data/adjacency_matrices/Esperanto.csv",
        r"../data/adjacency_matrices/FinBiblia.csv",
        r"../data/adjacency_matrices/FrePGR.csv",
        r"../data/adjacency_matrices/Haitian.csv",
        r"../data/adjacency_matrices/LvGluck8.csv",
        r"../data/adjacency_matrices/NorSMB.csv",
        r"../data/adjacency_matrices/PolGdanska.csv",
        r"../data/adjacency_matrices/PorNVA.csv",
        r"../data/adjacency_matrices/Swe1917.csv",
        r"../data/adjacency_matrices/nova_vulgata.csv",
    ]
    BIBLE_REMOVE_NUMBERING_LIST = [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    NP_INT_DATA_TYPE = np.uint16

    create_adjacency_matrix(NUM_FILES, INPUT_SENTENCES_FILENAMES, OUTPUT_ADJACENCY_FILENAMES,
                            BIBLE_REMOVE_NUMBERING_LIST, NP_INT_DATA_TYPE)

    print("finished create_adjacency_matrix.py")

if __name__ == "__main__":
    main()