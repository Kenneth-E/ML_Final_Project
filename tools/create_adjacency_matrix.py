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

def create_adjacency_matrix(num_files, input_sentences_filenames, output_adjacency_filenames, bible_remove_numbering_list, np_int_data_type, convert_to_proportions):
    max_int = np.iinfo(np_int_data_type).max
    for file_idx in range(num_files):
        print("building adjacency matrix")
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
                if convert_to_proportions:
                    for letter_pair_idx, count in enumerate(sample_adjacency_list):
                        letter_pair_proportion = count / num_letter_adjacencies
                        # use ceiling because it is rare to have a proportion of almost 1 so we can round up, but it will
                        # be common to have a proportion of almost 0, but it should not be rounded to 0
                        # yes, precision will be lost, and they won't sum to the max_int exactly
                        letter_pair_fixed_point = np.ceil(max_int * letter_pair_proportion)
                        letter_pair_fixed_point = int(letter_pair_fixed_point)
                        sample_adjacency_list[letter_pair_idx] = letter_pair_fixed_point
                adjacency_matrix.append(sample_adjacency_list)

        print("writing to disk")
        append_csv_row_to_file(output_filename, letter_pair_dict.keys())
        append_csv_matrix_to_file(output_filename, adjacency_matrix)

        print("summary")
        print(f"skipped_lines: {skipped_lines}")
        print(f"INPUT_FILENAME: {input_filename}")
        print(f"OUTPUT_FILENAME: {output_filename}")
        print(f"---------")

def main():
    print("beginning create_adjacency_matrix.py")

    NUM_FILES = 78
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
        r"../data/raw/Bible_texts/wycliffe.txt",
        r"../data/raw/coding/Assembyly/ex1.s",
        r"../data/raw/coding/Assembyly/ex2.s",
        r"../data/raw/coding/Assembyly/ex3.s",
        r"../data/raw/coding/cpp/ex1.cpp",
        r"../data/raw/coding/cpp/ex2.cpp",
        r"../data/raw/coding/cpp/ex3.cpp",
        r"../data/raw/coding/GDScript/ex1.gd",
        r"../data/raw/coding/GDScript/ex2.gd",
        r"../data/raw/coding/GDScript/ex3.gd",
        r"../data/raw/coding/go/ex1.go",
        r"../data/raw/coding/go/ex2.go",
        r"../data/raw/coding/go/ex3.go",
        r"../data/raw/coding/Haskell/ex1.hs",
        r"../data/raw/coding/Haskell/ex2.hs",
        r"../data/raw/coding/Haskell/ex3.hs",
        r"../data/raw/coding/HTML/ex1.html",
        r"../data/raw/coding/HTML/ex2.html",
        r"../data/raw/coding/HTML/ex3.html",
        r"../data/raw/coding/Java/ex1.java",
        r"../data/raw/coding/Java/ex2.java",
        r"../data/raw/coding/Java/ex3.java",
        r"../data/raw/coding/JS/ex1.js",
        r"../data/raw/coding/JS/ex2.js",
        r"../data/raw/coding/JS/ex3.js",
        r"../data/raw/coding/JSON/ex1.json",
        r"../data/raw/coding/JSON/ex2.json",
        r"../data/raw/coding/JSON/ex3.json",
        r"../data/raw/coding/Kotlin/ex1.kt",
        r"../data/raw/coding/Kotlin/ex2.kt",
        r"../data/raw/coding/Kotlin/ex3.kt",
        r"../data/raw/coding/Lua/ex1.lua",
        r"../data/raw/coding/Lua/ex2.lua",
        r"../data/raw/coding/Lua/ex3.lua",
        r"../data/raw/coding/OCaml/ex1.ml",
        r"../data/raw/coding/OCaml/ex2.ml",
        r"../data/raw/coding/OCaml/ex3.ml",
        r"../data/raw/coding/php/ex1.php",
        r"../data/raw/coding/php/ex2.php",
        r"../data/raw/coding/python/ex1.py",
        r"../data/raw/coding/python/ex2.py",
        r"../data/raw/coding/python/ex3.py",
        r"../data/raw/coding/R/ex1.r",
        r"../data/raw/coding/R/ex2.r",
        r"../data/raw/coding/R/ex3.r",
        r"../data/raw/coding/ruby/ex1.rb",
        r"../data/raw/coding/ruby/ex2.rb",
        r"../data/raw/coding/ruby/ex3.rb",
        r"../data/raw/coding/Rust/ex1.rs",
        r"../data/raw/coding/Rust/ex2.rs",
        r"../data/raw/coding/Rust/ex3.rs",
        r"../data/raw/coding/Shell/ex1.sh",
        r"../data/raw/coding/Shell/ex2.sh",
        r"../data/raw/coding/Shell/ex3.sh",
        r"../data/raw/coding/sql/ex1.sql",
        r"../data/raw/coding/sql/ex2.sql",
        r"../data/raw/coding/sql/ex3.sql",
        r"../data/raw/coding/Typescript/ex1.ts",
        r"../data/raw/coding/Typescript/ex2.ts",
        r"../data/raw/coding/Typescript/ex3.ts",
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
        r"../data/adjacency_matrices/wycliffe.csv",
        r"../data/adjacency_matrices/Assembyly_ex1.csv",
        r"../data/adjacency_matrices/Assembyly_ex2.csv",
        r"../data/adjacency_matrices/Assembyly_ex3.csv",
        r"../data/adjacency_matrices/cpp_ex1.csv",
        r"../data/adjacency_matrices/cpp_ex2.csv",
        r"../data/adjacency_matrices/cpp_ex3.csv",
        r"../data/adjacency_matrices/GDScript_ex1.csv",
        r"../data/adjacency_matrices/GDScript_ex2.csv",
        r"../data/adjacency_matrices/GDScript_ex3.csv",
        r"../data/adjacency_matrices/go_ex1.csv",
        r"../data/adjacency_matrices/go_ex2.csv",
        r"../data/adjacency_matrices/go_ex3.csv",
        r"../data/adjacency_matrices/Haskell_ex1.csv",
        r"../data/adjacency_matrices/Haskell_ex2.csv",
        r"../data/adjacency_matrices/Haskell_ex3.csv",
        r"../data/adjacency_matrices/HTML_ex1.csv",
        r"../data/adjacency_matrices/HTML_ex2.csv",
        r"../data/adjacency_matrices/HTML_ex3.csv",
        r"../data/adjacency_matrices/Java_ex1.csv",
        r"../data/adjacency_matrices/Java_ex2.csv",
        r"../data/adjacency_matrices/Java_ex3.csv",
        r"../data/adjacency_matrices/JS_ex1.csv",
        r"../data/adjacency_matrices/JS_ex2.csv",
        r"../data/adjacency_matrices/JS_ex3.csv",
        r"../data/adjacency_matrices/JSON_ex1.csv",
        r"../data/adjacency_matrices/JSON_ex2.csv",
        r"../data/adjacency_matrices/JSON_ex3.csv",
        r"../data/adjacency_matrices/Kotlin_ex1.csv",
        r"../data/adjacency_matrices/Kotlin_ex2.csv",
        r"../data/adjacency_matrices/Kotlin_ex3.csv",
        r"../data/adjacency_matrices/Lua_ex1.csv",
        r"../data/adjacency_matrices/Lua_ex2.csv",
        r"../data/adjacency_matrices/Lua_ex3.csv",
        r"../data/adjacency_matrices/OCaml_ex1.csv",
        r"../data/adjacency_matrices/OCaml_ex2.csv",
        r"../data/adjacency_matrices/OCaml_ex3.csv",
        r"../data/adjacency_matrices/php_ex1.csv",
        r"../data/adjacency_matrices/php_ex2.csv",
        r"../data/adjacency_matrices/python_ex1.csv",
        r"../data/adjacency_matrices/python_ex2.csv",
        r"../data/adjacency_matrices/python_ex3.csv",
        r"../data/adjacency_matrices/R_ex1.csv",
        r"../data/adjacency_matrices/R_ex2.csv",
        r"../data/adjacency_matrices/R_ex3.csv",
        r"../data/adjacency_matrices/ruby_ex1.csv",
        r"../data/adjacency_matrices/ruby_ex2.csv",
        r"../data/adjacency_matrices/ruby_ex3.csv",
        r"../data/adjacency_matrices/Rust_ex1.csv",
        r"../data/adjacency_matrices/Rust_ex2.csv",
        r"../data/adjacency_matrices/Rust_ex3.csv",
        r"../data/adjacency_matrices/Shell_ex1.csv",
        r"../data/adjacency_matrices/Shell_ex2.csv",
        r"../data/adjacency_matrices/Shell_ex3.csv",
        r"../data/adjacency_matrices/sql_ex1.csv",
        r"../data/adjacency_matrices/sql_ex2.csv",
        r"../data/adjacency_matrices/sql_ex3.csv",
        r"../data/adjacency_matrices/Typescript_ex1.csv",
        r"../data/adjacency_matrices/Typescript_ex2.csv",
        r"../data/adjacency_matrices/Typescript_ex3.csv",
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
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    NP_INT_DATA_TYPE = np.uint16
    CONVERT_TO_PROPORTIONS = True

    create_adjacency_matrix(NUM_FILES, INPUT_SENTENCES_FILENAMES, OUTPUT_ADJACENCY_FILENAMES,
                            BIBLE_REMOVE_NUMBERING_LIST, NP_INT_DATA_TYPE, CONVERT_TO_PROPORTIONS)

    print("finished create_adjacency_matrix.py")

if __name__ == "__main__":
    main()