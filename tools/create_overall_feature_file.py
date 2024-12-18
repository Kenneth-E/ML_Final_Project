import csv
import pandas as pd
import numpy as np
import os
import sys
from collections import Counter


def get_object_size_gib(obj, decimal_places=2):
    bytes = sys.getsizeof(obj)
    gibibytes = bytes / (1024 * 1024 * 1024)
    return round(gibibytes, decimal_places)

def get_file_size_gib(filename, decimal_places=2):
    bytes = os.path.getsize(filename)
    gibibytes = bytes / (1024 * 1024 * 1024)
    return round(gibibytes, decimal_places)

def combine_csv_column_headers(csv_files):
    all_headers = set()
    for file in csv_files:
        # read only the header (first row)
        headers = pd.read_csv(file, nrows=0).columns.tolist()
        all_headers.update(headers)  # Add headers to the combined list
    combined_headers = list(all_headers)
    return combined_headers

def combine_csvs(input_files, output_file, labels, start_index, random_seed, max_rows):
    combined_df = pd.DataFrame()

    # Iterate over each input file and its corresponding label
    print("iterating over input files")
    df_list = []
    combined_headers = combine_csv_column_headers(input_files)
    print(f"combined_headers: {combined_headers} of length {len(combined_headers)}")
    for idx, (file, label) in enumerate(zip(input_files, labels)):
        print(f"reading file: {file}")
        df = pd.read_csv(file, encoding="utf-8")

        # Fill missing values with 0s
        df.fillna(0, inplace=True)
        missing_headers = list(set(combined_headers) - set(df.columns.tolist()))
        df_with_combined_headers = pd.DataFrame(0, columns=missing_headers, index=df.index)
        df = pd.concat([df, df_with_combined_headers], axis=1)

        df = df.astype(np.uint16) # convert np.float64 to np.uint16 fixed point representation
        print(f"dtype {df.dtypes}")
        df['label'] = label
        df['ID'] = range(len(combined_df) + start_index, len(combined_df) + len(df) + start_index)
        print(f"current size of df: {get_object_size_gib(df)} GiB of shape {df.shape} (with {len(df_list)} other df)")
        df_list.append(df)

    print("combining df_list")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"current size of combined_df: {get_object_size_gib(combined_df)} GiB")

    print(f"dtype {combined_df.dtypes}")

    # Fill missing values with 0s, again
    print("filling missing values")
    combined_df.fillna(0, inplace=True)
    print(f"current size of combined_df: {get_object_size_gib(combined_df)} GiB")

    # Randomize the order of the rows
    print("randomizing row order")
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Reset IDs to be in increasing order
    print("resetting IDs")
    combined_df['ID'] = range(1, len(combined_df) + 1)

    # Reorder columns to ensure 'ID' is first and 'label' is last
    print("reordering columns")
    column_order = ['ID'] + [col for col in combined_df.columns if col not in ['ID', 'label']] + ['label']
    combined_df = combined_df[column_order]

    print(f"truncating rows: {combined_df.shape[0]} rows to {max_rows} rows (None = do not truncate)")
    if max_rows is not None:
        combined_df = combined_df.head(max_rows)
    print(f"current size of combined_df: {get_object_size_gib(combined_df)} GiB")

    # Write to file
    print("writing to file")
    combined_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Feature file:")
    print(combined_df)
    print(f"File size: {get_file_size_gib(output_file)} GiB")

def main():
    print("beginning create_overall_feature_file.py")
    INPUT_FILES = [
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
        r"../data/adjacency_matrices/Bible_texts/wycliffe.csv",
        r"../data/adjacency_matrices/coding/Assembyly/ex1.csv",
        r"../data/adjacency_matrices/coding/Assembyly/ex2.csv",
        r"../data/adjacency_matrices/coding/Assembyly/ex3.csv",
        r"../data/adjacency_matrices/coding/cpp/ex1.csv",
        r"../data/adjacency_matrices/coding/cpp/ex2.csv",
        r"../data/adjacency_matrices/coding/cpp/ex3.csv",
        r"../data/adjacency_matrices/coding/GDScript/ex1.csv",
        r"../data/adjacency_matrices/coding/GDScript/ex2.csv",
        r"../data/adjacency_matrices/coding/GDScript/ex3.csv",
        r"../data/adjacency_matrices/coding/go/ex1.csv",
        r"../data/adjacency_matrices/coding/go/ex2.csv",
        r"../data/adjacency_matrices/coding/go/ex3.csv",
        r"../data/adjacency_matrices/coding/Haskell/ex1.csv",
        r"../data/adjacency_matrices/coding/Haskell/ex2.csv",
        r"../data/adjacency_matrices/coding/Haskell/ex3.csv",
        r"../data/adjacency_matrices/coding/HTML/ex1.csv",
        r"../data/adjacency_matrices/coding/HTML/ex2.csv",
        r"../data/adjacency_matrices/coding/HTML/ex3.csv",
        r"../data/adjacency_matrices/coding/Java/ex1.csv",
        r"../data/adjacency_matrices/coding/Java/ex2.csv",
        r"../data/adjacency_matrices/coding/Java/ex3.csv",
        r"../data/adjacency_matrices/coding/JS/ex1.csv",
        r"../data/adjacency_matrices/coding/JS/ex2.csv",
        r"../data/adjacency_matrices/coding/JS/ex3.csv",
        r"../data/adjacency_matrices/coding/JSON/ex1.csv",
        r"../data/adjacency_matrices/coding/JSON/ex2.csv",
        r"../data/adjacency_matrices/coding/JSON/ex3.csv",
        r"../data/adjacency_matrices/coding/Kotlin/ex1.csv",
        r"../data/adjacency_matrices/coding/Kotlin/ex2.csv",
        r"../data/adjacency_matrices/coding/Kotlin/ex3.csv",
        r"../data/adjacency_matrices/coding/Lua/ex1.csv",
        r"../data/adjacency_matrices/coding/Lua/ex2.csv",
        r"../data/adjacency_matrices/coding/Lua/ex3.csv",
        r"../data/adjacency_matrices/coding/OCaml/ex1.csv",
        r"../data/adjacency_matrices/coding/OCaml/ex2.csv",
        r"../data/adjacency_matrices/coding/OCaml/ex3.csv",
        r"../data/adjacency_matrices/coding/php/ex1.csv",
        r"../data/adjacency_matrices/coding/php/ex2.csv",
        r"../data/adjacency_matrices/coding/python/ex1.csv",
        r"../data/adjacency_matrices/coding/python/ex2.csv",
        r"../data/adjacency_matrices/coding/python/ex3.csv",
        r"../data/adjacency_matrices/coding/R/ex1.csv",
        r"../data/adjacency_matrices/coding/R/ex2.csv",
        r"../data/adjacency_matrices/coding/R/ex3.csv",
        r"../data/adjacency_matrices/coding/ruby/ex1.csv",
        r"../data/adjacency_matrices/coding/ruby/ex2.csv",
        r"../data/adjacency_matrices/coding/ruby/ex3.csv",
        r"../data/adjacency_matrices/coding/Rust/ex1.csv",
        r"../data/adjacency_matrices/coding/Rust/ex2.csv",
        r"../data/adjacency_matrices/coding/Rust/ex3.csv",
        r"../data/adjacency_matrices/coding/Shell/ex1.csv",
        r"../data/adjacency_matrices/coding/Shell/ex2.csv",
        r"../data/adjacency_matrices/coding/Shell/ex3.csv",
        r"../data/adjacency_matrices/coding/sql/ex1.csv",
        r"../data/adjacency_matrices/coding/sql/ex2.csv",
        r"../data/adjacency_matrices/coding/sql/ex3.csv",
        r"../data/adjacency_matrices/coding/Typescript/ex1.csv",
        r"../data/adjacency_matrices/coding/Typescript/ex2.csv",
        r"../data/adjacency_matrices/coding/Typescript/ex3.csv",
    ]
    LABELS = [
        "keyboard_mash",
        "keyboard_mash_dvorak_simulated",
        "random",
        "German",
        "English",
        "Cebuano",
        "Czech",
        "Dutch",
        "Esperanto",
        "Finnish",
        "French",
        "Haitian",
        "Latvian",
        "Norwegian",
        "Polish",
        "Portugese",
        "Swedish",
        "Modern Ecclesiastical Latin",
        "Middle English",
        "Assembly",
        "Assembly",
        "Assembly",
        "cpp",
        "cpp",
        "cpp",
        "GDScript",
        "GDScript",
        "GDScript",
        "go",
        "go",
        "go",
        "Haskell",
        "Haskell",
        "Haskell",
        "HTML",
        "HTML",
        "HTML",
        "Java",
        "Java",
        "Java",
        "JS",
        "JS",
        "JS",
        "JSON",
        "JSON",
        "JSON",
        "Kotlin",
        "Kotlin",
        "Kotlin",
        "Lua",
        "Lua",
        "Lua",
        "OCaml",
        "OCaml",
        "OCaml",
        "php",
        "php",
        "python",
        "python",
        "python",
        "R",
        "R",
        "R",
        "ruby",
        "ruby",
        "ruby",
        "Rust",
        "Rust",
        "Rust",
        "Shell",
        "Shell",
        "Shell",
        "sql",
        "sql",
        "sql",
        "Typescript",
        "Typescript",
        "Typescript",
    ]
    OUTPUT_FILE = r"../data/combined_features/combined_features.csv"
    START_INDEX = 0
    RANDOM_SEED = 42
    MAX_ROWS = 10_000 # None = unlimited rows, may run out of memory

    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED, MAX_ROWS)

    print("finished create_overall_feature_file.py")

def main_subset():
    print("beginning create_overall_feature_file.py")
    INPUT_FILES = [
        r"../data/adjacency_matrices/keyboard_mash_adjacency.csv",
        r"../data/adjacency_matrices/keyboard_mash_sentences_dvorak_simulated.csv",
        r"../data/adjacency_matrices/random_adjacency.csv",
        r"../data/adjacency_matrices/de_AusDerChronikaEinesFahrendenSchlers.csv",
        r"../data/adjacency_matrices/ACV.csv",
        r"../data/adjacency_matrices/DutSVV.csv",
        r"../data/adjacency_matrices/FrePGR.csv",
    ]
    LABELS = [
        "keyboard_mash",
        "keyboard_mash_dvorak_simulated",
        "random",
        "German",
        "English",
        "Dutch",
        "French",
    ]
    OUTPUT_FILE = r"../data/combined_features/combined_features_subset.csv"
    START_INDEX = 1
    RANDOM_SEED = 42
    MAX_ROWS = 10_000 # None = unlimited rows, may run out of memory

    print("beginning create_overall_feature_file.py")
    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED, MAX_ROWS)
    print("finished create_overall_feature_file.py")

def test():
    INPUT_FILES = [
        r"../data/test/test_adjacency.csv",
        r"../data/test/test_adjacency_2.csv"
    ]
    LABELS = ["test1", "test2"]
    OUTPUT_FILE = r"../data/test/test_combined_features.csv"
    START_INDEX = 1
    RANDOM_SEED = 42
    MAX_ROWS = None # None = unlimited rows, may run out of memory

    print("beginning create_overall_feature_file.py")
    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED, MAX_ROWS)
    print("finished create_overall_feature_file.py")

if __name__ == "__main__":
    main()