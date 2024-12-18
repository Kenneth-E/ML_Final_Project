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

def combine_csvs(input_files, output_file, labels, start_index, random_seed):
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

    # Write to file
    print("writing to file")
    combined_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print("Feature file:")
    print(combined_df)
    print(f"File size: {get_file_size_gib(output_file)} GiB")

def main():
    print("beginning create_overall_feature_file.py")
    INPUT_FILES = [
        r"..\data\adjacency_matrices\keyboard_mash_adjacency.csv",
        r"..\data\adjacency_matrices\keyboard_mash_sentences_dvorak_simulated.csv",
        r"..\data\adjacency_matrices\random_adjacency.csv",
        r"..\data\adjacency_matrices\de_AusDerChronikaEinesFahrendenSchlers.csv",
        r"..\data\adjacency_matrices\ACV.csv",
        r"..\data\adjacency_matrices\CebPinadayag.csv",
        r"..\data\adjacency_matrices\CzeCSP.csv",
        r"..\data\adjacency_matrices\DutSVV.csv",
        r"..\data\adjacency_matrices\Esperanto.csv",
        r"..\data\adjacency_matrices\FinBiblia.csv",
        r"..\data\adjacency_matrices\FrePGR.csv",
        r"..\data\adjacency_matrices\Haitian.csv",
        r"..\data\adjacency_matrices\LvGluck8.csv",
        r"..\data\adjacency_matrices\NorSMB.csv",
        r"..\data\adjacency_matrices\PolGdanska.csv",
        r"..\data\adjacency_matrices\PorNVA.csv",
        r"..\data\adjacency_matrices\Swe1917.csv",
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
        "Spanish",
        "Finnish",
        "French",
        "Haitian",
        "Latvian",
        "Norwegian",
        "Polish",
        "Portugese",
        "Swedish",
    ]
    OUTPUT_FILE = r"..\data\combined_features\combined_features.csv"
    START_INDEX = 0
    RANDOM_SEED = 42

    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED)

    print("finished create_overall_feature_file.py")
def test():
    INPUT_FILES = [r"..\data\test\test_adjacency.csv", r"..\data\test\test_adjacency_2.csv"]
    LABELS = ["test1", "test2"]
    OUTPUT_FILE = r"..\data\test\test_combined_features.csv"
    START_INDEX = 1
    RANDOM_SEED = 42

    print("beginning create_overall_feature_file.py")
    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED)
    print("finished create_overall_feature_file.py")

if __name__ == "__main__":
    main()