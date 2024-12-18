import csv
import pandas as pd
import os

def combine_csvs(input_files, output_file, labels, start_index, random_seed):
    combined_df = pd.DataFrame()

    # Iterate over each input file and its corresponding label
    print("iterating over input files")
    for idx, (file, label) in enumerate(zip(input_files, labels)):
        df = pd.read_csv(file)
        df['label'] = label
        df['ID'] = range(len(combined_df) + start_index, len(combined_df) + len(df) + start_index)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Fill missing values with 0s
    print("filling missing values")
    combined_df.fillna(0, inplace=True)

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
    file_size_bytes = os.path.getsize(output_file)
    file_size_gibibytes = file_size_bytes / (1024 * 1024 * 1024)
    print("Feature file:")
    print(combined_df)
    print(f"File size: {round(file_size_gibibytes, 2)} GiB")

def main():
    print("beginning create_overall_feature_file.py")

    INPUT_FILES = [
        r"..\data\adjacency_matrices\keyboard_mash_adjacency.csv",
        r"..\data\adjacency_matrices\random_adjacency.csv",
        r"..\data\adjacency_matrices\de_AusDerChronikaEinesFahrendenSchlers.csv",
        r"..\data\adjacency_matrices\ACV.csv",
        r"..\data\adjacency_matrices\CebPinadayag.csv",
    ]
    LABELS = [
        "keyboard_mash",
        "random",
        "German",
        "English",
        "Cebuano",
    ]
    OUTPUT_FILE = r"..\data\combined_features\combined_features.csv"
    START_INDEX = 0
    RANDOM_SEED = 42

    r'''
    INPUT_FILES = [r"..\data\adjacency_matrices\test_adjacency_features.csv", r"..\data\adjacency_matrices\test_adjacency_features_2.csv"]
    LABELS = ["test1", "test2"]
    OUTPUT_FILE = r"..\data\combined_features\test_combined_features.csv"
    START_INDEX = 1
    '''

    combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS, START_INDEX, RANDOM_SEED)

    print("finished create_overall_feature_file.py")

if __name__ == "__main__":
    main()