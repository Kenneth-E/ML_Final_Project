import csv
import pandas as pd

INPUT_FILES = [r"..\src\keyboard_mash_adjacency_features.csv", r"..\src\random_features.csv"]
LABELS = ["keyboard_mash", "random"]
OUTPUT_FILE = r"..\src\combined_features.csv"
START_INDEX = 1

'''
INPUT_FILES = [r"..\src\test_adjacency_features.csv", r"..\src\test_adjacency_features_2.csv"]
LABELS = ["test1", "test2"]
OUTPUT_FILE = r"..\src\test_combined_features.csv"
START_INDEX = 1
'''

def combine_csvs(input_files, output_file, labels):
    combined_df = pd.DataFrame()

    # Iterate over each input file and its corresponding label
    for idx, (file, label) in enumerate(zip(input_files, labels)):
        df = pd.read_csv(file)
        df['label'] = label
        df['ID'] = range(len(combined_df) + START_INDEX, len(combined_df) + len(df) + START_INDEX)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Fill missing values with 0s
    combined_df.fillna(0, inplace=True)

    # Randomize the order of the rows
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Reset IDs to be in increasing order
    combined_df['ID'] = range(1, len(combined_df) + 1)

    # Reorder columns to ensure 'ID' is first and 'label' is last
    column_order = ['ID'] + [col for col in combined_df.columns if col not in ['ID', 'label']] + ['label']
    combined_df = combined_df[column_order]

    # Write to file
    combined_df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

combine_csvs(INPUT_FILES, OUTPUT_FILE, LABELS)