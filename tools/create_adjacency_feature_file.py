import csv
import numpy as np
'''
NUM_FILES = 1
INPUT_ADJACENCY_FILENAMES = [r"..\src\test_adjacency.csv"]
OUTPUT_FEATURE_FILENAME = [r"..\src\test_adjacency_features.csv"]
'''

# WARNING: deprecated, do not use

NUM_FILES = 3
INPUT_ADJACENCY_FILENAMES = [r"..\src\test_adjacency.csv", r"..\src\keyboard_mash_adjacency.csv", r"..\src\random_adjacency.csv"]
OUTPUT_FEATURE_FILENAME = [r"..\src\test_adjacency_features.csv", r"..\src\keyboard_mash_adjacency_features.csv", r"..\src\random_features.csv"]

letter_pairs_to_feature_id = dict()
features_outcome_matrix = np.array([])
'''
# get all of the applicable letter pairs

for file_idx in range(NUM_FILES):
    INPUT_FILENAME = INPUT_ADJACENCY_FILENAMES[file_idx]

    csv_reader = csv.reader(INPUT_FILENAME)
    for row in csv_reader:
        letter_1 = row[0]
        letter_2 = row[1]
        count_list = row[2]

        letter_pair = (letter_1, letter_2)
        if letter_pair not in letter_pairs_to_feature_id:
            letter_pairs_to_feature_id[letter_pair] = len(letter_pairs_to_feature_id)
'''
total_features = len(letter_pairs_to_feature_id)

def process_csv(input_filename, output_filename):

    with open(input_filename, 'r') as infile:
        csv_reader = csv.reader(infile)

        # Read all rows into a list
        rows = list(csv_reader)

        header = sorted(rows[0]) if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        max_columns = max(len(row) for row in data_rows) if data_rows else len(header)

        processed_rows = []

        if header:
            processed_rows.append(["ID"] + header)

        for idx, row in enumerate(data_rows):
            processed_row = [idx] + [float(value) if value else 0.0 for value in row]

            # Pad with zeros to ensure all rows have the same number of columns
            processed_row += [0.0] * (max_columns - len(row))

            processed_rows.append(processed_row)

    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)

for file_idx in range(NUM_FILES):
    process_csv(INPUT_ADJACENCY_FILENAMES[file_idx], OUTPUT_FEATURE_FILENAME[file_idx]);