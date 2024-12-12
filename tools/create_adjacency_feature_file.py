import csv
import numpy as np

NUM_FILES = 2
INPUT_ADJACENCY_FILENAMES = [r"..\src\keyboard_mash_adjacency.csv", r"..\src\random_adjacency.csv"]
OUTPUT_FEATURE_FILENAME = r"..\src\adjacency_features.txt"

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

for file_idx in range(NUM_FILES):
    INPUT_FILENAME = INPUT_ADJACENCY_FILENAMES[file_idx]

    csv_reader = csv.reader(INPUT_FILENAME)
    for row in csv_reader:
        letter_1 = row[0]
        letter_2 = row[1]
        count_list = row[2]

    letter_pair = (letter_1, letter_2)


# clear output file
with open(OUTPUT_FEATURE_FILENAME, "w") as output_file:
    pass