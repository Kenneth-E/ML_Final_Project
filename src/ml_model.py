from tkinter.tix import MAX
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
import pickle
import time
import json

def get_object_size_gib(obj, decimal_places=2):
    bytes = sys.getsizeof(obj)
    gibibytes = bytes / (1024 * 1024 * 1024)
    return round(gibibytes, decimal_places)

def get_file_size_gib(filename, decimal_places=2):
    bytes = os.path.getsize(filename)
    gibibytes = bytes / (1024 * 1024 * 1024)
    return round(gibibytes, decimal_places)

# TODO: PCA, wait & see output, validate that tree outputs are correct (not off-by-one), cross-validation
# TODO: Bible: what happens if all translations for the same sentence are in training? what if one in training, one in testing?
# TODO: is it important for each language to have about the same amount of data?
# TODO: agnostic to captialization? German nouns are capitalized, so maybe not?
#  https://scikit-learn.org/stable/modules/decomposition.html#pca
#  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
#  https://scikit-learn.org/stable/modules/cross_validation.html#group-cv
# return value format:
def ml_model(num_trials, test_ids, train_ids, max_depth_list, use_adaboost_list, adaboost_num_estimators_list, adaboost_learning_rate_list, combined_features_filename, return_value_save_filename):
    print(f"loading data ({get_file_size_gib(combined_features_filename)} GiB)...")
    dtypes = defaultdict(lambda: np.uint16)
    dtypes["ID"] = int
    dtypes["label"] = str
    dataset = pd.read_csv(combined_features_filename, index_col=0, parse_dates=True, dtype=dtypes)
    print("dataset:")
    print(dataset)
    print(f"size in RAM: {get_object_size_gib(dataset)} GiB")
    print("--------")

    print("dataset dtypes:")
    print(dataset.dtypes)
    print("--------")

    columns = dataset.columns.tolist()
    print("columns:")
    print(columns)
    print("--------")

    labels = dataset["label"]
    codes, uniques = pd.factorize(labels)
    print("uniques:")
    print(uniques)
    print("--------")

    ret_val = []

    for trial_idx in range(num_trials):
        start_time = time.time()
        trial_num = trial_idx
        test_id_slice = test_ids[trial_idx]
        train_id_slice = train_ids[trial_idx]
        max_depth = max_depth_list[trial_idx]
        use_adaboost = use_adaboost_list[trial_idx]
        adaboost_num_estimators = adaboost_num_estimators_list[trial_idx]
        adaboost_learning_rate = adaboost_learning_rate_list[trial_idx]
        # yay! manual type checking! it's almost as if static type checking is useful!
        if not isinstance(trial_num, int):
            raise TypeError('trial_num must be an instance of int')
        if not isinstance(test_id_slice, slice):
            raise TypeError('test_id_slice must be an instance of slice')
        if not isinstance(train_id_slice, slice):
            raise TypeError('train_id_slice must be an instance of slice')
        if not isinstance(max_depth, int) and not max_depth is None:
            raise TypeError('max_depth must be an instance of int or None')
        if not isinstance(use_adaboost, bool):
            raise TypeError('use_adaboost must be an instance of bool')
        if not isinstance(adaboost_num_estimators, int) and not adaboost_num_estimators is None:
            raise TypeError('adaboost_num_estimators must be an instance of int or None')
        if not isinstance(adaboost_learning_rate, float) and not adaboost_learning_rate is None:
            raise TypeError('adaboost_learning_rate must be an instance of float or None')

        print(f"running trial {trial_num}")
        print("--------")

        if use_adaboost:
            estimator = tree.DecisionTreeClassifier(max_depth = max_depth)
            clf = AdaBoostClassifier(n_estimators=adaboost_num_estimators, estimator=estimator, learning_rate=adaboost_learning_rate)
        else:
            clf = tree.DecisionTreeClassifier(max_depth = max_depth)

        train_y = codes[train_id_slice]
        print("train_y:")
        print(train_y)
        print("--------")

        X = dataset.drop("label", axis = 1)
        X_columns = X.columns.tolist()
        train_X = X[train_id_slice]
        print("train_X:")
        print(train_X)
        print("--------")

        print("fitting data...")
        clf = clf.fit(train_X, train_y)

        test_X = X[test_id_slice]
        print("test_X:")
        print(test_X)
        print("--------")

        test_y = codes[test_id_slice]
        print("test_y:")
        print(test_y)
        print("--------")

        predict_y = clf.predict(test_X)
        print("predict_y:")
        print(predict_y)
        print("--------")

        train_predict_y = clf.predict(train_X)
        print("train_predict_y:")
        print(train_predict_y)
        print("--------")

        train_matching_elements = np.sum(train_predict_y == train_y)
        train_total_elements = np.shape(train_y)[0]
        train_accuracy = train_matching_elements / train_total_elements
        print(f"train_matching_elements: {train_matching_elements}")
        print(f"train_total_elements: {train_total_elements}")
        print(f"train_accuracy: {train_accuracy}")
        print("--------")

        test_matching_elements = np.sum(predict_y == test_y)
        test_total_elements = np.shape(predict_y)[0]
        test_accuracy = test_matching_elements / test_total_elements
        print(f"test_matching_elements: {test_matching_elements}")
        print(f"test_total_elements: {test_total_elements}")
        print(f"test_accuracy: {test_accuracy}")
        print("--------")

        print("test params:")
        print(f"trial_num: {trial_num}")
        print(f"test_id_slice: {test_id_slice}")
        print(f"train_id_slice: {train_id_slice}")
        print(f"max_depth: {max_depth}")
        print(f"use_adaboost: {use_adaboost}")
        print(f"adaboost_num_estimators: {adaboost_num_estimators}")
        print(f"adaboost_learning_rate: {adaboost_learning_rate}")
        print("--------")

        if not use_adaboost:
            tree.plot_tree(clf)
            plt.savefig(r"../data/tree_diagrams/tree_raw_" + str(trial_num) + r".svg")
            plt.savefig(r"../data/tree_diagrams/tree_raw_" + str(trial_num) + r".png")

            tree.plot_tree(clf, class_names=uniques, feature_names=X_columns)
            plt.savefig(r"../data/tree_diagrams/tree_" + str(trial_num) + r".svg")
            plt.savefig(r"../data/tree_diagrams/tree_" + str(trial_num) + r".png")

            with open(r"../data/tree_diagrams/tree_raw_" + str(trial_num) + r".txt", "w", encoding="utf8") as file:
                tree_as_text = tree.export_text(clf, max_depth=100, show_weights=True)
                file.write(tree_as_text)

            with open(r"../data/tree_diagrams/tree_" + str(trial_num) + r".txt", "w", encoding="utf8") as file:
                tree_as_text = tree.export_text(clf, class_names=uniques, feature_names=X_columns, max_depth=100, show_weights=True)
                file.write(tree_as_text)

        pickle_filename = r"../data/tree_diagrams/tree_" + str(trial_num) + r".pickle"
        with open(pickle_filename, 'wb') as pickle_file:
            pickle.dump(clf, pickle_file)
        # to load:
        # with open(pickle_filename, 'rb') as f:
        #     data = pickle.load(f)
        end_time = time.time()
        execution_seconds = end_time - start_time
        ret_val_row = {
            "trial_num": trial_num,
            "max_depth": max_depth,
            # "test_X": str(test_X),
            # "test_y": str(test_y),
            # "train_matching_elements": str(train_matching_elements),
            # "train_total_elements": str(train_total_elements),
            "train_accuracy": train_accuracy,
            # "test_matching_elements": str(test_matching_elements),
            # "test_total_elements": str(test_total_elements),
            "test_accuracy": test_accuracy,
            # "execution_seconds": str(execution_seconds),
            "use_adaboost": use_adaboost,
            "adaboost_num_estimators": adaboost_num_estimators,
            "adaboost_learning_rate": adaboost_learning_rate,
            "execution_seconds": execution_seconds,
        }
        ret_val.append(ret_val_row)
    with open(return_value_save_filename, "w", encoding="utf-8") as file:
        json.dump(ret_val, file)
    return ret_val


def main():
    TEST_IDS = [slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000)]
    TRAIN_IDS = [slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000)]
    MAX_DEPTH = [5, 10, 20, 40, None, 2] # None = unlimited depth
    USE_ADABOOST = [False, False, False, False, False, True]
    ADABOOST_NUM_ESTIMATORS = [None, None, None, None, None, 100]
    ADABOOST_LEARNING_RATE = [None, None, None, None, None, 1.0]
    NUM_TRIALS = 6
    COMBINED_FEATURES_FILENAME = '../data/combined_features/combined_features.csv'
    RETURN_VALUE_SAVE_FILENAME = '../data/tree_diagrams/10k.json'

    ret_val = ml_model(NUM_TRIALS, TEST_IDS, TRAIN_IDS, MAX_DEPTH, USE_ADABOOST, ADABOOST_NUM_ESTIMATORS, ADABOOST_LEARNING_RATE, COMBINED_FEATURES_FILENAME, RETURN_VALUE_SAVE_FILENAME)
    print(f"ret_val: {ret_val}")
    plot_results(ret_val=ret_val)

def main_subset():
    TEST_IDS = [slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000)]
    TRAIN_IDS = [slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000), slice(1_000, 10_000)]
    MAX_DEPTH = [5, 10, 20, 40, None, 1] # None = unlimited depth
    USE_ADABOOST = [False, False, False, False, False, True]
    ADABOOST_NUM_ESTIMATORS = [None, None, None, None, None, 100]
    ADABOOST_LEARNING_RATE = [None, None, None, None, None, 1.0]
    NUM_TRIALS = 6
    COMBINED_FEATURES_FILENAME = '../data/combined_features/combined_features_subset.csv'

    ret_val = ml_model(NUM_TRIALS, TEST_IDS, TRAIN_IDS, MAX_DEPTH, USE_ADABOOST, ADABOOST_NUM_ESTIMATORS, ADABOOST_LEARNING_RATE, COMBINED_FEATURES_FILENAME)
    print(f"ret_val: {ret_val}")
    plot_results(ret_val=ret_val)

def plot_results(ret_val):

    # Test accuracy vs. maximum depth

    df = pd.DataFrame(ret_val)

    print(df.to_markdown(index=False))

    df.to_csv("../data/results/results_table.csv", index=False)

if __name__ == "__main__":
    main()