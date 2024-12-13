import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

TEST_IDS = [slice(0, 100), slice(0, 100), slice(0, 100), slice(0, 100)]
TRAIN_IDS = [slice(100, 20000), slice(100, 20000), slice(100, 20000), slice(100, 20000)]
MAX_DEPTH = [5, 10, 20, 40]
NUM_TRIALS = 4

def id3_only(test_id_slice, train_id_slice, max_depth, trial_num):
    if not isinstance(test_id_slice, slice):
        raise TypeError('test_id_slice must be an instance of slice')
    if not isinstance(train_id_slice, slice):
        raise TypeError('train_id_slice must be an instance of slice')
    if not isinstance(max_depth, int):
        raise TypeError('max_depth must be an instance of int')
    if not isinstance(trial_num, int):
        raise TypeError('trial_num must be an instance of int')

    clf = tree.DecisionTreeClassifier(max_depth = max_depth)

    dataset = pd.read_csv('../src/combined_features.csv', index_col = 0, parse_dates = True)
    print("dataset:")
    print(dataset)
    print("--------")

    labels = dataset["label"]
    codes, uniques = pd.factorize(labels)
    train_y = codes[train_id_slice]
    print("train_y:")
    print(train_y)
    print("--------")

    X = dataset.drop("label", axis = 1)
    train_X = X[train_id_slice]
    print("train_X:")
    print(train_X)
    print("--------")

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

    tree.plot_tree(clf)
    plt.savefig(r"../src/id3_only_tree" + str(trial_idx) + r".svg")
    plt.savefig(r"../src/id3_only_tree" + str(trial_idx) + r".png")

if __name__ == "__main__":
    for trial_idx in range(NUM_TRIALS):
        id3_only(TEST_IDS[trial_idx], TRAIN_IDS[trial_idx], MAX_DEPTH[trial_idx], trial_idx)