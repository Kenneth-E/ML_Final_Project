import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

TEST_IDS = [slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000), slice(0, 1_000)]
TRAIN_IDS = [slice(1_000, 100_000), slice(1_000, 100_000), slice(1_000, 100_000), slice(1_000, 100_000), slice(1_000, 100_000), slice(1_000, 100_000)]
MAX_DEPTH = [5, 10, 20, 40, None, 2] # None = unlimited depth
USE_ADABOOST = [False, False, False, False, False, True]
ADABOOST_NUM_ESTIMATORS = [None, None, None, None, None, 100]
ADABOOST_LEARNING_RATE = [None, None, None, None, None, 1.0]
NUM_TRIALS = 6

# TODO: PCA, wait & see output, validate that tree outputs are correct (not off-by-one), cross-validation
# TODO: Bible: what happens if all translations for the same sentence are in training? what if one in training, one in testing?
# TODO: is it important for each language to have about the same amount of data?
# TODO: agnostic to captialization? German nouns are capitalized, so maybe not?
#  https://scikit-learn.org/stable/modules/decomposition.html#pca
#  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html#sklearn.decomposition.IncrementalPCA
#  https://scikit-learn.org/stable/modules/cross_validation.html#group-cv
def ml_model(trial_num, test_id_slice, train_id_slice, max_depth, use_adaboost, adaboost_num_estimators, adaboost_learning_rate):
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

    dataset = pd.read_csv('../data/combined_features/combined_features.csv', index_col = 0, parse_dates = True)
    print("dataset:")
    print(dataset)
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
        plt.savefig(r"../src/tree_raw_" + str(trial_idx) + r".svg")
        plt.savefig(r"../src/tree_raw_" + str(trial_idx) + r".png")

        tree.plot_tree(clf, class_names=uniques, feature_names=columns)
        plt.savefig(r"../src/tree_" + str(trial_idx) + r".svg")
        plt.savefig(r"../src/tree_" + str(trial_idx) + r".png")

if __name__ == "__main__":
    for trial_idx in range(NUM_TRIALS):
        ml_model(trial_idx, TEST_IDS[trial_idx], TRAIN_IDS[trial_idx], MAX_DEPTH[trial_idx], USE_ADABOOST[trial_idx], ADABOOST_NUM_ESTIMATORS[trial_idx], ADABOOST_LEARNING_RATE[trial_idx])