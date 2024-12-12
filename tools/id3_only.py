import numpy as np
import pandas as pd

from lib.JHU_CS475.homework1.id3 import TreeBase, TreeNode, DecisionTree, evaluate

if __name__ == "__main__":
    dataset = pd.read_csv('../src/combined_features.csv', index_col = 0, parse_dates = True)
    print(dataset)

    dt = DecisionTree(dataset, 'label', 5)
    data_test1 = dataset.iloc[:6]
    assert np.isclose(dt.entropy(data_test1), 1.0)  # Test entropy implementation
    dt.fit()  # Test build the tree and make sure there is no error

    print('----------')
    dt.print_tree()  # Print the tree
    print('----------')

    sample = dataset.iloc[:]
    print(sample)
    print('----------')
    print('prediction:', dt.predict(sample))  # Test predict function
'''
from lib.DecisionTreeID3.id3 import DecisionTreeID3

# NOTE: this gives anomalous results, to be updated

if __name__ == "__main__":
    df = pd.read_csv('../src/combined_features.csv', index_col = 0, parse_dates = True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTreeID3(max_depth = 3, min_samples_split = 2)
    tree.fit(X, y)
    predictions = tree.predict(X)
    print(f"Predictions:     {predictions}")
    print(f"Actual Labels:   {list(y)}")
    print(f"Input: \n{X}")
'''