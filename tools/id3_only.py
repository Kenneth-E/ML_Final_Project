import numpy as np
import pandas as pd

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