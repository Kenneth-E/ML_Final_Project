import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class TreeNode(object): # Do not modify this class
    '''
    A node class for a decision tree.
    '''
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature # feature to split on
        self.value = value # value used for splitting
        self.left = left # left child
        self.right = right # right child
        self.label = label # label for the node

class TreeBase(object): # Do not modify this class
    def __init__(self, data, label, max_depth=5):
        '''
        Constructor
        Parameters:
            data: DataFrame, the data for the tree
            label: str, the label of the target
            max_depth: int, the maximum depth of the tree
        '''
        self.data = data
        self.root = None
        self.max_depth = max_depth
        self.label = label
        self.features = data.columns.drop(label)

    def select_best_feature(self, data, features):
        '''
        Select the feature with the highest information gain
        Parameters:
            data: DataFrame
            features: list of features
        Returns:
            best_feature: str
            best_value: str
        '''
        best_gain = 0

        for feature in features: #Iterate over all features
            values = data[feature].unique() #Get all unique values of the feature
            for value in values: #Iterate over all values
                gain = self.information_gain(data, feature, value) #Calculate the information gain
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def entropy(self, data):
        raise NotImplementedError

    def information_gain(self, data, feature, value):
        raise NotImplementedError

    def predict(self, data_point):
        '''
        Predict the label for a single instance
        Parameters:
            data_point: pandas Series, with keys being the feature names
        Returns:
            label: the label of the instance
        '''
        node = self.root
        label = None
        while True:
            if node.label is not None: #Leaf node
                label = node.label
                break
            if type(node.value) == str: #Categorical feature
                go_left = data_point[node.feature] == node.value
            else: #Numerical feature
                go_left = data_point[node.feature] < node.value

            if go_left:
                node = node.left
            else:
                node = node.right
        return label

    def fit(self):
        self.root = self.build_tree(TreeNode(), self.data, self.features, 0) #Build the tree

    def build_tree(self, node, data, features, depth):
        '''
        Recursively build the decision tree
        Parameters:
            node: TreeNode, current node to be split
            data: DataFrame, the data for the tree
            features: list of features
            depth: int, the current depth of the tree
        Returns:
            node: TreeNode, the root of the built tree
        '''

        #Stop if the entropy is 0, or the depth >= max_depth, or all data points has the same feature value.
        stop = np.isclose(self.entropy(data), 0) or \
                 depth >= self.max_depth or \
                 (data.values[0] == data.values).all()
        if stop:
            node = TreeNode()
            node.label = data[self.label].mode()[0] #Get the most common label, only set label for leaf nodes
            return node

        feature, value = self.select_best_feature(data, features)
        node = TreeNode(feature=feature, value=value)
        if type(value) == str: #Categorical feature
            left_data, right_data = data[data[feature] == value], data[data[feature] != value]
        else: #Numerical feature
            left_data, right_data = data[data[feature] < value], data[data[feature] >= value]

        node.left = self.build_tree(node.left, left_data, features, depth + 1)
        node.right = self.build_tree(node.right, right_data, features, depth + 1)
        return node

    def print_tree(self):
        '''
        Print the tree
        Parameters:
            node: TreeNode, the current node
            depth: int, the current depth of the tree
        '''
        self._print_tree(self.root, 0)

    def _print_tree(self, node, depth):
        '''
        Recursively print the tree
        Parameters:
            node: TreeNode, the current node
            depth: int, the current depth of the tree
        '''
        if node is None:
            return
        if node.label is not None:
            print(" " * depth, node.label)
        else:
            if type(node.value) == str:
                print(" " * depth, node.feature, "=", node.value)
            else:
                print(" " * depth, node.feature, "<", node.value)
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)

class DecisionTree(TreeBase):
    '''
    Binary decision tree class, inherits from TreeBase
    '''
    def __init__(self, data, label, max_depth=5):
        super(DecisionTree, self).__init__(data, label, max_depth)

    def entropy(self, data):
        '''
        Calculate the entropy of the data
        Parameters:
            data: DataFrame
        Returns:
            the entropy of the data
        '''
        label_data = data[self.label]
        num_total = np.shape(label_data)[0]
        outcome_counts = label_data.value_counts()
        entropy = 0
        for outcome, value in outcome_counts.items():
            proportion = value / num_total
            entropy -= proportion * np.log2(proportion)
        # if the proportion is 0, the limit of n log n is 0, so no need to change the entropy
        # if the num_total is 0, proportion_yes_play and proportion_no_play are both 0 and entropy is 0
        return entropy
    def information_gain(self, data, feature, value):
        '''
        Calculate the information gain
        Parameters:
            data: DataFrame
            feature: the feature to split
            vavlue: the value of the feature
        Returns:
            the information gain

        '''
        if len(data.xs(feature, axis=1)) != 0 and type(data.xs(feature, axis=1).iloc[0]) == str: # categorical variable
            conditional_data = data[data[feature] == value]
            false_conditional_data = data[data[feature] != value]
        else: # numerical variable
            conditional_data = data[data[feature] < value]
            false_conditional_data = data[data[feature] >= value]
        num_conditional_rows = np.shape(conditional_data)[0]
        num_false_conditional_rows = np.shape(false_conditional_data)[0]
        num_total_rows = np.shape(data)[0]
        information_gain_conditional_component = - num_conditional_rows / num_total_rows * self.entropy(conditional_data)
        information_gain_false_conditional_component = - num_false_conditional_rows / num_total_rows * self.entropy(false_conditional_data)
        information_gain = self.entropy(data) + information_gain_conditional_component + information_gain_false_conditional_component
        return information_gain

def evaluate(dt: TreeBase, dataset: pd.DataFrame):
    '''
    Evaluate the decision tree
    Parameters:
        dt: DecisionTree
        dataset: DataFrame
    Returns:
        accuracy: the accuracy of the model on the dataset
    '''
    num_success = 0
    num_total = 0
    for index, row in dataset.iterrows():
        prediction = dt.predict(row)
        if prediction == row[dt.label]:
            num_success += 1
        num_total += 1
    return num_success / num_total


if __name__ == "__main__":
    data = [['Sunny', 'Hot', 'High', 'Weak', 'No'],
           ['Sunny', 'Hot', 'High', 'Strong', 'No'],
           ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
           ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
           ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
           ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
           ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
           ['Sunny', 'Mild', 'High', 'Weak', 'No'],
           ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
           ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
           ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
           ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
           ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
           ['Rain', 'Mild', 'High', 'Strong', 'No']]
    colums = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']
    dataset = pd.DataFrame(data, columns = colums)
    dataset

    dt = DecisionTree(dataset, 'Play', 5)
    data_test1 = dataset.iloc[:6]
    assert np.isclose(dt.entropy(data_test1), 1.0) # Test entropy implementation
    dt.fit() # Test build the tree and make sure there is no error

    dt.print_tree() # Print the tree


    sample = dataset.iloc[0]
    print(sample)
    print('----------')
    print('prediction:', dt.predict(sample)) # Test predict function

    # test evaluate function
    print('accuracy:', evaluate(dt, dataset)) # Test evaluate function

    # Don't modify this cell, we will use this code to generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    # Create a pandas DataFrame for the train/test data
    columns = ['feature_' + str(i) for i in range(X_train.shape[1])] + ['label']
    df_train = pd.DataFrame(np.column_stack((X_train, y_train)), columns=columns)
    df_test = pd.DataFrame(np.column_stack((X_test, y_test)), columns=columns)
    df_train.head()

    # test fitting the decision tree
    dt = DecisionTree(df_train, 'label', 0)
    dt.fit()

    # test evaluate function
    train_acc_value, test_acc_value = evaluate(dt, df_train), evaluate(dt, df_test)
    print('Train accuracy:', train_acc_value)
    print('Test accuracy:', test_acc_value)

    max_depth = 10 #you can vary the depth, the larger the depth, the longer it will take to train. This is just our recommended number
    train_acc, test_acc = [], [] #store the training/test accuracy for each tree depth
    for tree_depth in range(max_depth + 1):
        dt = DecisionTree(df_train, 'label', tree_depth)
        dt.fit()
        train_acc_value, test_acc_value = evaluate(dt, df_train), evaluate(dt, df_test)
        train_acc.append(train_acc_value)
        test_acc.append(test_acc_value)
        # Fit the decision tree for each tree depth and evaluate the accuracy of on the training and test data
        pass

    # write the code to plot the train and test accuracy for each tree depth

    # plot
    fig, ax = plt.subplots()
    x = range(max_depth + 1)
    ax.plot(x, train_acc, label='Train Accuracy')
    ax.plot(x, test_acc, label='Test Accuracy')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train and Test Accuracy vs Depth')
    ax.legend()
    plt.show()