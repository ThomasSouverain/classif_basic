from typing import Tuple 

import numpy as np
import pandas as pd

def pandas_to_numpy(df:pd.DataFrame)->np.ndarray:
    """Get the numpy format of a previous pandas object.

    Args:
        df (pd.DataFrame): pandas object

    Returns:
        np.ndarray: numpy format of the previous pandas object
    """
    if isinstance(df, pd.Series): 
        df = df.to_numpy().reshape(-1,1)
    elif isinstance(df, pd.DataFrame):
        df = df.to_numpy()
    else:
        raise NotImplementedError("The df object you pass must be either of pd.Series or pd.DataFrame type.")
    return df 

class Node():
    """
    A class representing a node in a decision tree.

    Attributes:
        feature: The feature used for splitting at this node. Defaults to None.
        threshold: The threshold used for splitting at this node. Defaults to None.
        left: The left child node. Defaults to None.
        right: The right child node. Defaults to None.
        gain: The gain of the split. Defaults to None.
        value: If this node is a leaf node, this attribute represents the predicted value
            for the target variable. Defaults to None.

    Methods:
        __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
            Initializes the attributes of the Node object using the values provided for the arguments.
            If no values are provided for an attribute, it is set to None. If value is set to a non-None
            value, it means that this node is a leaf node and value represents the predicted value for
            the target variable at this leaf node.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None.
            gain: The gain of the split. Defaults to None.
            value: If this node is a leaf node, this attribute represents the predicted value
                for the target variable. Defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value

class DecisionTreeCausalOrder():
    """
    A decision tree classifier for binary classification problems, granting that the tree respects the causal order between features specified in the dictionary feature_order_dict. 

    Ex: with the features ["age", "education", "job"], the dict can be {0:1, 1:2, 2:3} meaning the ancestorship of age->education->job
        This ancestorship will be respected by the tree, i.e. an ancestor can never be used after a child for a split.
        It translates the basic intuition of human reasoning that your age can change your job, but your job will never make you become younger or older 
        - at least biologically...

    Attributes:
        min_samples (int), by default 2: Minimum number of samples required to split an internal node.
        max_depth (int), by default 2: Maximum depth of the decision tree.   

    Methods:
        split_data(self, dataset:pd.DataFrame, feature_name:str, threshold:float)->Tuple[pd.DataFrame, pd.DataFrame]:
                Splits the given dataset into two datasets based on the given feature and threshold.
        entropy(self, y:np.ndarray)->float:
                Computes the entropy of the given label values.
        information_gain(self, parent:pd.DataFrame, left:pd.DataFrame, right:pd.DataFrame)->float:
                Computes the information gain from splitting the parent dataset into two datasets. 
        best_split(self, dataset:pd.DataFrame, list_features:list)->dict:
                Finds the best split for the given dataset, among the features specified in list_features. 
        calculate_leaf_value(self, y:list)->float:
                Calculates the most occurring value in the given list of y values.  
        build_tree(self, dataset:pd.DataFrame, feature_order_dict:dict, current_depth:int=0)->Node():
                Recursively builds a decision tree from the given dataset.  
        fit(self, X:pd.DataFrame, y:pd.DataFrame, feature_order_dict:dict):
                Builds and fits the decision tree to the given X and y values, respecting the causal order of features. 
        predict(self, X:pd.DataFrame)->np.ndarray:
                Predicts the class labels for each instance in the feature matrix X.  
        make_prediction(self, x:np.ndarray, list_columns:list, node:Node(), individual_interpretability:bool=False)->np.ndarray:
                Traverses the decision tree to predict the target value for the given feature vector.       

    """

    def __init__(self, min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.

        Parameters:
            min_samples (int), by default 2: Minimum number of samples required to split an internal node.
            max_depth (int), by default 2: Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_data(self, dataset:pd.DataFrame, feature_name:str, threshold:float)->Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the given dataset into two datasets based on the given feature and threshold.

        Parameters:
            dataset (pd.DataFrame): Input dataset.
            feature_name (str): Name of the feature to be split on.
            threshold (float): Threshold value to split the feature on.

        Returns: Tuple[pd.DataFrame, pd.DataFrame]
            left_dataset (pd.DataFrame): Subset of the dataset with values less than or equal to the threshold.
            right_dataset (pd.DataFrame): Subset of the dataset with values greater than the threshold.
        """        
        # On each row in the dataset, split based on the given feature and threshold
        left_dataset = dataset.loc[dataset[feature_name]<=threshold]
        right_dataset =dataset.loc[dataset[feature_name]>threshold]

        return left_dataset, right_dataset

    def entropy(self, y:np.ndarray)->float:
        """
        Computes the entropy of the given label values.

        Parameters:
            y (np.ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """
        entropy = 0

        # Find the unique label values in y and loop over each value
        labels = np.unique(y)
        for label in labels:
            # Find the examples in y that have the current label
            label_examples = y[y == label]
            # Calculate the ratio of the current label in y
            pl = len(label_examples) / len(y)
            # Calculate the entropy using the current label and ratio
            entropy += -pl * np.log2(pl)

        # Return the final entropy value
        return entropy

    def information_gain(self, parent:pd.DataFrame, left:pd.DataFrame, right:pd.DataFrame)->float:
        """
        Computes the information gain from splitting the parent dataset into two datasets.

        Parameters:
            parent (pd.dataFrame): Input parent dataset.
            left (pd.DataFrame): Subset of the parent dataset after split on a feature.
            right (pd.DataFrame): Subset of the parent dataset after split on a feature.

        Returns:
            information_gain (float): Information gain of the split.
        """
        for df in [parent, left, right]:
            df = pandas_to_numpy(df)
        # set initial information gain to 0
        information_gain = 0
        # compute entropy for parent
        parent_entropy = self.entropy(parent)
        # calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        # compute entropy for left and right nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # calculate weighted entropy 
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        # calculate information gain 
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset:pd.DataFrame, list_features:list)->dict:
        """
        Finds the best split for the given dataset, among the features specified in list_features. 

        This list enables feature selection, which excludes the parents of the last splitting feature.
            Ex: if 'job' has been used for a split, the next splits will exclude 'age' of the features candidate for a split.
            Else, it would have been causal non-sense...

        Args:
        dataset (pd.DataFrame): The dataset to split. 
            (!) Must contain as columns list_features and the 'target' to be predicted. 
        list_features (list): List of the features in the dataset.

        Returns:
        dict: A dictionary with the best split feature name, threshold, gain, 
              left and right datasets.
        """
        # dictionary to store the best split values
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        # loop over all the CHILD features of the last splitting feature 
        # among the features, select only those who are causal childs of the last features used for splits
        # first, check if all the features in list_features and the 'target' are in the dataset
        for column in list_features+['target']:
            if column not in dataset.columns:
                raise NotImplementedError(f"The column {column} is not in the dataset. \n"
                    "The columns of the dataset must contain all features in list_features and the 'target'")

        for feature_name in list_features:
            #get the feature at the current feature_index
            feature_values = dataset[feature_name]
            #get unique values of that feature
            thresholds = np.unique(feature_values)
            # loop over all values of the feature
            for threshold in thresholds:
                # get left and right datasets
                left_dataset, right_dataset = self.split_data(dataset=dataset, feature_name=feature_name, threshold=threshold)
                # check if either datasets is empty
                if len(left_dataset) and len(right_dataset):
                    # get y values of the parent and left, right nodes
                    parent_y, left_y, right_y = dataset['target'], left_dataset['target'], right_dataset['target']
                    # compute information gain based on the y values
                    information_gain = self.information_gain(parent=parent_y, left=left_y, right=right_y)
                    # update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_name
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split

    
    def calculate_leaf_value(self, y:list)->float:
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            float: The most occurring value in the list.
        """
        y = list(y)
        #get the highest present class in the array
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset:pd.DataFrame, feature_order_dict:dict, current_depth:int=0)->Node():
        """
        Recursively builds a decision tree from the given dataset.

        The causal hierarchy is specified here by a dictionary feature_order_dict, linking each feature with its causal ancestorship (1=1st parents...).

        Args:
        dataset (pd.DataFrame): The dataset to build the tree from. 
            (!) Must contain as columns list_features and the 'target' to be predicted.         
        feature_order_dict (dict): dict with the order in which the features appear in the causal hierarchy (directed acyclic graph) selected by the user
            Ex: if X.columns == ["age", "education", "job"], the dict can be {0:1, 1:2, 2:3} meaning the ancestorship of age->education->job
            (!) Must contain all the features' indexes as keys
        current_depth (int) : The current depth of the tree, by default 0 to initialise the structure of the tree

        Returns:
        Node: The root node of the built decision tree.
        """
        # split the dataset into X, y values
        X, y = dataset.loc[:, dataset.columns != 'target'], dataset['target']
        n_samples, n_features = X.shape
        # initialise the selection of dataset features
        features_for_next_split_list = list(X.columns)
        # keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth: # while instead of if, in the recursive loop? TODO test
            # Get the best split
            # TODO here select the new list of features for possible split...
            best_split = self.best_split(dataset=dataset, list_features=features_for_next_split_list)
            # and actualises the dataset with only the causal childs of the splitting feature 
            print(f"\n {best_split['feature']} <= {best_split['gain']}")

            rank_of_split_feature = feature_order_dict[best_split['feature']]

            # the next possible splits will only consider the causal ancestors of the current splitting feature
            features_for_next_split_dict = dict((k, v) for k, v in feature_order_dict.items() if v >= rank_of_split_feature)
            features_for_next_split_list = list(features_for_next_split_dict.keys())

            # splitted data must be CHILD data as well, i.e. must contain no causal ancestor in the features!!
            best_split["left_dataset"] = best_split["left_dataset"][features_for_next_split_list]
            best_split["right_dataset"] = best_split["right_dataset"][features_for_next_split_list]

            # Check if gain isn't zero
            if best_split["gain"]:
                # continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], features_for_next_split_dict, current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], features_for_next_split_dict, current_depth + 1)
                # return decision node
                return Node(feature=best_split["feature"], 
                            threshold=best_split["threshold"],
                            left=left_node, 
                            right=right_node, 
                            gain=best_split["gain"])

        # compute leaf node value
        y = pandas_to_numpy(y)
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node value
        return Node(value=leaf_value)
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame, feature_order_dict:dict):
        """
        Builds and fits the decision tree to the given X and y values, respecting the causal order of features. 

        The causal hierarchy is specified here by a dictionary feature_order_dict, linking each feature with its causal ancestorship (1=1st parents...).

        Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.DataFrame): The target values.
        feature_order_dict (dict): dict with the order in which the features appear in the causal hierarchy (directed acyclic graph) selected by the user
            Ex: if X.columns == ["age", "education", "job"], the dict can be {0:1, 1:2, 2:3} meaning the ancestorship of age->education->job
            (!) Must contain all the features' indexes as keys
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            dataset = X.copy()
            dataset['target'] = y
            #dataset = dataset.to_numpy()
        # elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        #     dataset = np.concatenate((X, y), axis=1)  
        else:
            raise NotImplementedError("X and y must be of pandas types")

        self.root = self.build_tree(dataset=dataset, feature_order_dict=feature_order_dict)

    def predict(self, X:pd.DataFrame)->np.ndarray:
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (pd.DataFrame): The feature matrix to make predictions for.

        Returns:
            np.ndarray: The predicted class labels.
        """
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        list_columns = list(X.columns)
        X = pandas_to_numpy(X)
        for x in X:
            prediction = self.make_prediction(x=x, list_columns=list_columns, node=self.root)
            # Append the prediction to the list of predictions
            predictions.append(prediction)
        # Convert the list to a numpy array and return it
        predictions = np.array(predictions)
        return predictions
    
    def make_prediction(self, x:np.ndarray, list_columns:list, node:Node(), individual_interpretability:bool=False)->np.ndarray:
        """
        Traverses the decision tree to predict the target value for the given feature vector.

        Args:
        x (np.ndarray): The feature vector to predict the target value for.
        list_columns (list): The list with the names of features of X in order 
        node (Node): The current node being evaluated.
        individual_interpretability (bool), by default False: If True, prints the decision path inside the tree for the individual x. 

        Returns:
            np.ndarray: The predicted target value for the given feature vector.

        Example: for individual interpretability
            indiv_0 = X_test.values[0]
            model.make_prediction(indiv_0, list_columns=list(X_test.columns), node=model.root, individual_interpretability=True)
            '''Out:
                relationship == 0.140006254435931 > 0.0488861091395863
                relationship == 0.140006254435931 <= 0.9511138908604136
                job == 1.0 > 0.5038495393064493
            '''
        """
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse through the tree accordingly
            feature_name = node.feature 
            # get index of the feature, to select the score of the tree for the individual's feature 
            feature_index = list_columns.index(feature_name)
            feature_index = x[feature_index]
            if feature_index <= node.threshold:
                #print_individual_split(feature_name, feature_index, node.threshold, individual_interpretability, is_inf=True)
                # print(f"{feature_name} == {feature_index} <= {node.threshold}") # TODO activate for individual interpretability 
                return self.make_prediction(x=x, list_columns=list_columns, node=node.left)
            else:
                #print_individual_split(feature_name, feature_index, node.threshold, individual_interpretability, is_inf=False)
                # print(f"{feature_name} == {feature_index} > {node.threshold} \n") # TODO activate for individual interpretability 
                return self.make_prediction(x=x, list_columns=list_columns, node=node.right)

# def print_individual_split(feature_name:str, feature_index:float, threshold:float, individual_interpretability:bool, is_inf:bool)->str:
#     if individual_interpretability==True: # TODO get the function further than the first split: Bool => only used one time? 
#         if is_inf==True:
#             print(f"{feature_name} == {feature_index} <= {threshold}")
#         else:
#             print(f"{feature_name} == {feature_index} > {threshold}")


def perf_measure(y_true:np.ndarray, y_pred:np.ndarray)->Tuple[float]:
    """For binary classification, from a model's predictions (y_pred) and the actual class of each individual (y_true)
    Computes in absolute numbers true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN)

    Args:
        y_true (np.ndarray): the predictions of the model
            Must be set to a value in {0,1}
        y_pred (np.ndarray): the true labels of individuals
            Must be set to a value in {0,1}

    Returns:
        Tuple[float]: in absolute numbers true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN)
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
            FP += 1
        if y_true[i]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
            FN += 1

    return(TP, FP, TN, FN)

def prediction_ratios(y_true:np.ndarray, y_pred:np.ndarray)->Tuple[float]:
    """For binary classification, from a model's predictions (y_pred) and the actual class of each individual (y_true)
    Computes in percentages accuracy (ACC), false positives rate (FPR), and true positives rate (TPR)

    Args:
        y_true (np.ndarray): the predictions of the model
            Must be set to a value in {0,1}
        y_pred (np.ndarray): the true labels of individuals
            Must be set to a value in {0,1}

    Returns:
        Tuple[float]:  in percentages accuracy (ACC), false positives rate (FPR), and true positives rate (TPR)
    """

    TP, FP, TN, FN = perf_measure(y_true, y_pred)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    print(f"Accuracy : {ACC}")
    print(f"False positive rate: {FPR}")
    print(f"True positive rate: {TPR}")
    
    return ACC, FPR, TPR
