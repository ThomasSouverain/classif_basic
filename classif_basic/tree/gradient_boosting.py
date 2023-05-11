import numpy as np
import pandas as pd

from classif_basic.tree.decision_tree import DecisionTreeCausalOrder

class GradientBoostingCausalOrder():
    """A class to build gradient boosting trees, by granting that each tree respects the causal order between features specified in the dictionary feature_order_dict. 

    Ex: with the features ["age", "education", "job"], the dict can be {0:1, 1:2, 2:3} meaning the ancestorship of age->education->job
        This ancestorship will be respected by each tree, i.e. an ancestor can never be used after a child for a split.
        It translates the basic intuition of human reasoning that your age can change your job, but your job will never make you become younger or older 
        - at least biologically...
    
    Attributes:
        n_trees (int): The number of trees which will successively improve, based on past trees information (gradient)
        learning_rate: The learning rate to weight the predictions of each new tree 
        min_samples (int), by default 2: Minimum number of samples required to split an internal node.
        max_depth (int);, by default 1: Maximum depth of the decision tree.
    
    Methods:
        __init__(self, n_trees:int, learning_rate:float, min_samples:int=2, max_depth:int=1): Constructor for GradientBoostingCausalOrder class

        fit(self, X:pd.DataFrame, y:pd.DataFrame, feature_order_dict:dict): Builds and fits the succession of trees to the given X and y values.

        predict(self, X:pd.DataFrame)->np.ndarray: Predicts the class labels for each instance in the feature matrix X.

    """
    
    def __init__(self, n_trees:int, learning_rate:float, min_samples:int=2, max_depth:int=1):
        """
        Constructor for GradientBoostingCausalOrder class.

        Parameters:
            n_trees (int): The number of trees which will successively improve, based on past trees information (gradient)
            learning_rate: The learning rate to weight the predictions of each new tree 
            min_samples (int), by default 2: Minimum number of samples required to split an internal node.
            max_depth (int);, by default 1: Maximum depth of the decision tree.
        """
        self.n_trees=n_trees
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.min_samples=min_samples 
        
    def fit(self, X:pd.DataFrame, y:pd.DataFrame, feature_order_dict:dict):
        """
        Builds and fits the succession of trees to the given X and y values.

        Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.DataFrame): The target values.
        feature_order_dict (dict): dict with the order in which the features appear in the causal hierarchy (directed acyclic graph) selected by the user
            Ex: if X.columns == ["age", "education", "job"], the dict can be {0:1, 1:2, 2:3} meaning the ancestorship of age->education->job
            (!) Must contain all the features' indexes as keys
        """
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        self.trees = []
        self.F0 = y.mean()
        Fm = self.F0 
        for _ in range(self.n_trees):            
            tree = DecisionTreeCausalOrder(min_samples=self.min_samples, max_depth=self.max_depth)
            # convert the gradient signal of the past tree to a pandas compatible format
            y_with_gradient = y - Fm
            y_with_gradient = pd.Series(y_with_gradient)
            tree.fit(X, y_with_gradient, feature_order_dict)
            Fm = Fm.squeeze() + self.learning_rate * tree.predict(X).squeeze()
            Fm = pd.Series(Fm)
            self.trees.append(tree)
            
    def predict(self, X:pd.DataFrame)->np.ndarray:
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (pd.DataFrame): The feature matrix to make predictions for.

        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.F0 + self.learning_rate * np.sum([tree.predict(X) for tree in self.trees], axis=0)
