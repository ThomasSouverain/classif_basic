import json
import os
import pickle
import time
from io import BytesIO
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import shap
from xgboost.core import Booster 
from xgboost.sklearn import XGBModel

# for XGB trees' visualisation

# from ._typing import PathLike
# from .core import Booster
# from .sklearn import XGBModel

Axes = Any  # real type is matplotlib.axes.Axes
GraphvizSource = Any  # real type is graphviz.Source

PathLike = Union[str, os.PathLike]

def features_importances_from_pickle(
    augmented_train_valid_set: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    model_task: str,
    uncorrected_model_path: str = None,
) -> list:
    """Loads the uncorrected_model and returns a list of the features whose SHAP influence on the output is > 1%.

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which columns are divided into groups of individuals *
        and inspected to detect gaps of fair_scores.
    X_train_valid : pd.DataFrame
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")

    Returns
    -------
    list
    """
    if uncorrected_model_path is None:
        uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    with open(uncorrected_model_path, "rb") as inp:
        uncorrected_model = pickle.load(inp)

    explainer = shap.TreeExplainer(uncorrected_model)
    shap_values = np.array(explainer.shap_values(X_train_valid))

    # in case of multiclass, return for all individual only the SHAP values corresponding to one's predicted class
    if model_task == "multiclass":
        multi_Y_pred_train_valid_uncorrected = augmented_train_valid_set[
            "multi_predicted_uncorrected"
        ].to_numpy()

        # for multiclass, reduce shape of shap_values_test: from shape (nb_classes, nb_individuals, nb_features) -> (nb_individuals, nb_features)
        shap_values_test = np.transpose(shap_values, (1, 0, 2))
        # to take the shap values which correspond to argmax class (i.e. predicted class), expand the Y_pred labels (multi_Y_pred_train_valid_uncorrected)
        # => same shape than shap_values_test
        pred_indices = np.expand_dims(multi_Y_pred_train_valid_uncorrected, axis=(1, 2))

        # then: for all individual, select the shap_values corresponding to the indice of the predicted label (e.g. 1 housing night)
        shap_values_by_argmax = np.take_along_axis(shap_values_test, pred_indices, axis=1)
        # finally, drop the now useless array of (nb_classes): from shape (predicted_class=1, nb_individuals, nb_features) -> (nb_individuals, nb_features)
        shap_values = np.squeeze(shap_values_by_argmax, axis=1)

    # compute the mean of shap values on predicted output
    # from shap_values of shape (nb_individuals, nb_features) -> shap_sum of shape (nb_features,)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_train_valid.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]
    importance_df = importance_df.sort_values("shap_importance", ascending=False)

    # select only features on which the uncorrected_model's output is influenced > 1% => on which the uncorrected_model really has a discriminant impact
    important_features_df = importance_df.loc[importance_df["shap_importance"] > 0.01]
    important_features_list = important_features_df["column_name"]

    # for the user, plot of the features whose importance > 1%
    X_train_valid_with_important_features = X_train_valid.loc[:, important_features_list]

    print(
        "\n Features whose influence is > 1%\n ''' Fairness analysis on these important features on which predictions are based ''' \n"
    )
    shap.summary_plot(
        shap_values, X_train_valid, max_display=X_train_valid_with_important_features.shape[1]
    )

    return important_features_list

def select_important_features(
    augmented_train_valid_set: pd.DataFrame,
    model_name: str,
    model_task: str,
    uncorrected_model_path: str = None,
) -> list:
    """Returns a list with the important features for the model's prediction (i.e. SHAP influence on output > 1%).

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which columns are divided into groups of individuals *
        and inspected to detect gaps of fair_scores.
    model_name : str
        Name of the model whose results are inspected.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")

    Returns
    -------
    list
    """
    features = list(
        set(augmented_train_valid_set.columns)
        - set(
            [
                "target_train_valid",
                f"proba_{model_name}",
                f"predicted_{model_name}",
                f"true_positive_{model_name}",
                f"false_positive_{model_name}",
                f"true_negative_{model_name}",
                f"false_negative_{model_name}",
            ]
        )
    )

    # eliminate columns not to be inspected (because not features) for multiclass
    if model_task == "multiclass":
        features = list(
            set(features)
            - set(
                [
                    "multi_target_train_valid",
                    f"multi_proba_{model_name}",
                    f"multi_predicted_{model_name}",
                ]
            )
        )

    # select list of features on which to raise discrimination alerts, based on features importances
    X_train_valid = augmented_train_valid_set.loc[:, features]

    # TODO make explicit the uncorrected_model_path
    uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    important_features_list = features_importances_from_pickle(
        augmented_train_valid_set, X_train_valid, model_task, uncorrected_model_path
    )

    return important_features_list
 
def augment_train_valid_set_with_results(
    model_name: str,
    previous_train_valid_set: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_pred_train_valid: np.ndarray,
    model_task: str,
    multi_Y_train_valid: np.ndarray = None,
    multi_predict_proba_train_valid: np.ndarray = None,
) -> pd.DataFrame:
    """Add predictions and statistics to a model to the dataset train&valid,
    to enable further comparison and selection between the initial and fair train models.

    Parameters
    ----------
    model_name : str
        Name of the model whose results will be integrated.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    previous_train_valid_set : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)
    Y_train_valid : pd.DataFrame
        Target, ie true labels of Y on train_valid set.
    Y_pred_train_valid : np.ndarray
        Vector of classes (0,1) returned by the model given an optimised threshold (shape == Y_train_valid.shape)
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification"}
    multi_Y_train_valid: np.ndarray, by default None
        Only when model_task == "multiclass", to inspect stat performances of the model on different classes
        shape(Y_train_valid,): true labels, to further compare with the models' predicted labels
    multi_predict_proba_train_valid: np.ndarray, by default None
        Only when model_task == "multiclass", to inspect stat performances of the model on different classes
        shape(Y_train_valid, nb_labels)

    Returns
    -------
    pd.DataFrame
        Previous train_valid set augmented with probabilities, predicted labels, (true/false) (positive/negative)
        for further computation of fair_score of the model.
    """
    if model_task in {"classification", "multiclass"}:

        previous_train_valid_set["target_train_valid"] = Y_train_valid
        previous_train_valid_set[f"predicted_{model_name}"] = Y_pred_train_valid

        previous_train_valid_set[f"true_positive_{model_name}"] = np.where(
            (
                previous_train_valid_set[f"predicted_{model_name}"]
                == previous_train_valid_set["target_train_valid"]
            )
            & (previous_train_valid_set[f"predicted_{model_name}"] == 1),
            1,
            0,
        )
        previous_train_valid_set[f"false_positive_{model_name}"] = np.abs(
            1 - previous_train_valid_set[f"true_positive_{model_name}"]
        )
        previous_train_valid_set[f"true_negative_{model_name}"] = np.where(
            (
                previous_train_valid_set[f"predicted_{model_name}"]
                == previous_train_valid_set["target_train_valid"]
            )
            & (previous_train_valid_set[f"predicted_{model_name}"] == 0),
            1,
            0,
        )
        previous_train_valid_set[f"false_negative_{model_name}"] = np.abs(
            1 - previous_train_valid_set[f"true_negative_{model_name}"]
        )

        if model_task == "multiclass":
            # add predict_probas by label, to further compute the model's stat perf
            # pack the initial vector of predict_probas with a vector column: for all individual (i.e. line), list of probabilities by label
            previous_train_valid_set[f"multi_proba_{model_name}"] = np.array(
                pd.DataFrame({0: multi_predict_proba_train_valid.tolist()})
            )
            # add multi_pred_train_valid to inspect stat performances on different classes
            multi_Y_pred_train_valid = multi_predict_proba_train_valid.argmax(axis=-1)

            previous_train_valid_set[f"multi_predicted_{model_name}"] = multi_Y_pred_train_valid
            previous_train_valid_set[f"multi_target_train_valid"] = multi_Y_train_valid

    elif model_task == "regression":

        previous_train_valid_set["target_train_valid"] = Y_train_valid
        previous_train_valid_set[f"predicted_{model_name}"] = Y_pred_train_valid

    return previous_train_valid_set


def to_graphviz(
    booster: Booster,
    fmap: PathLike = "",
    num_trees: int = 0,
    rankdir: Optional[str] = None,
    yes_color: Optional[str] = None,
    no_color: Optional[str] = None,
    condition_node_params: Optional[dict] = None,
    leaf_node_params: Optional[dict] = None,
    **kwargs: Any
) -> GraphvizSource:
    """Convert specified tree to graphviz instance. IPython can automatically plot
    the returned graphviz instance. Otherwise, you should call .render() method
    of the returned graphviz instance.
    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "UT"
        Passed to graphviz via graph_attr
    yes_color : str, default '#0000FF'
        Edge color when meets the node condition.
    no_color : str, default '#FF0000'
        Edge color when doesn't meet the node condition.
    condition_node_params : dict, optional
        Condition node configuration for for graphviz.  Example:
        .. code-block:: python
            {'shape': 'box',
             'style': 'filled,rounded',
             'fillcolor': '#78bceb'}
    leaf_node_params : dict, optional
        Leaf node configuration for graphviz. Example:
        .. code-block:: python
            {'shape': 'box',
             'style': 'filled',
             'fillcolor': '#e48038'}
    \\*\\*kwargs: dict, optional
        Other keywords passed to graphviz graph_attr, e.g. ``graph [ {key} = {value} ]``
    Returns
    -------
    graph: graphviz.Source
    """
    try:
        from graphviz import Source
    except ImportError as e:
        raise ImportError('You must install graphviz to plot tree') from e
    if isinstance(booster, XGBModel):
        booster = booster.get_booster()

    # squash everything back into kwargs again for compatibility
    parameters = 'dot'
    extra = {}
    for key, value in kwargs.items():
        extra[key] = value

    if rankdir is not None:
        kwargs['graph_attrs'] = {}
        kwargs['graph_attrs']['rankdir'] = rankdir
    for key, value in extra.items():
        if kwargs.get("graph_attrs", None) is not None:
            kwargs['graph_attrs'][key] = value
        else:
            kwargs['graph_attrs'] = {}
        del kwargs[key]

    if yes_color is not None or no_color is not None:
        kwargs['edge'] = {}
    if yes_color is not None:
        kwargs['edge']['yes_color'] = yes_color
    if no_color is not None:
        kwargs['edge']['no_color'] = no_color

    if condition_node_params is not None:
        kwargs['condition_node_params'] = condition_node_params
    if leaf_node_params is not None:
        kwargs['leaf_node_params'] = leaf_node_params

    if kwargs:
        parameters += ':'
        parameters += json.dumps(kwargs)
    tree = booster.get_dump(
        fmap=fmap,
        dump_format=parameters)[num_trees]
    g = Source(tree)
    return g

def plot_tree(
    booster: Booster,
    fmap: PathLike = "",
    num_trees: int = 0,
    rankdir: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any
) -> Axes:
    """Plot specified tree.
    Parameters
    ----------
    booster : Booster, XGBModel
        Booster or XGBModel instance
    fmap: str (optional)
       The name of feature map file
    num_trees : int, default 0
        Specify the ordinal number of target tree
    rankdir : str, default "TB"
        Passed to graphviz via graph_attr
    ax : matplotlib Axes, default None
        Target axes instance. If None, new figure and axes will be created.
    kwargs :
        Other keywords passed to to_graphviz
    Returns
    -------
    ax : matplotlib Axes
    """
    try:
        from matplotlib import pyplot as plt
        from matplotlib import image
    except ImportError as e:
        raise ImportError('You must install matplotlib to plot tree') from e

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(25,15))

    g = to_graphviz(booster, fmap=fmap, num_trees=num_trees, rankdir=rankdir,
                    **kwargs)

    s = BytesIO()
    s.write(g.pipe(format='png'))
    s.seek(0)
    img = image.imread(s)

    ax.imshow(img)
    ax.axis('off')
    return ax

# the following functions join the trees of XGBoost sharing the same first splitting feature
def extract_first_splitting_feature(booster:Booster, num_trees:int)->str:
    """Extract the feature of the first node of the XGB tree.

    Args:
        booster (Booster): trained and loaded XGB Model
        num_trees (int): index of the XGBoost tree

    Returns:
        str: name of the feature used as a first node, i.e. with the most importante 'discriminative' power for the tree 
    """
    
    list_description_trees = booster.get_booster().get_dump()

    first_split = list_description_trees[num_trees].split('[')[1]
    first_feature = first_split.split(']')[0]

    return first_feature

def get_df_first_splits(booster:Booster,get_max_split_feature:bool=False,nb_min_trees:int=None)->pd.DataFrame:
    """Joins the XGB trees sharing the same first splitting feature, through the feature name (key:str) and the indexes of the trees (value:list(int)).

    Args:
        booster (Booster): trained and loaded XGB Model
        get_max_split_feature (bool, optional): Returns only in the pd.DataFrame the feature mostly used in trees for the first split. Defaults to False.
        nb_min_trees (int, optional): Returns only in the pd.DataFrame the features used by 'nb_min_trees' trees or more. Defaults to None.

    Returns:
        pd.DataFrame: stores the feature of the first split, the indexes of corresponding trees, and their number,
            In the 3 columns "first_splitting_feature","trees_index", "nb_trees".     """

    # t_begin = time.time()

    # creates a dict to store the indexes of a tree associated with the same first (i.e. most important) splitting feature
    # s.t. name of the common splitting feature (key:str) and corresponding indexes of the trees (value:list(int))
    dict_trees = {}

    # booster is a XGBoost model fitted using the sklearn API
    list_description_trees = booster.get_booster().get_dump()
    trees_total_number = len(list_description_trees)

    for tree_nb in range(trees_total_number):

        first_splitting_feature = extract_first_splitting_feature(booster=booster, num_trees=tree_nb)

        dict_trees.setdefault(first_splitting_feature,[])
        dict_trees[first_splitting_feature].append(tree_nb)

    # t_end = time.time()
    # print(f"Getting {trees_total_number} trees of XGBoost with the same first splitting feature took {t_end - t_begin} seconds")

    # then, structure this information in a pd.DataFrame 
    df_first_splits = pd.DataFrame(dict_trees.items(),columns=["first_splitting_feature","trees_index"])
    df_first_splits["nb_trees"] = df_first_splits["trees_index"].str.len()
    # sort df by number of trees in which the feature is used as first splitting node 
    df_first_splits = df_first_splits.sort_values(by="nb_trees",ascending=False)
    df_first_splits = df_first_splits.set_index("first_splitting_feature")

    # select the pd.DataFrame in case options are set by the user
    if (nb_min_trees is not None) and (get_max_split_feature is True):
        return NotImplementedError("get_max_split_feature and nb_min_trees can not be set as options together. Either choose one, or none of the options.")
    elif get_max_split_feature is True:
        df_first_splits = df_first_splits.loc[df_first_splits["nb_trees"].idxmax()]
    elif nb_min_trees is not None:
        df_first_splits = df_first_splits.loc[df_first_splits["nb_trees"]>=nb_min_trees]
    else:
        pass

    return df_first_splits
