import os
import pickle

import numpy as np
import pandas as pd
import shap

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
