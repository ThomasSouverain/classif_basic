import os
import pickle
import random
from math import floor
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve

from matplotlib import pyplot

import xgboost

def train_naive_xgb(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
) -> np.ndarray:
    """Quickly train a XGB model to test classif_basic in a notebook. In real life, will be replaced by Brain predictions.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_valid : pd.DataFrame
    X_train_valid : pd.DataFrame
    X_test : pd.DataFrame
    Y_train : pd.DataFrame
    Y_valid : pd.DataFrame
    Y_train_valid : pd.DataFrame
    Y_test : pd.DataFrame
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    adjusted_sample_weight_train: np.ndarray, by default None
        Array with the weight of the error for each individual (row_number),
        i.e. during learning, how much the model is penalised when it goes away of the target for each individual
        Will be used during correction, to set higher weights to protected individuals
        shape : X_train.shape[0]
    save_model : bool, by default False
        If True, enables to save the model in path "/work/data/models/uncorrected_model.pkl" and computes a list with the best important features (i.e. importance > 1%).
        Used in detection phase, to select only relevant features to launch discrimination alerts (i.e. features on which the uncorrected_model is really discriminant)

    Returns
    -------
    np.ndarray of shape == Y_train_valid.shape:
    Y_pred_train_valid: Vector of classes (0,1) returned by the model given an optimised threshold
    """

    ## fixed parameters

    SEED = 7
    VALID_SIZE = 0.15

    early_stopping_rounds = 20
    verbose = 100

    ## then training of model as a XGB (quick, but already fixed parameters)
    print(
        f"Training model with {X_train.shape[1]} features, on {X_train.shape[0]} rows (valid {X_valid.shape[0]} rows, test {X_test.shape[0]} rows) "
    )

    if model_task == "classification":

        xgb_classif_params = {
            "seed": SEED,
            "objective": "binary:logistic",
            "n_estimators": 1000,
            "max_depth": 3,
            "importance_type": "gain",
            "use_label_encoder": False,
        }

        model = xgboost.XGBClassifier(**xgb_classif_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"auc", "aucpr"}:
            xgb_eval_metric = stat_criteria
        elif stat_criteria == "mix_auc_aucpr":
            xgb_eval_metric = "aucpr"  # TODO add custom eval metrics of mix_auc_aucpr for uncorrected model
        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}."
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    elif model_task == "multiclass":

        xgb_multiclass_params = {
            "seed": SEED,
            "objective": "multi:softptob",
            "n_estimators": 500,
            "max_depth": 3,
            "max_delta_step": 5,
            "learning_rate": 0.3,
        }

        model = xgboost.XGBClassifier(**xgb_multiclass_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"merror", "mlogloss", "auc", "f1_score"}:
            xgb_eval_metric = stat_criteria

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}"
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    elif model_task == "regression":

        xgb_reg_params = {
            "seed": SEED,
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": 3,
            "importance_type": "gain",
            "use_label_encoder": False,
        }

        model = xgboost.XGBRegressor(**xgb_reg_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"rmse", "mape"}:
            xgb_eval_metric = stat_criteria

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error"
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    model.fit(
        X_train,
        Y_train,
        sample_weight=adjusted_sample_weight_train,
        eval_metric=xgb_eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=verbose,
    )

    Y_pred_train_valid = prediction_train_valid_by_task(
        model, X_valid, X_train_valid, Y_valid, Y_train_valid, model_task
    )

    # in case of detection (for the first "uncorrected" model)
    # pre-select features where it is relevant to detect discriminations (i.e. important features for the model)
    # save the model on disk storage, to then: use it, and compute features influences
    if save_model == True:
        pickle_save_model(model)

    return Y_pred_train_valid

def pickle_save_model(uncorrected_model: xgboost, uncorrected_model_path: str = None):
    """Creates a directory "/work/data/models" and store the model in the model_path (here, "/work/data/models/uncorrected_model.pkl").
    Useful to re-use the uncorrected_model, and compute features importances.

    Parameters
    ----------
    uncorrected_model : xgboost
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")
    """

    # if the path does not already exist, creates the directory where the uncorrected_model is stored
    if uncorrected_model_path is None:
        uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    models_directory = "/".join(uncorrected_model_path.split("/")[:-1])

    Path(models_directory).mkdir(parents=True, exist_ok=True)

    with open(uncorrected_model_path, "wb") as outp:  # Overwrites any existing file
        pickle.dump(uncorrected_model, outp, pickle.HIGHEST_PROTOCOL)

def prediction_train_valid_by_task(
    model: xgboost,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    model_task: str,
) -> np.ndarray:
    """Returns in a np.ndarray the predicted labels (for classification) of values (for regression) of the model for X_train_valid.
    For multiclass: returns in a np.ndarray of shape(nb_individuals, nb_labels) the predicted probas by label, better to compute stat score.

    Parameters
    ----------
    model : xgboost
        Model built depending on model_task, in {xgboost.XGBClassifier, xgboost.XGBRegressor},
        already fitted with (X_train, Y_train) and cross-validated on X_train & X_valid
    X_valid : pd.DataFrame
    X_train_valid : pd.DataFrame
    Y_valid : pd.DataFrame
    Y_train_valid : pd.DataFrame
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}

    Returns
    -------
    Y_pred_train_valid : np.ndarray
        If model_task in {"classification", "regression"}, returns the predicted class or value: shape(nb_individuals,)
        If model_task == "multiclass", returns predict_proba by label: shape(nb_individuals, nb_labels)

    Raises
    ------
    NotImplementedError
    """
    if model_task == "classification":

        proba_valid = model.predict_proba(X_valid)[:, 1]
        proba_train_valid = model.predict_proba(X_train_valid)[:, 1]

        ## set y predicted with optimised thresholds
        best_threshold, best_fscore = compute_best_fscore(Y_valid, proba_valid)

        Y_pred_train_valid = (proba_train_valid >= best_threshold).astype(int)

    elif model_task == "multiclass":

        # in case of "multiclass", returns predict_proba (by label: shape(nb_individuals, nb_labels))
        # => will enable to compute different types of stat_scores (based on Y_pred -> auc, or on predict_proba -> log_loss)
        multi_predict_proba_train_valid = model.predict_proba(X_train_valid)
        Y_pred_train_valid = multi_predict_proba_train_valid

    elif model_task == "regression":

        Y_pred_train_valid = model.predict(X_train_valid)

        mse = mean_squared_error(Y_train_valid, Y_pred_train_valid)
        print("The mean squared error (MSE) on train_valid set: {:.4f}\n".format(mse))

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    return Y_pred_train_valid

def compute_best_fscore(Y_splitted_set: pd.DataFrame, proba_splitted_set: pd.DataFrame) -> pyplot:
    """Based on fscore, optimises the threshold of the model.
    It will permit to convert predicted probabilities into labels (Y predicted).

    Parameters
    ----------
    Y_splitted_set : pd.DataFrame
        Target, ie true labels of Y.
    proba_splitted_set : pd.DataFrame
        Vector of probabilities predicted by the model = p(Y==1) for binary classification.

    Returns
    -------
    pyplot
        A plot of precision / recall curve for the model showing the best threshold.
    """
    precision_valid, recall_valid, thresholds = precision_recall_curve(
        Y_splitted_set, proba_splitted_set
    )
    # convert to f score
    fscore = (2 * precision_valid * recall_valid) / (precision_valid + recall_valid)
    # locate the index of the largest f score
    ix = np.argmax(fscore)

    print(f"len(thresholds): {len(thresholds)}")
    print(Y_splitted_set.shape)

    best_threshold = thresholds[ix]
    best_fscore = fscore[ix]

    print("Best Threshold=%f, with F-Score=%.3f" % (best_threshold, best_fscore))

    # plot the roc curve for the model
    no_skill = len(Y_splitted_set[Y_splitted_set == 1]) / len(Y_splitted_set)
    train_fig = pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    train_fig = pyplot.plot(recall_valid, precision_valid, marker=".", label="Model")
    train_fig = pyplot.scatter(
        recall_valid[ix], precision_valid[ix], marker="o", color="black", label="Best"
    )
    # axis labels
    train_fig = pyplot.title("Statistical performance on valid set (PR AUC)")
    train_fig = pyplot.xlabel("Recall")
    train_fig = pyplot.ylabel("Precision")
    train_fig = pyplot.legend()
    # show the plot
    pyplot.show(train_fig)

    return best_threshold, best_fscore