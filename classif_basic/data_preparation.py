import os
import random
from math import floor
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xgboost
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost import XGBRegressor

# General data preparations
def handle_cat_features(X: pd.DataFrame, preprocessing_cat_features: str) -> pd.DataFrame:
    """Returns the initial DataFrame with the categorical features handled for machine-learning computation.

    Args:
        X : pd.DataFrame
            DataFrame with all features (entered as columns) concerning individuals
        preprocessing_cat_features: str
            Set the way categorial features are handled. 
            Keep unique columns and replace their values by numbers ("label_encoding"), or create one column per feature's value ("one_hot_encoding").
            Must be set to a value in {"label_encoding", "one_hot_encoding"}

    Raises:
        NotImplementedError:
            When the way the user wants to handle categorial features is not implemented. Must be set to a value in {'label_encoding', 'one_hot_encoding'}

    Returns:
        pd.DataFrame: the initial DataFrame with the categorical features handled for machine-learning computation.
    """

    if preprocessing_cat_features == "label_encoding":

        X_category = X.loc[:, X.dtypes == "category"]

        le = LabelEncoder()
        X_category = X_category.apply(le.fit_transform)

        # for further interpretation, to get back the original categories
        # X_category.apply(le.inverse_transform)

        # then, reconstitute the dataset with the label-encoded categorial columns 
        X_non_category = X.loc[:, X.dtypes != "category"]

        X = pd.concat([X_non_category, X_category], axis=1)

    elif preprocessing_cat_features == "one_hot_encoding":
        X = pd.get_dummies(X)
    
    else:
        raise NotImplementedError("The way you want to handle categorial features is not implemented. Must be set to a value in {'label_encoding', 'one_hot_encoding'}")
    
    return X

def train_valid_test_split(X: pd.DataFrame, Y: pd.DataFrame, model_task: str, preprocessing_cat_features: str="one_hot_encoding") -> pd.DataFrame:
    """Splits data into train, valid, and test set -> to train a model with cross-validation
    & train_valid set -> to inspect if the model is discriminant on the data it trained

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with all features (entered as columns) concerning individuals
    Y : pd.DataFrame
        Target to be predicted by the model (1 column for binary classification: int in {0,1})
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification"}
    preprocessing_cat_features: str, by default "one_hot_encoding"
        Set the way categorial features are handled. 
        Keep unique columns and replace their values by numbers ("label_encoding"), or create one column per feature's value ("one_hot_encoding", taken by default).
        Must be set to a value in {"label_encoding", "one_hot_encoding"}

    Returns
    -------
    pd.DataFrame
        The initial data is splitted in 8 DataFrames for training and unfairness detection purposes:
        X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test

    """

    SEED = 7
    VALID_SIZE = 0.15

    X = handle_cat_features(X=X, preprocessing_cat_features=preprocessing_cat_features)

    if model_task in {"classification", "multiclass"}:

        # Keep test values to ensure model is behaving properly
        X_model, X_test, Y_model, Y_test = train_test_split(
            X, Y, test_size=VALID_SIZE, random_state=SEED, stratify=Y
        )

        # Split valid set for early stopping & model selection
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_model, Y_model, test_size=VALID_SIZE, random_state=SEED, stratify=Y_model
        )

        # assess model's predictions (discriminant?) on the set it was trained (train&valid)
        X_train_valid = X_train.append(X_valid)
        Y_train_valid = Y_train.append(Y_valid)

    elif model_task == "regression":

        # Keep test values to ensure model is behaving properly
        X_model, X_test, Y_model, Y_test = train_test_split(
            X,
            Y,
            test_size=VALID_SIZE,
            random_state=SEED,  # stratify=Y # TODO stratify or stratify K-fold with regression?
        )

        # Split valid set for early stopping & model selection
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_model,
            Y_model,
            test_size=VALID_SIZE,
            random_state=SEED,  # stratify=Y_model
        )

        # assess model's predictions (discriminant?) on the set it was trained (train&valid)
        X_train_valid = X_train.append(X_valid)
        Y_train_valid = Y_train.append(Y_valid)

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    return X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test

def set_target_if_feature(
        df_response_if_feature: pd.DataFrame,
        df_no_response_if_feature: pd.DataFrame,
        df_response_if_not_feature: pd.DataFrame,
        df_no_response_if_not_feature: pd.DataFrame,
        len_dataset: int,
        percentage_feature: int,
        percentage_response_if_feature: int,
        percentage_response_if_not_feature: int)->pd.DataFrame:
    '''
    Selects a sub-sample of the dataset to control for the effect of a feature on the outcome. 
    Set the % when the target is reached (called "response"), given a specific feature's group. 
    Ex: if percentage_feature (for example 'sex_Male') = 40, then 40% of the new dataset will be men. Given these selected proportions,
    If then percentage_response_if_feature = 70, then 70% of the men will have a positive outcome (e.g. a loan granted),
    If the percentage_response_if_not_feature = 20, then 20% of the women will have a positive outcome. 

    df_response_if_feature: pd.DataFrame
        DataFrame with the individuals of the "feature" group entailing a positive outcome. 
        All sub-datasets are then used to compose the new dataset with controlled effects of the feature. 
    df_no_response_if_feature: pd.DataFrame
        DataFrame with the individuals of the "feature" group entailing a negative outcome.
    df_response_if_not_feature: pd.DataFrame
        DataFrame with the individuals which are not of the "feature" group entailing a positive outcome.
    df_no_response_if_not_feature: pd.DataFrame
        DataFrame with the individuals which are not of the "feature" group entailing a negative outcome.
    len_dataset: int
        Desired lenght of the selected dataset (number of lines, e.g. individuals).
        Warning: when there are too few individuals in one of the sub-DataFrames, the dataset must be shorter.
    percentage_feature: int
        Percentage of "feature" group that the user wants in the selected dataset.
    percentage_response_if_feature: int
        Percentage of positive outcomes in the "feature" group that the user wants in the selected dataset.
    percentage_response_if_not_feature: int)
        Percentage of positive outcomes out of the "feature" group that the user wants in the selected dataset.

    ''Returns''
    df_selected: pd.DataFrame
        The selected sub-sample of the dataset to control for the effect of a feature on the outcome, according to the percentages fixed by the user.

    '''
        
    # illustration here: say 40% indivs are feature and 80% of them respond, else without treatment 10% response
    # 80% of 40% indivs respond if feature
    ix_size_feature_resp = floor(len_dataset*percentage_feature/100*percentage_response_if_feature/100)
    print(f"len_dataset: {len_dataset}")
    print(f"nb indivs feature with response: {ix_size_feature_resp}")
    ix_feature_resp = np.random.choice(len(df_response_if_feature), size=ix_size_feature_resp, replace=False)
    df_response_if_feature = df_response_if_feature.iloc[ix_feature_resp]

    # (100-80)% of 40% indivs do not respond if feature
    ix_size_feature_no_resp = floor(len_dataset*percentage_feature/100*(100-percentage_response_if_feature)/100)
    print(f"nb indivs feature with no response: {ix_size_feature_no_resp}")
    ix_feature_no_resp = np.random.choice(len(df_no_response_if_feature), size=ix_size_feature_no_resp, replace=False)
    df_no_response_if_feature = df_no_response_if_feature.iloc[ix_feature_no_resp]

    percentage_not_feature = 100 - percentage_feature

    # 10% of (100-40)% indivs respond if not_feature
    ix_size_not_feature_resp = floor(len_dataset*percentage_not_feature/100*percentage_response_if_not_feature/100)
    print(f"nb indivs not_feature with response: {ix_size_not_feature_resp}")
    ix_not_feature_resp = np.random.choice(len(df_response_if_not_feature), size=ix_size_not_feature_resp, replace=False)
    df_response_if_not_feature = df_response_if_not_feature.iloc[ix_not_feature_resp]

    # (1-10)% of (100-40)% indivs do not respond if not_feature
    ix_size_not_feature_no_resp = floor(len_dataset*percentage_not_feature/100*(100-percentage_response_if_not_feature)/100)
    print(f"nb indivs not_feature with no response: {ix_size_not_feature_no_resp}")
    ix_not_feature_no_resp = np.random.choice(len(df_no_response_if_not_feature), size=ix_size_not_feature_no_resp, replace=False)
    df_no_response_if_not_feature = df_no_response_if_not_feature.iloc[ix_not_feature_no_resp]

    df_selected = df_response_if_feature.append([df_no_response_if_feature, df_response_if_not_feature, df_no_response_if_not_feature])
    
    return df_selected 

# Data Preparations for Specific Datasets 
def automatic_preprocessing(dataset_name: str) -> pd.DataFrame:
    """According to a specified dataset, returns the dataset pre-processed to allow more performant training.
    For the moment, only available with dataset_name == "housing_nights_dataset".

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    pd.DataFrame
    """
    if dataset_name == "housing_nights_dataset":

        requests_train = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/requests_train.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        requests_test = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/requests_test.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        composition = requests_train["group_composition_label"]
        nights = requests_train["granted_number_of_nights"]

        individuals_train = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/individuals_train.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        individuals_test = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/individuals_test.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        train_set = requests_train.loc[
            :,
            (
                "granted_number_of_nights",
                "child_situation",
                "district",
                "group_composition_id",
                "housing_situation_id",
                "number_of_underage",
                "request_id",
            ),
        ].set_index("request_id")

        # Step 1: Dataset with categorical features to be one-hot encoded
        list_train_set = [
            (individuals_train, "individual_role", "child"),
            (individuals_train, "gender", "female"),
            (requests_train, "victim_of_violence", "t"),
            (individuals_train, "individual_role", "isolated parent"),
            (requests_train, "child_to_come", "t"),
            (individuals_train, "individual_role_2_label", "child/underage with family"),
            (
                individuals_train,
                "housing_situation_2_label",
                "hotel paid by the emergency structure",
            ),
            (individuals_train, "housing_situation_2_label", "on the street"),
            (individuals_train, "housing_situation_2_label", "emergency accomodation"),
            (requests_train, "housing_situation_label", "hotel paid by an association"),
            (requests_train, "housing_situation_label", "mobile or makeshift shelter"),
            (
                requests_train,
                "housing_situation_label",
                "religious place (church, mosque, synogogue)",
            ),
            (requests_train, "housing_situation_label", "inclusion structure"),
            (requests_train, "housing_situation_label", "other"),
            (requests_train, "housing_situation_label", "emergency structure"),
            (requests_train, "housing_situation_label", "public hospital"),
        ]

        for previous_dataset, individual_column, criteria in list_train_set:
            train_set = new_dataset_column(train_set, previous_dataset, individual_column, criteria)

        # Step 2: add time data (to consider potentially vulnerable people, with min or max age)
        individuals_train["birth_year"] = individuals_train["birth_year"].fillna(
            individuals_train["birth_year"].mean()
        )
        train_set["min_birth_year"] = (
            individuals_train.loc[:, ["birth_year", "request_id"]].groupby("request_id").min()
        )
        train_set["max_birth_year"] = (
            individuals_train.loc[:, ["birth_year", "request_id"]].groupby("request_id").max()
        )

        # Step 3 : add the dtypes int64 from individuals_train (groupby -> min)
        train_set["housing_situation_2_id"] = (
            individuals_train.loc[:, ["housing_situation_2_id", "request_id"]]
            .groupby("request_id")
            .min()
        )
        train_set["individual_role_2_id"] = (
            individuals_train.loc[:, ["individual_role_2_id", "request_id"]]
            .groupby("request_id")
            .min()
        )
        train_set["marital_status_id"] = (
            individuals_train.loc[:, ["marital_status_id", "request_id"]]
            .groupby("request_id")
            .min()
        )

        requests_train["answer - group creation date"] = pd.to_datetime(
            requests_train["answer_creation_date"]
        ).values.astype(np.int64) - pd.to_datetime(
            requests_train["group_creation_date"]
        ).values.astype(
            np.int64
        )

        train_set["answer - group creation date"] = (
            requests_train.loc[:, ["answer - group creation date", "request_id"]]
            .groupby("request_id")
            .min()
        )
        # replace negative values of 'answer - group creation date' with -0.5, i.e. indicate a special category to XGBoost
        train_set.loc[
            train_set["answer - group creation date"] < 0, "answer - group creation date"
        ] = -1

    return train_set


def new_dataset_column(
    train_set: pd.DataFrame, previous_dataset: pd.DataFrame, individual_column: str, criteria: str
) -> pd.DataFrame:
    """Add a column "criteria" of the individuals_dataset to the train_set (grouped by households), by joining.
    For the moment, only available with "housing_nights_dataset".

    Parameters
    ----------
    train_set : pd.DataFrame
        The dataset to be augmented with data concerning individuals
    previous_dataset : pd.DataFrame
    individual_column : str
    criteria : str

    Returns
    -------
    pd.DataFrame
    """
    new_column = (
        previous_dataset[["request_id"]]
        .join(pd.get_dummies(previous_dataset[individual_column]))
        .groupby("request_id")
        .sum()
        .loc[:, criteria]
    )
    # If the new feature corresponds to the name of the previous_dataset column (binary, takes 2 values: true 't'==1 or false 't'==0)
    if criteria == "t":
        train_set[individual_column] = new_column
    # If the new feature corresponds to a sub-criteria of the individuals_train_column
    else:
        # we add "nb" because we grouped individuals of the household meeting this criteria
        train_set["nb_" + criteria] = new_column
    return train_set
