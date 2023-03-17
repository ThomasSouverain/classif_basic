from math import floor

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import FactorAnalysis

def check_attributes_graph_data(data:torch):
    """Check if the graph data have all the required attributes for our GCN training (with the function train_GNN).

    Args:
        data (torch_geometric.data.Data): the balanced_df in a graph 
    """
    if data.x is None:
        raise AttributeError("The x 'data.x', must be specified to build the GCN."
                             "\n data.x must be a torch tensor of shape (data.num_nodes, data.num_features)"
                             "\n For clients binary classification, data.x might be (nb_clients, nb_features not used as edges)")
    if data.edge_index is None:
        raise AttributeError("The edge_index 'data.edge_index', must be specified to build the GCN."
                             "\n data.edge_index must be a torch tensor of shape (2, num_edges),"
                             "\n adjacency matrix connecting the couples of nodes (e.g. clients sharing the same type of job and hours per week)")
    if data.y is None:
        raise AttributeError("The target 'data.y', must be specified to build the GCN."
                             "\n data.y must be a torch tensor of shape (data.num_nodes, data.num_classes-1)"
                             "\n For clients binary classification, data.y might be (0, ..., 1) of shape (nb_clients)")
    if data.num_classes is None:
        raise AttributeError("The number of classes 'data.num_classes' must be specified to build the GCN.")
    if data.num_node_features is None:
        raise AttributeError("The number of node features 'data.num_node_features' must be specified to build the GCN.")
    if data.train_mask is None:
        raise AttributeError("The mask 'data.train_mask', must be specified to build the GCN."
                             "\n data.train_mask must be a boolean tensor of shape (data.num_nodes), indicating if the node is used for training")
    if data.valid_mask is None:
        raise AttributeError("The valid_mask 'data.valid_mask', must be specified to build the GCN."
                             "\n data.valid_mask must be a boolean tensor of shape (data.num_nodes), indicating if the node is used for validation during GNN training")

    return 

def check_batch_info(data_total:torch, batch_size:int, nb_batches:int)->tuple:
    # check that valid batch information is passed, and complete the information on batches (size of splits, number of individual per splits)
    if batch_size is not None:
        nb_indivs_total = data_total.x.shape[0]        
        nb_batches = int(nb_indivs_total/batch_size)        

    elif nb_batches is not None:
        nb_indivs_total = data_total.x.shape[0]        
        batch_size = int(nb_indivs_total/nb_batches)

    else:
        raise NotImplementedError("For GNN training on large data, you must specify either the size of splits (batch_size) or the number of individuals per split (nb_batches)")

    return batch_size, nb_batches

def tensor_to_numpy(tensor:torch.tensor)->np.array:
    """Converts a tensor into numpy (if tensor).
     Serves to compute and plot metrics & avoids waste of GPU memory (CPU device).

    Args:
        tensor (torch.tensor) 

    Returns:
        np.array
    """
    if torch.is_tensor(tensor):
        numpy_vector = tensor.cpu().detach().numpy()
    else:
        raise NotImplementedError("The vector you want to convert to numpy is not a tensor")
    return numpy_vector


def get_balanced_df(X:pd.DataFrame, Y:pd.DataFrame)->pd.DataFrame:
    """From an imbalanced DataFrame with 2 classes "0" and "1" (for the moment, in favour of class "0"), returns a new 50/50 balanced DataFrame.

    Args:
        Y (pd.DataFrame): DataFrame with the individuals as rows and features as columns
        Y (pd.DataFrame): DataFrame with the classes (0,1) as rows and features as columns

    Returns:
        pd.DataFrame: new 50/50 balanced DataFrame across the 2 classes "0" and "1"
    """

    df_total = X.copy()
    df_total["target"] = Y

    df_true_positive = df_total.loc[df_total["target"]==1]
    nb_max_true_positive = df_true_positive.shape[0]
    print(f"nb_max_true_positive: {nb_max_true_positive}")

    df_true_negative = df_total.loc[df_total["target"]==0]
    nb_max_true_negative = df_true_negative.shape[0]
    print(f"nb_max_true_negative: {nb_max_true_negative}")

    # reset indexes to enable shuffling and merges of selected proportions of targets
    df_true_positive.reset_index(drop=True, inplace=True)
    df_true_negative.reset_index(drop=True, inplace=True)

    ix_size_new_negative = nb_max_true_positive
    print(f"New nb indivs with true negative: {ix_size_new_negative}")
    ix_new_negative = np.random.choice(df_true_negative.index, size=ix_size_new_negative, replace=False)
    df_true_negative_new = df_true_negative.iloc[ix_new_negative]

    print(df_true_negative_new.shape[0])

    balanced_df=df_true_positive.copy()
    balanced_df=balanced_df.append(df_true_negative_new).reset_index(drop=True)

    print(f"Number of individuals (balanced_df): {balanced_df.shape[0]}")
    return balanced_df

def get_unified_col(X: pd.DataFrame, list_cols_to_join: list, new_col_name:str)->pd.DataFrame:
    """From 2 columns of X, generates a new DataFrame with 1 column - correlating them through factor analysis.
    Will serve to merge information of columns of the same type to set causal constraints (e.g. from "education" and "education-num" -> "education")

    Args:
        X (pd.DataFrame): DataFrame with the features 
        list_cols_to_join (list): list of the 2 columns to unify
            Must be of len == 2;
            Must be set to values in X.columns
        new_col_name (str): name of the new column, that will replace the 2 previous in the new DataFrame

    Returns:
        pd.DataFrame: X replacing the 2 previous columns by X[new_col_name]
    """
# unify same type columns to set causal constraints 

    two_to_one_factor = FactorAnalysis(n_components=1, rotation='varimax') 

    unified_col = two_to_one_factor.fit_transform(X.filter(items = list_cols_to_join))

    X = X.drop(list_cols_to_join, axis=1)
    X[new_col_name] = unified_col

    return X

def normalize_df(df:pd.DataFrame, normalization:str)->pd.DataFrame:
    """Returns the DataFrame normalized with either the 

    Args:
        df (pd.DataFrame): the initial DataFrame
        normalization (str): either by mean, or min-max value
            Must be set to a value in {'mean','min_max'}

    Returns:
        pd.DataFrame: the normalized DataFrame
    """
    if normalization=='mean':
        normalized_df=(df-df.mean())/df.std()

    elif normalization=='min_max':
        normalized_df=(df-df.min())/(df.max()-df.min())
    
    else:
        raise NotImplementedError("This normalization technique of the DataFrame is not implemented. Must be set to a value in {'mean','min_max'}")
    
    return normalized_df

def activate_gpu()->torch.device:
    """Activate and signal the use of GPU for faster processing

    Returns (torch.device):
        Activated device for GNN computation, using either GPU or CPU
    """
    if torch.cuda.is_available():    
        print("\n Using GPU!")    
        device = torch.device("cuda")
    else:    
        print("\n Using CPU!")       
        device = torch.device("cpu")
    return device
