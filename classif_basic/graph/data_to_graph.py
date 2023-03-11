import itertools
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from classif_basic.data_preparation import handle_cat_features

def initialise_previous_edges(previous_edge:np.array)->np.array:
    """_summary_

    Args:
        previous_edge (np.array): Array combining the couples of clients sharing attributes, of shape (2,nb_permutations_per_attribute_and_clients)

    Returns:
        np.array: initialise an empty index (or index with the previous edge values), enabling to add the new edges
    """
    if previous_edge is None:
        previous_edges = np.array([], dtype=np.int32).reshape((0, 2))
    
    elif previous_edge is not None:
        previous_edges = previous_edge.transpose()
    
    return previous_edges

def build_permutations_same_attribute(attribute_df:pd.DataFrame)->np.array:
    """Returns all the couples of clients sharing the same attributes (or intersection of attributes if a list is provided)

    Args:
        attribute_df (pd.DataFrame): DataFrame selecting clients with a unique value of attribute (or intersection of attributes)

    Returns:
        np.array: all the couples of clients sharing the same attributes (or intersection of attributes if a list is provided)
    """
    clients = attribute_df["clients_id"].values        
    # Build all combinations between clients with the same attribute e.g. job (without knowing their label)
    permutations = list(itertools.combinations(clients, 2))
    edges_source = [e[0] for e in permutations] # starting client -> to other client with the same attribute e.g. job
    edges_target = [e[1] for e in permutations] # ending client -> from other client with the same attribute e.g. job
    clients_edges = np.column_stack([edges_source, edges_target]) # convert combinations to array

    return clients_edges 

def add_new_edge(data:pd.DataFrame, previous_edge:np.array, list_edges_names:list)-> np.array: # with the list of columns to combine in the edge # TODO possible couples AND single features at the same time to create edges
    """Based on 1 or 2 joint features, computes combinations of individuals with same attributes - to create connections as edges of a future data graph. 

    Args:
        data (pd.DataFrame): Dataset to be transformed in a graph
        previous_edge (np.array): Array combining the couples of clients sharing attributes, of shape (2,nb_permutations_per_attribute_and_clients)
        list_edges_names (List(str)): list with the names of the columns that will be added as edges (if [feat1, feat2], will be directed edge from feat1->feat2)
            str: names of the columns, must be in a value in X.columns.

    Raises:
        NotImplementedError: if the user wants to combine more than 2 features to create connections between clients

    Returns:
        np.array: Array combining the indexes of couple of clients sharing the new attributes, i.e. common values of list_edges_names
    """
    previous_edges = initialise_previous_edges(previous_edge=previous_edge)

    # first, reset IDs
    # to enable the computation of all combinations of clients sharing some attribute, i.e. column value (e.g. the same type of job)
    data["clients_id"] = data.reset_index().index

    if len(list_edges_names)==1: # when a unique feature is chosen to form an edge

        col_name = list_edges_names[0]
        attribute_values = data[col_name].unique()

        for attribute in attribute_values:
            # select clients with the same job
            attribute_df = data[data[col_name] == attribute]
            clients_edges = build_permutations_same_attribute(attribute_df=attribute_df)
            # complete with each new attribute (e.g. new type of job), to get all couples of clients with the same attribute
            previous_edges = np.vstack([previous_edges, clients_edges]) 
    
    elif len(list_edges_names) == 2: # for the moment, maximum combination of 2 columns to create an edge
    
    # TODO join if too many categories (e.g. hours of work per week)
    # else, 1050 combinations of types of jobs and hours per week - a bit hard to compute
    # and irrelevant (mini-categories of clients as edges)...
        col_1 = list_edges_names[0]
        col_2 = list_edges_names[1]
            
        combinations_vals_cols_1_to_2 = np.array(np.meshgrid(data[col_1].unique(), data[col_2].unique())).T.reshape(-1,2)

        for attr1, attr2 in combinations_vals_cols_1_to_2:
            attribute_df = data.loc[(data[col_1] == attr1) & (data[col_2] == attr2)]
            clients_edges = build_permutations_same_attribute(attribute_df=attribute_df)
            # complete with each new attribute (e.g. new type of job), to get all couples of clients with the same attribute
            previous_edges = np.vstack([previous_edges, clients_edges]) 
    
    else:
        raise NotImplementedError("The maximum number of features you specify in list_edges_names to create an edge must be 2.")

    # Convert to Pytorch Geometric format
    edge_index = previous_edges.transpose()
    # edge_index # [2, num_edges]
    # then convert to torch, for further compatibility avec the torch GNN
    # edge_index = torch.from_numpy(edge_index)

    return edge_index

def create_mask(X:pd.DataFrame, X_subset:pd.DataFrame)->torch.tensor:
    """Creates a mask indicating if the row of X (future node of the graph data) is in the subset of X.
    Will be used when transforming tabular data to graph, in order to keep the train/valid/test indices on graph data while GNN training.

    Args:
        X (pd.DataFrame):
            DataFrame with all features (entered as columns) concerning individuals
        X_subset (pd.DataFrame): 
            Subset of X generated with the train/test/valid split

    Returns:
        torch.tensor: the subset mask, i.e. boolean tensor indicating if the individual is in X_subset
    """
    X_copy = X.copy()
    X_copy.loc[X_subset.index, "mask"] = 1
    X_copy["mask"] = X_copy["mask"].fillna(0).astype(bool)

    mask = torch.tensor(X_copy['mask'].values)

    return mask

def train_valid_split_mask(X_model:pd.DataFrame, Y_model: pd.DataFrame, preprocessing_cat_features:str="label_encoding")->Tuple:
    """To train a GNN model with cross-validation,
    Splits data into train/valid set -> get the train/valid indices in "masks" that will be passed to graph data
    (!) must exclude the test samples 

    Args:
        X_model : pd.DataFrame
            DataFrame with all features (entered as columns) concerning individuals,
            Which will be used for training (!) must exclude the test samples
        Y_model : pd.DataFrame
            Target to be predicted by the model (1 column for binary classification: int in {0,1})
            Which will be used for training (!) must exclude the test samples
        preprocessing_cat_features: str, by default "label_encoding"
            Set the way categorial features are handled. 
            Keep unique columns and replace their values by numbers ("label_encoding", taken by default), or create one column per feature's value ("one_hot_encoding").
            Must be set to a value in {"label_encoding", "one_hot_encoding"}
    Returns:
        Tuple(torch.tensor): the 3 masks (train/valid/test)
            E.g. train_mask is a boolean tensor indicating if the individual is in X_train
    """

    SEED = 7
    VALID_SIZE = 0.15

    X_model = handle_cat_features(X=X_model, preprocessing_cat_features=preprocessing_cat_features)

    # Split valid set for early stopping & model selection
    # "stratify=Y" to keep the same proportion of target classes in train and valid sets 
    X_train, X_valid, y, Y_valid = train_test_split(
        X_model, Y_model, test_size=VALID_SIZE, random_state=SEED, stratify=Y_model
    )

    # get the masks corresponding to train/valid/test samples, to keep the train/valid/test indices on graph data while GNN training
    train_mask = create_mask(X=X_model, X_subset=X_train)
    valid_mask = create_mask(X=X_model, X_subset=X_valid)

    return train_mask, valid_mask


def table_to_graph(X:pd.DataFrame, Y:pd.DataFrame, list_edges_names:list, edges: np.array=None)->torch:
    """Transforms tabular data in a graph,
    From a given tabular pd.DataFrame separated in X and Y, and a list of features to connect them.

    Args:
        X (pd.DataFrame): Tabular Dataset (without the target)
        Y (pd.DataFrame): Tabular Dataset (target)
        list_edges_names (List(str)): list with the names of the columns that are used as edges (=> not as node features for the moment)
            str: names of the columns, must be in a value in X.columns.
        mask (bool), by default None: boolean tensor indicating if the individual is in X_train
            Must be specified if the graph will be transformed into mini-batches (for faster computing) through sample-neighborhood

    Returns:
        torch_geometric.data.Data: graph of the previous tabular data, connected with the features of list_edges_names
    """
    
    # train/valid split: keep the train/valid indices (masks) on graph data for future GNN training
    train_mask, valid_mask = train_valid_split_mask(
        X_model=X,
        Y_model=Y)

    #Make sure that we have no duplicate nodes
    assert(X.index.unique().shape[0] == X.shape[0])
    
    # first of all, reset the IDs of clients
    X["clients_id"] = X.reset_index().index
    
    # Extract the node features

        # The node features are typically represented in a matrix of the shape (num_nodes, node_feature_dim).
        # For each of the bank clients, we simply extract their attributes (except here the "occupation", that would be used as an "actionable" edge to connect them)
    list_X_cols = X.columns.to_list()
    list_nodes_names = [col for col in list_X_cols if col not in list_edges_names]
    node_features = X[list_nodes_names]
        # That's already our node feature matrix. The number of nodes and the ordering is implicitly defined by it's shape. Each row corresponds to one node in our final graph. 
    
    # Convert to numpy
    x = node_features.to_numpy()
    # x.shape # [num_nodes x num_features]
    # then convert to torch, for further compatibility avec the torch GNN
    x = torch.from_numpy(x)
    
    # Extract the labels
    labels = Y
        # Those are simply the wealthiness of each of the clients (if their income is >$50 000). This corresponds to a node-level prediction problem. 
        # Therefore we have as many labels as we have nodes.
    
    # to make the graph functioning, check that the nodes follow the same order than the labels (rows n°)
        # else, sort values by ids
    nb_corresponding_nodes_labels = (labels.index == node_features.index).sum()
    assert(nb_corresponding_nodes_labels == X.shape[0])
    
    # Convert to numpy
    y = labels.to_numpy()
    #y.shape # [num_nodes, 1] --> node regression
    # get the number of classes
    num_classes=np.unique(y).shape[0]
    # then convert to torch, for further compatibility avec the torch GNN
    y = torch.from_numpy(y)

    # if the edges are not provided, extract the edges with our function to combine columns of "list_edges_names"
    if edges is None:
        # TODO enhance connection between edges and graph creation (in the 2 functions)
        previous_edge = None
        data = X # TODO delete the name X to make graph from tabular data clearer (X_train, or X_valid...)? But what about Y?
        edges = add_new_edge(data=data, previous_edge=previous_edge, list_edges_names=list_edges_names)
    
    # then convert to torch, for further compatibility avec the torch GNN
    edge_index = torch.from_numpy(edges)
    # and transform the edges indexes into int64, to enable forward propagation of GNN
    edge_index = edge_index.long()
    
    # finally, build the graph (if other attributes e.g. edge_features, you can also pass it there)
    data = Data(x=x, edge_index=edge_index, y=y, num_classes=num_classes, is_directed=True, 
                train_mask=train_mask, valid_mask=valid_mask)
    
    return data
