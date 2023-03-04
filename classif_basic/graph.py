import itertools
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential

from classif_basic.data_preparation import handle_cat_features

class GCN(torch.nn.Module): 
    """Class to generate a basic GCN with 2 convolutional layers.

    Here intervenes the quick "introduction by example" of GCN by torch
    in 'https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html'

    Methods:

        __init__
        Instantiates the basic GCN with 2 convolutional layers.
            data (torch_geometric.data.Data): the dataset shaped in a graph form
                data.x (torch.tensor): contains the input nodes (e.g. clients IDs)
                    shape (data.num_nodes, data.num_nodes_features)
                data.edge_index (torch.tensor): contains all the edges (i.e. adjacency matrix) joining the couples (2) of nodes
                    shape (2, data.num_edges)
                data.num_node_features (int): number of input nodes
                data.num_classes (int): number of classes to be predicted (e.g. 2 for binary classification)

        __forward__
        Forward propagation for training of the basic GCN with 2 convolutional layers.
            x: data.x
            edge_index: data.edge_index
        Must be passed independently of data, to enable to pass data.x (and not data.edge_index) 
        to GPU device during training (in our function train_GNN)
    """

    def __init__(self, data:torch):
        """Instantiates the basic GCN with 2 convolutional layers.

        Args:
            data.x (torch.tensor): contains the input nodes (e.g. clients IDs)
                shape (nb_nodes, nb_nodes_features)
            data.edge_index (torch.tensor): contains all the edges (i.e. adjacency matrix) joining the couples of nodes
                shape (2, nb_edges)            
        """
        super().__init__()

        if data.num_node_features is None:
            raise AttributeError("The number of node features 'data.num_node_features' must be specified to build the GCN.")
        if data.num_classes is None:
            raise AttributeError("The number of classes 'data.num_classes' must be specified to build the GCN.")

        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, x:torch, edge_index:torch)->torch:
        """
        Note: "backward" function is directly inherited from torch.nn.Module.

        Args:
            x: data.x
                data.x (torch.tensor): contains the input nodes (e.g. clients IDs)
                    shape (nb_nodes, nb_nodes_features)
            edge_index: data.edge_index
                data.edge_index (torch.tensor): contains all the edges (i.e. adjacency matrix) joining the couples of nodes
                    shape (2, nb_edges)

        x and edge_index must be passed independently of data, to enable to pass data.x (and not data.edge_index) 
        to GPU device during training (in our function train_GNN)

        Returns:
            torch: signal of layers activation for forward propagation (through softmax)
        """

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

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

def add_new_edge(data:pd.DataFrame, previous_edge:np.array, list_col_names:list)-> np.array: # with the list of columns to combine in the edge # TODO possible couples AND single features at the same time to create edges
    """Based on 1 or 2 joint features, computes combinations of individuals with same attributes - to create connections as edges of a future data graph. 

    Args:
        data (pd.DataFrame): Dataset to be transformed in a graph
        previous_edge (np.array): Array combining the couples of clients sharing attributes, of shape (2,nb_permutations_per_attribute_and_clients)
        list_col_names (List(str)): list with the names of the columns
            str: names of the columns, must be in a value in X.columns.

    Raises:
        NotImplementedError: if the user wants to combine more than 2 features to create connections between clients

    Returns:
        np.array: Array combining the indexes of couple of clients sharing the new attributes, i.e. common values of list_col_names
    """
    previous_edges = initialise_previous_edges(previous_edge=previous_edge)

    # first, reset IDs
    # to enable the computation of all combinations of clients sharing some attribute, i.e. column value (e.g. the same type of job)
    data["clients_id"] = data.reset_index().index

    if len(list_col_names)==1: # when a unique feature is chosen to form an edge

        col_name = list_col_names[0]
        attribute_values = data[col_name].unique()

        for attribute in attribute_values:
            # select clients with the same job
            attribute_df = data[data[col_name] == attribute]
            clients_edges = build_permutations_same_attribute(attribute_df=attribute_df)
            # complete with each new attribute (e.g. new type of job), to get all couples of clients with the same attribute
            previous_edges = np.vstack([previous_edges, clients_edges]) 
    
    elif len(list_col_names) == 2: # for the moment, maximum combination of 2 columns to create an edge
    
    # TODO join if too many categories (e.g. hours of work per week)
    # else, 1050 combinations of types of jobs and hours per week - a bit hard to compute
    # and irrelevant (mini-categories of clients as edges)...
        col_1 = list_col_names[0]
        col_2 = list_col_names[1]
            
        combinations_vals_cols_1_to_2 = np.array(np.meshgrid(data[col_1].unique(), data[col_2].unique())).T.reshape(-1,2)

        for attr1, attr2 in combinations_vals_cols_1_to_2:
            attribute_df = data.loc[(data[col_1] == attr1) & (data[col_2] == attr2)]
            clients_edges = build_permutations_same_attribute(attribute_df=attribute_df)
            # complete with each new attribute (e.g. new type of job), to get all couples of clients with the same attribute
            previous_edges = np.vstack([previous_edges, clients_edges]) 
    
    else:
        raise NotImplementedError("The maximum number of features you specify in list_col_names to create an edge must be 2.")

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

def train_valid_test_split_masks(X:pd.DataFrame, Y: pd.DataFrame, preprocessing_cat_features:str="label_encoding")->Tuple:
    """To train a GNN model with cross-validation,
    Splits data into train, valid, and test set -> get the train/valid/test indices in "masks" that will be passed to graph data 

    Args:
        X : pd.DataFrame
            DataFrame with all features (entered as columns) concerning individuals
        Y : pd.DataFrame
            Target to be predicted by the model (1 column for binary classification: int in {0,1})
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

    X = handle_cat_features(X=X, preprocessing_cat_features=preprocessing_cat_features)

    # Keep test values to ensure model is behaving properly
    X_model, X_test, Y_model, Y_test = train_test_split(
        X, Y, test_size=VALID_SIZE, random_state=SEED, stratify=Y # "stratify=Y" to keep the same proportion of target classes in train, valid and test sets
    )

    # Split valid set for early stopping & model selection
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_model, Y_model, test_size=VALID_SIZE, random_state=SEED, stratify=Y_model
    )

    # get the masks corresponding to train/valid/test samples, to keep the train/valid/test indices on graph data while GNN training
    train_mask = create_mask(X=X, X_subset=X_train)
    valid_mask = create_mask(X=X, X_subset=X_valid)
    test_mask = create_mask(X=X, X_subset=X_test)

    return train_mask, valid_mask, test_mask


def table_to_graph(X:pd.DataFrame, Y:pd.DataFrame, list_col_names:list, edges: np.array=None)->torch:
    """Transforms tabular data in a graph,
    From a given tabular pd.DataFrame separated in X and Y, and a list of features to connect them.

    Args:
        X (pd.DataFrame): Tabular Dataset (without the target)
        Y (pd.DataFrame): Tabular Dataset (target)
        list_col_names (List(str)): list with the names of the columns that are used as node features (=> not as edges, for the moment)
            str: names of the columns, must be in a value in X.columns.
        mask (bool), by default None: boolean tensor indicating if the individual is in X_train
            Must be specified if the graph will be transformed into mini-batches (for faster computing) through sample-neighborhood

    Returns:
        torch_geometric.data.Data: graph of the previous tabular data, connected with the features of list_col_names
    """
    
    # train/valid/test split: keep the train/valid/test indices (masks) on graph data for future GNN training
    train_mask, valid_mask, test_mask = train_valid_test_split_masks(
        X=X,
        Y=Y)

    #Make sure that we have no duplicate nodes
    assert(X.index.unique().shape[0] == X.shape[0])
    
    # first of all, reset the IDs of clients
    X["clients_id"] = X.reset_index().index
    
    # Extract the node features

        # The node features are typically represented in a matrix of the shape (num_nodes, node_feature_dim).
        # For each of the bank clients, we simply extract their attributes (except here the "occupation", that would be used as an "actionable" edge to connect them)
    list_X_cols = X.columns.to_list()
    list_nodes_names = [col for col in list_X_cols if col not in list_col_names]
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

    # if the edges are not provided, extract the edges with our function to combine columns of "list_col_names"
    if edges is None:
        # TODO enhance connection between edges and graph creation (in the 2 functions)
        previous_edge = None
        data = X # TODO delete the name X to make graph from tabular data clearer (X_train, or X_valid...)? But what about Y?
        edges = add_new_edge(data=data, previous_edge=previous_edge, list_col_names=list_col_names)
    
    # then convert to torch, for further compatibility avec the torch GNN
    edge_index = torch.from_numpy(edges)
    # and transform the edges indexes into int64, to enable forward propagation of GNN
    edge_index = edge_index.long()
    
    # finally, build the graph (if other attributes e.g. edge_features, you can also pass it there)
    data = Data(x=x, edge_index=edge_index, y=y, num_classes=num_classes, is_directed=True, 
                train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
    
    return data

def check_attributes_graph_data(data:torch):
    """Check if the graph data have all the required attributes for our GCN training (with the function train_GNN).

    Args:
        data (torch_geometric.data.Data): the dataset in a graph 
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
    if data.test_mask is None:
        raise AttributeError("The test_mask 'data.test_mask', must be specified."
                             "\n data.test_mask must be a boolean tensor of shape (data.num_nodes), indicating if the node is used for testing the model on unseen data graph")

    return 

def train_GNN(
    data_total:torch,
    loader_method:str,
    batch_size:int = 32,
    epoch_nb:int = 35,
    learning_rate:float = 0.001,
    nb_neighbors_per_sample:int = 30,
    nb_iterations_per_neighbors:int = 2)->GCN:
    """_summary_

    Args:
        data_total (torch_geometric.data.Data): the whole dataset in a graph 
            Must have the attributes 
        loader_method (str): _description_
        batch_size (int, optional): _description_. Defaults to 32.
        epoch_nb (int, optional): _description_. Defaults to 35.
        learning_rate (float, optional): _description_. Defaults to 0.001.
        nb_neighbors_per_sample (int, optional): _description_. Defaults to 30.
        nb_iterations_per_neighbors (int, optional): _description_. Defaults to 2.

    Returns:
        GCN: _description_
    """

    # first of all, check if data_total entails all the attributes for our GNN batch training
    check_attributes_graph_data(data_total)

    print(f"\n Construction of the loader with {batch_size} batches and the method {loader_method}") 
    # TODO internal function to build the loaders
        # test different batch construction to enforce causal hierarchy -> mini-graphs, neighborhoods...
    loader = NeighborLoader(
        data_total,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[nb_neighbors_per_sample] * nb_iterations_per_neighbors,
        # Use a batch size of 128 for sampling training nodes
        batch_size=batch_size,
        input_nodes=data_total.train_mask,
    )

    t_basic_1 = time.time()

    # activate and signal the use of GPU for faster processing
    if torch.cuda.is_available():    
        print("Using GPU!")    
        device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')   
    else:    
        print("Using CPU!")       
        device = torch.device("cpu")

    # initialize the structure of the classifier, and prepare for GNN training (with GPU)
    classifier = GCN(data_total).to(device)

    # classifier = GraphClassifier(v_in=71, e_in=6, v_g=30, e_g=6, v_out=200, mc_out=22, i_types=11).float().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()

    print('starting training')
    classifier.train()

    for epoch in range(epoch_nb):
        
        epoch_loss = 0
        total = 0
        correct = 0
        classifier.train()
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(device)
            x = data.x.to(device)
            edge_index=data.edge_index.to(device)
            target = data.y.to(device)
            preds = classifier(x=x.float(), edge_index=edge_index)
            error_train = loss(preds[data.train_mask], target[data.train_mask])
            _, preds_temp = torch.max(preds.data, 1)
            total = total + len(target)
            correct = correct + (preds_temp == target).sum().item()
            epoch_loss = epoch_loss + error_train.item()
            error_train.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1} Loss = {epoch_loss/(i+1)} Train Accuracy = {correct / total}') 

    t_basic_2 = time.time()            

    print(f"Training of the basic GCN on Census on {data_total.x.shape[0]} nodes and {data_total.edge_index.shape[1]} edges, \n with {batch_size} batches and {epoch_nb} epochs took {(t_basic_2 - t_basic_1)/60} mn")

    return classifier
