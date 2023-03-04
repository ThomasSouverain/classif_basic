import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential

class GCN(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, x, edge_index):

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

# TODO enhance the function (and then include it in the package)

def table_to_graph(X:pd.DataFrame, Y:pd.DataFrame, list_col_names:list, edges: np.array=None, train_mask:bool=None)->torch:
    """Transforms tabular data in a graph,
    From a given tabular pd.DataFrame separated in X and Y, and a list of features to connect them.

    Args:
        X (pd.DataFrame): Tabular Dataset (without the target)
        Y (pd.DataFrame): Tabular Dataset (target)
        list_col_names (List(str)): list with the names of the columns
            str: names of the columns, must be in a value in X.columns.
        train_mask (bool), by default None: boolean tensor indicating if the individual is in X_train
            Must be specified if the graph will be transformed into mini-batches (for faster computing) through sample-neighborhood

    Returns:
        torch_geometric.data.data.Data: graph of the previous tabular data, connected with the features of list_col_names
    """
    
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
    data = Data(x=x, edge_index=edge_index, y=y, num_classes=num_classes, is_directed=True, train_mask=train_mask)
    
    return data

def train_GNN(
    data_total,
    loader_method,
    batch_size = 32,
    epoch_nb = 35,
    learning_rate = 0.001,
    nb_neighbors_per_sample = 30,
    nb_iterations_per_neighbors = 2):

    print(f"{data_total.train_mask.size} training examples \n")
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
            err = loss(preds, target)
            _, preds_temp = torch.max(preds.data, 1)
            total += len(target)
            correct += (preds_temp == target).sum().item()
            epoch_loss += err.item()
            err.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1} Loss = {epoch_loss/(i+1)} Train Accuracy = {correct / total}') 

    t_basic_2 = time.time()            
    print(f"Training of the basic GCN on Census with {batch_size} batches and {epoch_nb} epochs took {(t_basic_2 - t_basic_1)/60} mn")

    return classifier
