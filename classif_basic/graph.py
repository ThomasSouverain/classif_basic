import itertools
import time
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential

from classif_basic.data_preparation import handle_cat_features
from classif_basic.loader import IndexLoader

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

    def forward(self, data:torch, device:torch.device)->torch: # TODO adapt docstring if it works 
    # def forward(self, x:torch, edge_index:torch)->torch:
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

        x = data.x.float().to(device)
        edge_index=data.edge_index.to(device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GCN_ancestor_edges(torch.nn.Module):
    # TODO pass data each time a causal child is added -> really scalable? Test it on large data, and improve it ; investigate philosophical basis 
    # test a Sequential model - which will then respect the order between layer 1 and layer 2
    # reasoning: neural network => causal "productive" influence of layer 1 on layer 2
    # TODO doc if it works 
    def __init__(self, data):
        super().__init__()

        if data.num_node_features is None:
            raise AttributeError("The number of node features 'data.num_node_features' must be specified to build the GCN.")
        if data.num_classes is None:
            raise AttributeError("The number of classes 'data.num_classes' must be specified to build the GCN.")

        self.conv1 = GCNConv(data.num_node_features, 16)
        # intermediate convolutions when adding node features (descendant graph-data)
        self.conv2 = GCNConv(data.num_node_features+1, 16) # TODO initialize these layers with the exact nb of node features computed here -> pass data_childs to __init__()
        self.conv3 = GCNConv(data.num_node_features+2, 16)
        self.conv_end = GCNConv(16, data.num_classes)

    def forward(self, x_parent, x_child1, x_child2, x_final_descendants, edge_index_parent, edge_index_child1, edge_index_child2, edge_index_final_descendants, device): 

        x_parent = x_parent.float().to(device)
        x_child1 = x_child1.float().to(device)
        x_child2 = x_child2.float().to(device)
        x_final_descendants = x_final_descendants.float().to(device)

        edge_index_parent=edge_index_parent.to(device)
        edge_index_child1=edge_index_child1.to(device)
        edge_index_child2=edge_index_child2.to(device)
        edge_index_final_descendants = edge_index_final_descendants.to(device)

        # layer 1: only causal parent information (e.g. edge = "sex")
        x_parent = self.conv1(x_parent, edge_index_parent)
        print(f"shape of x_parent after convolution: {x_parent.shape} \n")        
        x = F.relu(x_parent)
        x = F.dropout(x, training=self.training)

        # layer 2: "add" causal descendant information through child1 edge (e.g. "sex" -> "education")
        x_child1 = self.conv2(x_child1, edge_index_child1) + x + x_parent # add the shortcut to not erase the ascendance of parent edge (cause: x_parent built with only parent edge)
        print(f"shape of x_child1 after convolution: {x_child1.shape} \n")        
        x = F.relu(x_child1)
        x = F.dropout(x, training=self.training)
  
        # layer 3: "add" causal descendant information through child2 edge (e.g. "education" -> "job")
        x_child2 = self.conv3(x_child2, edge_index_child2) + x + x_child1 # add the shortcut to not erase the ascendance of child1 edge (cause: x_parent built with only parent edge)
        print(f"shape of x_child2 after convolution: {x_child2.shape} \n")        
        x = F.relu(x_child2)
        x = F.dropout(x, training=self.training)

        # last convolutional layer: finish with all the "final descendant" features -> for more powerful computation
        # layer 4: link with the final descendant (e.g. "job" -> "hours of work per week")
        x_final_descendants = self.conv2(x_final_descendants, edge_index_final_descendants) + x + x_child1 + x_child2 # conv2 because 3 node features (!) TODO define explicitly in __init__() TODO only pass nb, not full date (to enhance computation)
        print(f"shape of x_final_descendants after convolution: {x_final_descendants.shape} \n")        
        x = F.relu(x_final_descendants)
        x = F.dropout(x, training=self.training)

        x = self.conv_end(x, edge_index_final_descendants)

        return F.log_softmax(x, dim=1)

class GCN_ancestor(torch.nn.Module):
    '''Train a GNN by passing multiple (successive) data-graphs, progressively integrating new features.
    - data are directly passed for training as initialization of the layers, instead of x and edge_index (vs GCN_ancestor_edge which will try to be lighter)
    - which makes possible loading for GNN training with large data? 
    But not very scalable (each causal descendant + last layer with all features => add a new graph to the loader...)
    '''
    # TODO pass data each time a causal child is added -> really scalable? Test it on large data, and improve it ; investigate philosophical basis 
    # test a Sequential model - which will then respect the order between layer 1 and layer 2
    # reasoning: neural network => causal "productive" influence of layer 1 on layer 2
    # TODO doc if it works 
    def __init__(self, list_data):
        super().__init__()

        # idea: insert in a list convolutional function for every graph-data, 
        # as with with a new graph-data the input shape may change (adding new features, feature nodes shape is changing)
        list_conv_fcts = []

        for i, data in enumerate(list_data):

            if data.num_node_features is None:
                raise AttributeError("The number of node features 'data.num_node_features' must be specified to build the GCN.")
            if data.num_classes is None:
                raise AttributeError("The number of classes 'data.num_classes' must be specified to build the GCN.")

            conv = GCNConv(data.num_node_features, 16)
            list_conv_fcts.append(conv)

        self.list_conv_fcts = list_conv_fcts

        self.conv_end = GCNConv(16, data.num_classes)
        # temporary check: OK if conv layer initialized?
        self.conv1 = GCNConv(data.num_node_features, 16)

    def forward(self, list_data, device): 

        # TODO delete temporary diminutions (self.conv1 -> uniquely specify "self"?, no shortcut through new_x_parent -> broadcast?)

        # layer 1: only causal parent information (e.g. edge = "sex")
        # layer 2: "add" causal descendant information through child1 edge (e.g. "sex" -> "education")
        # layer 3: "add" causal descendant information through child2 edge (e.g. "education" -> "job")
        # ... until last convolutional layer: finish with all the "final descendant" features -> for more powerful computation
        # layer 4: link with the final descendant (e.g. "job" -> "hours of work per week")
        # (!) do not pass too many layers (else, GNN over-smoothing problem)

        # for all graph-data, pass its (shape-fitted) convolutional function
        # and add previous parent information as shortcuts <=> do not erase the previous parent information
        new_x_parents = 0
        
        for i, data in enumerate(list_data[:-1]): # loop for the (n) first ancestor data
            print(f"\n Layer with causal ascendance {i+1} : ")
            data = data.to(device)
            x = data.x.float().to(device)
            edge_index=data.edge_index.to(device)

            print(f"shape of x before convolution: {x.shape} ; ")        

            x = self.conv1(x, edge_index) #+ new_x_parents # add the shortcut of previous parents
            # x = self.list_conv_fcts[i](x, edge_index) + new_x_parents # add the shortcut of previous parents
            print(f"shape of x after convolution: {x.shape} \n")        
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            # TODO below -> add the child(n) as parent of child (n+1) for further shortcuts
            # new_x_parents = new_x_parents + x

        print(f"Last Child Layer : ")
        # last convolutional layer with the last child data (n) (i.e. the last data_end_childs registered)
        data_end_childs = list_data[-1]
        x_end_childs = data_end_childs.x.float().to(device)
        edge_index_end_childs = data_end_childs.edge_index.to(device)

        x_end_childs = self.conv1(x_end_childs, edge_index_end_childs) #+ new_x_end_childs_parents # add the shortcut of previous parents
        # x_end_childs = self.list_conv_fcts[i](x_end_childs, edge_index_end_childs) + new_x_end_childs_parents # add the shortcut of previous parents
        print(f"shape of x_end_childs after convolution: {x_end_childs.shape} \n")        
        x_end_childs = F.relu(x_end_childs)
        x_end_childs = F.dropout(x_end_childs, training=self.training)
        x = self.conv_end(x_end_childs, edge_index_end_childs)

        return F.log_softmax(x, dim=1)

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
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_model, Y_model, test_size=VALID_SIZE, random_state=SEED, stratify=Y_model
    )

    # get the masks corresponding to train/valid/test samples, to keep the train/valid/test indices on graph data while GNN training
    train_mask = create_mask(X=X_model, X_subset=X_train)
    valid_mask = create_mask(X=X_model, X_subset=X_valid)

    return train_mask, valid_mask


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
                train_mask=train_mask, valid_mask=valid_mask)
    
    return data

def get_loader(
    data_total:torch,
    loader_method:str,
    batch_size:int = 150,
    nb_batches:int = 300,
    num_neighbors:int = 30,
    nb_iterations_per_neighbors:int = 2)->torch:
    """Returns the data split in batches of subgraphs for faster GNN training, with the chosen sampling method.

    Args:
        data_total (torch_geometric.data.Data): the whole dataset in a graph 
            Must have the attributes x, edge_index, y, num_classes, num_node_features, train_mask, valid_mask
        loader_method (str): name of the method to build the batch
            Must be set to a value in {'neighbor_nodes', 'index_groups'}
        batch_size (int, optional): number of individuals per batch, s.t. data will be split in nb_total_indivs/batch_size for faster GNN training. Defaults to 32.

        if loader_method == 'neighbor_nodes':
            num_neighbors (int, optional): number of 'similar' nodes to join per batch. Defaults to 30.
            nb_iterations_per_neighbors (int, optional): number of iterations of the loader to find the 'similar' nodes. Defaults to 2.
        
        if loader_method == 'index_groups':
            nb_batches (int, optional): number of splits (i.e. batches) on which the GNN will be trained. Defaults to 4.

    Returns:
        torch_geometric.loader: data split in batches of subgraphs for faster GNN training
    """

    # first of all, check if data_total entails all the attributes to constitute the batches 
    check_attributes_graph_data(data_total)

    t_loader_1 = time.time()

    print(f"\n Construction of the loader with {nb_batches} batches of {batch_size} individuals and the method {loader_method}") 

    if loader_method == 'index_groups':
        loader = IndexLoader(
            data_total=data_total,
            nb_batches=nb_batches,
        )

    elif loader_method == 'neighbor_nodes':
        loader = NeighborLoader(
            data_total,
            # E.g. Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[num_neighbors] * nb_iterations_per_neighbors,
            # E.g. Use a batch size of 128 for sampling training nodes
            batch_size=batch_size,
            input_nodes=data_total.train_mask,
        )
    
    else:
        raise NotImplementedError("The way you want to build batches to train the GCN is not implemented."
                                  "Must be set to a value in {'neighbor_nodes, 'index_groups'}")

    t_loader_2 = time.time()            

    print(f"\n Loading the data in batches of {batch_size} individuals took {round((t_loader_2 - t_loader_1)/60)} mn")

    return loader

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

    return 

def check_batch_info(data_total:torch, batch_size:int, nb_batches:int)->tuple:
    # check that valid batch information is passed, and complete the information on batches (size of splits, number of individual per splits)
    if batch_size is not None:
        nb_indivs_total = data_total.x.shape[0]        
        batch_size = int(nb_indivs_total*batch_size)        

    elif nb_batches is not None:
        nb_indivs_total = data_total.x.shape[0]        
        batch_size = int(nb_indivs_total/nb_batches)

    else:
        raise NotImplementedError("For GNN training on large data, you must specify either the size of splits (batch_size) or the number of individuals per split (nb_batches)")

    return batch_size, nb_batches

def train_GNN(
    data_total:torch,
    loader_method:str,
    batch_size:int = 32,
    epoch_nb:int = 5,
    learning_rate:float = 0.001,
    num_neighbors:int = 30,
    nb_iterations_per_neighbors:int = 2)->GCN:
    """Train a basic GNN over data_total (which must contain a train mask and a valid mask), with the chosen batch method.
    The batch (loader) method enables to split the data in batches of subgraphs, for faster GNN training.

    Args:
        data_total (torch_geometric.data.Data): the whole dataset in a graph 
            Must have the attributes x, edge_index, y, num_classes, num_node_features, train_mask, valid_mask
        loader_method (str): name of the method to build the batch
            Must be set to a value in {'neighbor_nodes'}
        batch_size (int, optional): number of individuals per batch, s.t. data will be split in nb_total_indivs/batch_size for faster GNN training. Defaults to 32.
        epoch_nb (int, optional): number of times the model gets trained on train_data. Defaults to 5.
        learning_rate (float, optional): rapidity of the gradient descent. Defaults to 0.001.

        if loader_method == 'neighbor_nodes':
            num_neighbors (int, optional): number of 'similar' nodes to join per batch. Defaults to 30.
            nb_iterations_per_neighbors (int, optional): number of iterations of the loader to find the 'similar' nodes. Defaults to 2.

    Returns:
        GCN: model of our class GCN(torch.nn.Module), trained on the train set of data_total
    """

    # first of all, check if data_total entails all the attributes for our GNN batch training
    check_attributes_graph_data(data_total)

    # then, split the data into batches for GNN training - load the data with the chosen batch method
    loader = get_loader(data_total=data_total, 
                        loader_method=loader_method,
                        batch_size=batch_size,
                        num_neighbors=num_neighbors,
                        nb_iterations_per_neighbors=nb_iterations_per_neighbors)

    t_basic_1 = time.time()

    # activate and signal the use of GPU for faster processing
    device = activate_gpu()
    # initialize the structure of the classifier, and prepare for GNN training (with GPU)
    classifier = GCN(data_total).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()

    print('\n Training Start \n')
    classifier.train()

    for epoch in range(epoch_nb):
        
        epoch_loss_train = 0
        epoch_loss_valid = 0
        total = 0
        correct = 0
        classifier.train()
        for i, data in enumerate(loader): # try to pass only data (and device to compute in the adapted format)
            optimizer.zero_grad()
            data = data.to(device)
            target = data.y.to(device)
            preds = classifier(data=data, device=device)
            # compute train loss
            error_train = loss(preds[data.train_mask], target[data.train_mask])
            epoch_loss_train = epoch_loss_train + error_train.item()
            # compute valid loss
            error_valid = loss(preds[data.valid_mask], target[data.valid_mask])
            epoch_loss_valid = epoch_loss_valid + error_valid.item()
            # compute overall train&valid accuracy
            _, preds_temp = torch.max(preds.data, 1)
            total = total + len(target)
            correct = correct + (preds_temp == target).sum().item()
            # retropropagate train loss
            error_train.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss_train = {round(epoch_loss_train/(i+1),2)} Loss_valid = {round(epoch_loss_valid/(i+1),2)}"
              f" Train & Valid Accuracy = {round(correct / total, 2)}") 

        # TODO plot the scores across the epochs 

    t_basic_2 = time.time()            

    print(f"Training of the basic GCN on Census on {data_total.x.shape[0]} nodes and {data_total.edge_index.shape[1]} edges, \n with {batch_size} batches and {epoch_nb} epochs took {round((t_basic_2 - t_basic_1)/60)} mn")

    return classifier

def train_GNN_ancestor(
    list_data_total:torch,
    loader_method:str,
    batch_size:int = 150,
    nb_batches:int = 300,
    epoch_nb:int = 5,
    learning_rate:float = 0.001,
    num_neighbors:int = 30,
    nb_iterations_per_neighbors:int = 2)->GCN:
    """Train a GNN adapted to progressive integration of causal descendants, i.e. new graph-data through GCN_ancestor() layers

    I foresee 2 methods: 
        (1) get a loader_i for all graph-data -> split batches into successive loaders (e.g. loader_parent: batches 1-15, loader_child: batches 15-30...)
            Each time, get the 15 "best" batches (couples of neighbors)? May lead to promising results (as not constituted with the same edges)
            Then, pass them to "normal" GCN
        (2) OR iterate over several loaders -> pass them into GNC_ancestor... But is it possible to iterate over a n-uplet of loaders? To be, or not to be...

    Here, we try the method (2)...
        
    TODO doc if it works 
    """
    # first, check that data are passed, either already splitted in batches (list_loader) or only in full-size graph format (list_data_total)
    # and complete the missing information (size of the batches / nb of batches)
    batch_size, nb_batches = check_batch_info(data_total=list_data_total[0], batch_size=batch_size, nb_batches=nb_batches)

    list_loader = []
    for data_total in list_data_total:

        # check if data_total entails all the attributes for our GNN batch training
        check_attributes_graph_data(data_total)

        # then, split the data into batches for GNN training - load the data with the chosen batch method
        loader = get_loader(data_total=data_total, 
                            loader_method=loader_method,
                            batch_size=batch_size,
                            num_neighbors=num_neighbors,
                            nb_iterations_per_neighbors=nb_iterations_per_neighbors,
                            nb_batches=nb_batches)
        list_loader.append(loader)

    t_basic_1 = time.time()

    # activate and signal the use of GPU for faster processing
    device = activate_gpu()
    # initialize the structure of the classifier, and prepare for GNN training (with GPU)
    classifier = GCN_ancestor(list_data_total).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()

    print("\n'''Training Start '''\n")
    classifier.train()

    for epoch in range(epoch_nb):
        
        epoch_loss_train = 0
        epoch_loss_valid = 0
        total = 0
        correct = 0
        classifier.train()

        i=0
        for batch_parent, batch_child in zip(list_loader[0], list_loader[1]): # TODO handle if the number of data-graph (i.e. len(list_loader) is not known)
            print(f"Training Batch {i+1} \n")
            list_data = [batch_parent, batch_child]
            optimizer.zero_grad()
            # data = data.to(device) TODO delete as useless? See after GNN training...
            preds = classifier(list_data=list_data, device=device)
            # compute train loss
            # (!) TODO think about the training target: 
            # as the last layer is evaluated on the last batch (here, batch_child)
            # and batch_child had the more complete set of information (nb_features, causal messages), we choose its targets and masks for error measurement
            target = batch_child.y.to(device)
            error_train = loss(preds[batch_child.train_mask], target[batch_child.train_mask])
            epoch_loss_train = epoch_loss_train + error_train.item()
            epoch_loss_train = epoch_loss_train + error_train.item()
            # compute valid loss
            error_valid = loss(preds[batch_child.valid_mask], target[batch_child.valid_mask])
            epoch_loss_valid = epoch_loss_valid + error_valid.item()
            # compute overall train&valid accuracy
            _, preds_temp = torch.max(preds.data, 1)
            total = total + len(target)
            correct = correct + (preds_temp == target).sum().item()
            # retropropagate train loss
            error_train.backward()
            optimizer.step()
            i=i+1

        print(f"||| Epoch {epoch + 1} Loss_train = {round(epoch_loss_train/(i+1),2)} Loss_valid = {round(epoch_loss_valid/(i+1),2)}"
              f" Train & Valid Accuracy = {round(correct / total, 2)}\n") 

        # TODO plot the scores across the epochs 

    t_basic_2 = time.time()  

    print(f"Training of the basic GCN on Census on {data_total.x.shape[0]} nodes and {data_total.edge_index.shape[1]} edges, \n with {nb_batches} batches each of {batch_size} individuals and {epoch_nb} epochs took {round((t_basic_2 - t_basic_1)/60)} mn")

    return classifier

def evaluate_gnn(classifier:GCN, data_test:torch, loss_name:str="cross_entropy"):
    """Computes the loss and accuracy of a given GCN classifier on test data

    Args:
        classifier (GCN): already trained GCN classifier (!) must not have been trained on data_test
        data_test (torch): new graph-data to evaluate the GCN
            Must have the same attributes (feature nodes, edges...) than data used for GNN training.
        loss_name (str, optional): type of loss the user wants to compute on test data. Defaults to "cross_entropy".
    """

    # first of all, check if data_test shares the attributes of the data-graph (data_total) used for our GNN batch training
    check_attributes_graph_data(data_test)

    # set the loss (for the moment, cross_entropy by default) TODO extend to other losses 
    if loss_name == "cross_entropy":
        print("Loss is measured through cross entropy")
        loss = torch.nn.CrossEntropyLoss()

    device = activate_gpu()

    #classifier = classifier.to(device)
    classifier.eval()

    data_test = data_test.to(device)
    target = data_test.y.to(device)

    preds = classifier(data=data_test, device=device)

    error_test = loss(preds, target)
    print(f"\nError on test: {error_test:.4f} \n")

    # compute overall train&valid accuracy
    _, preds_temp = torch.max(preds.data, 1)
    total = len(target)
    correct = (preds_temp == target).sum().item()
    print(f"Test Accuracy = {round(correct / total, 2)}") 

    # TODO compute and plot ROC-AUC

    return

def index_loader(
    X:pd.DataFrame, 
    Y:pd.DataFrame, 
    list_col_names:list, 
    edges: np.array=None,
    nb_batches:int = 4):
    """Manually split large data in the desired number of batches, splitting the individuals by index order.
    2 advantages vs other loaders:
        - the individuals contained in the batches are easy to trace back
        Ex: batch1 (indivs 1, .., 99); batch 2 (indivs 100, .., 199)

        - when iterating on parallel batches, it ensures the **same individuals** are handled by the network
        => per batch only causal information changes (through different directed edges per layers)
        => during training, the GNN adds information on these same individuals
        Ex: 

        list_loader[0] = index_loader(graph_data0) -> with edge "age -> job"
        list_loader[1] = index_loader(graph_data1) -> with edge "job" -> "work hours per week"

        for batch_parent, batch_child in zip(list_loader[0], list_loader[1]):
            TRAINING PER BATCH
            & batch_parent, batch_child contains the **same individuals**
    
    (!) Here, our approach starts with the DataFrame => individuals are joined by index, not by simplicity 
    TODO shuffle index before table_to_graph()?

    """

    list_batches = []

    # reconstitute the DataFrame, to pass the target (and enable shuffling) during data-graph construction per batch
    df = X.copy()
    df["target"] = Y

    nb_indivs_total = X.shape[0]         
    batch_size = int(nb_indivs_total/nb_batches)

    list_df_split = [df.iloc[n:n+batch_size] for n in range(nb_batches)]

    for i, df_group_index in enumerate(list_df_split):
        print(f"Loading batch {i}")# grouping individuals {df_group_index.index}")
        X_group_index = df_group_index.drop("target", axis=1)
        Y_group_index = df_group_index["target"]

        batch = table_to_graph(X=X_group_index, Y=Y_group_index, list_col_names=list_col_names, edges=edges)
        # TODO then load the batch for faster computation?
        # batch = DataLoader(batch)

        list_batches.append(batch)
    
    loader = list_batches # TODO set in an adapted format for iteration during GNN training? Is it not already the case? 

    return loader
