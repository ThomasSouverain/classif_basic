import time 

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from classif_basic.graph.data_to_graph import table_to_graph
from classif_basic.graph.utils import check_attributes_graph_data

class IndexLoader(torch.utils.data.DataLoader):
    """A data loader that samples nodes within a graph following the order of their indexes, and returns
    their induced subgraph. This allows knowledge augmentation through multiple graphs of the same data -> see GCN_ancestor() in classif_basic

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
    
    (!) For the moment, individuals are joined by index, not by proximity of edges or node attributes TODO?

    (!) We assume here the presence of only homogeneous data:
    In our use case indeed, only 1 table with clients and their feature is used to (i) train the GNN (ii) while respecting minimal causal paths

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        nb_batches (int): The number of batches (i.e. subgraphs) the user wants to obtain. 
            If a higher number of batches can improve the GNN performance on train_set, 
            it can also lead to poor performance on valid data (overfitting) or slow down computation. 
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(
        self,
        data_total: Data,
        nb_batches: int,
        **kwargs,
    ):
        self.data_total = data_total
        self.nb_batches = nb_batches

        self.num_nodes = self.data_total.x.shape[0]        
        self.batch_size = int(self.num_nodes/self.nb_batches)

        self.list_subgraphs = self.get_successive_node_indices()

    def get_successive_node_indices(self)->list:
        """Get a list with of subgraphs, split in 'nb_batches' following the order of their indexes

        Returns:
            list(data): list of subgraphs for GNN iteration
        """
        self.list_subgraphs = []

        for i in range(self.nb_batches):
            batch_index = torch.tensor(range(i*self.batch_size, i*self.batch_size+self.batch_size))
            
            subgraph = self.data_total.subgraph(batch_index)
            
            self.list_subgraphs.append(subgraph)

        return self.list_subgraphs

    def __iter__(self):
        """Enable iteration on the list of subgraphs, for GNN training by epochs
        """
        self.list_subgraphs = self.get_successive_node_indices()
        return iter(self.list_subgraphs)


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


def index_loader(
    X:pd.DataFrame, 
    Y:pd.DataFrame, 
    list_edges_names:list, 
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
    
    (!) Here, our approach starts with the DataFrame => individuals are joined by index, not by proximity of edges or node attributes 
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

        batch = table_to_graph(X=X_group_index, Y=Y_group_index, list_edges_names=list_edges_names, edges=edges)
        # TODO then load the batch for faster computation?
        # batch = DataLoader(batch)

        list_batches.append(batch)
    
    loader = list_batches # TODO set in an adapted format for iteration during GNN training? Is it not already the case? 

    return loader
