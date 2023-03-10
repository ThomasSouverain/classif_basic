import torch
from torch_geometric.data import Data

class IndexLoader(torch.utils.data.DataLoader):
    """A data loader that samples nodes within a graph following the order of their indexes, and returns
    their induced subgraph.

    Using IndexLoader, (n) graph-data with the same nodes (in the same order) will always have the same nodes in their subgraphs.
    This allows knowledge augmentation through multiple graphs of the same data -> see GCN_ancestor() in classif_basic

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
