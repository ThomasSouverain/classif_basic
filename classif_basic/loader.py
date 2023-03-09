import torch
from torch_geometric.data import Data
from torch_geometric.sampler import BaseSampler

class IndexLoader(torch.utils.data.DataLoader):#O inherit from BaseSampler?
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.
    .. note::
        For an example of using
        :class:`~torch_geometric.loader.RandomNodeLoader`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.
    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    (!) We assume here the presence of only homogeneous data (for our use case with 1 table)
    """
    def __init__(
        self,
        data_total: Data,
        nb_batches: int,
        persistent_workers:int,
        num_workers:int,
        **kwargs,
    ):
        self.data_total = data_total
        self.nb_batches = nb_batches

        self.num_nodes = self.data_total.x.shape[0]        
        self.batch_size = int(self.num_nodes/self.nb_batches)

        #self._iterator = self._get_iterator()

        # add required arguments for iteration on PyTorch loaders
        self.persistent_workers=persistent_workers
        self.num_workers=num_workers

        self._iterator = Iterator()#self._get_iterator()
    
        # super().__init__(
        #     range(self.num_nodes), # TODO what is it?
        #     batch_size=self.batch_size,
        #     collate_fn=self.collate_fn,
        #     **kwargs,
        # )

    def get_successive_node_indices(self):
        self.list_subgraphs = []

        for i in range(self.nb_batches):
            batch_index = torch.tensor(range(i*self.batch_size, i*self.batch_size+self.batch_size))
            
            subgraph = self.data_total.subgraph(batch_index)
            
            self.list_subgraphs.append(subgraph)

        return self.list_subgraphs

    def _get_iterator(self):
        self.list_subgraphs = self.get_successive_node_indices()
        return self.list_subgraphs

    def collate_fn(self, index):
        return self.data_total.subgraph(index)

class Iterator(IndexLoader):

    def _get_iterator(self):
        self.list_subgraphs = self.get_successive_node_indices()
        return self.list_subgraphs

    def _reset(self):
        return self.list_subgraphs
