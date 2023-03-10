import torch 
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential

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
