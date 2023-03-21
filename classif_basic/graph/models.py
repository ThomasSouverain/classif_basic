import torch 
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import LSTMAggregation
from torch_geometric.nn import Sequential
from torch_geometric.nn.conv import GATConv

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

    def forward(self, list_data, device, skip_connection): 

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
            # print(f"\n Layer with causal ascendance {i+1} : ")
            data = data.to(device)
            x = data.x.float().to(device)
            edge_index=data.edge_index.to(device)

            # print(f"shape of x before convolution: {x.shape} ; ") 
            # print(f"shape of new_x_parents before convolution: {new_x_parents.shape} ; ") 

            x = self.conv1(x, edge_index) + new_x_parents # TODO add the shortcut of previous parents
            # x = self.list_conv_fcts[i](x, edge_index) + new_x_parents # add the shortcut of previous parents
            # print(f"shape of x after convolution: {x.shape} \n")        
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            # TODO below -> add the child(n) as parent of child (n+1) for further shortcuts
            if skip_connection==True: # else stays to 0
                new_x_parents = new_x_parents + x

        # print(f"Last Child Layer : ")
        # last convolutional layer with the last child data (n) (i.e. the last data_end_childs registered)
        data_end_childs = list_data[-1]

        x_end_childs = data_end_childs.x.float().to(device)
        edge_index_end_childs = data_end_childs.edge_index.to(device)

        x_end_childs = self.conv1(x_end_childs, edge_index_end_childs) #+ new_x_end_childs_parents # add the shortcut of previous parents
        # x_end_childs = self.list_conv_fcts[i](x_end_childs, edge_index_end_childs) + new_x_end_childs_parents # add the shortcut of previous parents
        # print(f"shape of x_end_childs after convolution: {x_end_childs.shape} \n")        
        x_end_childs = F.relu(x_end_childs)
        x_end_childs = F.dropout(x_end_childs, training=self.training)
        x = self.conv_end(x_end_childs, edge_index_end_childs)

        return F.log_softmax(x, dim=1)
    
class GCN_ancestor_sequential(torch.nn.Module):
    def __init__(self, list_data):
        super().__init__()

        data = list_data[-1] # build the structure of the Sequential GNN with any data-graph, as they all share the same shape (node features == ancestors)

        self.seq_model = Sequential('x, edge_index', [
            (Dropout(p=0.5), 'x -> x'),
            (GCNConv(data.num_node_features, 64), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x1, edge_index -> x2'),
            ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            Linear(2 * 64, data.num_classes),
            ReLU(inplace=True),
        ])

        self.seq_1 = Sequential('x, edge_index', [ # TODO only replace by seq_1 and seq_2
            (Dropout(p=0.5), 'x -> x'),
            (GCNConv(data.num_node_features, 16), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (GCNConv(16, 16), 'x1, edge_index -> x2'),
            ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (JumpingKnowledge("cat", 16), 'xs -> x3'),
            ReLU(inplace=True),
            (Dropout(p=0.5), 'x3 -> x'),
        ])

        self.seq_2 = Sequential('x, edge_index', [
            (Dropout(p=0.5), 'x -> x'),
            (GCNConv(data.num_node_features, 16), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (GCNConv(16, 16), 'x1, edge_index -> x2'),
            ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (JumpingKnowledge("cat", 16, num_layers=2), 'xs -> x3'),
            ReLU(inplace=True),
            (Dropout(p=0.5), 'x3 -> x3'),
            (GCNConv(16*2, data.num_classes), 'x3, edge_index -> x'),
            (lambda x: F.log_softmax(x, dim=1))
        ])

        self.conv_end = GCNConv(16, data.num_classes)


    def forward(self, list_data, device, skip_connection): 
        # new_x_parents in case of skip connection <=> keep ancestors' values 
        list_classif_outputs = []
        
        new_x_parents = 0
        for i, data in enumerate(list_data): # loop for the (n) first ancestor data
            data = data.to(device)
            x = data.x.float().to(device)
            edge_index=data.edge_index.to(device)

            list_classif_outputs[i] = self.seq_model(x=x, edge_index=edge_index) + new_x_parents # TODO seq_1

            # TODO below -> add the child(n) as parent of child (n+1) for further shortcuts
            if skip_connection==True: # else stays to 0
                new_x_parents = new_x_parents + x
        
        # TODO the len(list_data) will become inputs of an Aggregation layer => join them
        
        # return binary result (for classif) in last child layer
        data_end_childs = list_data[-1]

        x_end_childs = data_end_childs.x.float().to(device)
        edge_index_end_childs = data_end_childs.edge_index.to(device)

        x_end_childs = self.seq_model(x_end_childs, edge_index_end_childs) # TODO seq_2

        return x

class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2*out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT_ancestor(torch.nn.Module):
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
        """Dense version of GAT."""
        super().__init__()
        data = list_data[0] # as here the data.x have the same dim (only ancestors), the GAT layers have all the same shape

        nfeat=data.num_nodes
        nhid=3 # nb of hidden layers? TODO verify
        nclass=data.num_classes
        dropout=0.3 # drop 30% of irrelevant features? TODO verify
        alpha=0.01
        nheads=2

        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, list_data, device, skip_connection): 
        data = list_data[0] # for the moment, implemented only for one data graph TODO etxend if it works 

        data = data.to(device)
        x = data.x.float().to(device)
        edge_index=data.edge_index.to(device)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.log_softmax(x, dim=1)

class GAT_conv_ancestor(torch.nn.Module):
    def __init__(self, list_data, hidden_size=2):
        super().__init__()
        data = list_data[0] # for the moment, implemented only for one data graph TODO etxend if it works 

        self.hidden_size = hidden_size
        self.num_features = data.num_features # 3 in toy example
        self.target_size = 2 # data.num_classes # 1 in toy example
        # self.convs = [GATConv(self.num_features, self.hidden_size)]
        self.convs = [GATConv(self.num_features, self.hidden_size),
                      GATConv(self.hidden_size, self.hidden_size)]
        self.linear = torch.nn.Linear(self.hidden_size, self.target_size)

        # conv = GATConv(data.num_features, 32)
        # conv(data.x.float(), data.edge_index)
        
    def forward(self, list_data, device, skip_connection=False):
        data = list_data[0].cpu()
        x = data.x.float().cpu()
        edge_index=data.edge_index.cpu()
        #edge_attr = data.edge_attr#.to(device) if data.edge_attr is not None else None # edge_attr not used for the moment? TODO combine features?
        x=self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index) 
        # print("forward before linear!")
        x = self.linear(x.to(device)) # TODO improve use of the device (initialisation & forward -> test other attention matrices, other weights?)
        # print("forward after linear!")
        return F.relu(x) # since we know Y = log_gdp > 0, enforce via relu        

        # for conv in self.convs[:-1]: 
        #     print("conv!\n")
        #     x = conv(x, edge_index)#.to(device)
        #     x = F.relu(x)
        #     x = F.dropout(x, training=self.training)
        # x = self.convs[-1](x, edge_index) 
        # x = self.linear(x)
        # return F.relu(x) # since we know Y = log_gdp > 0, enforce via relu

    # # TODO doc if it works 
    # def __init__(self, list_data):
    #     """Dense version of GAT."""
    #     super().__init__()
    #     data = list_data[0] # as here the data.x have the same dim (only ancestors), the GAT layers have all the same shape

    #     nfeat=data.num_nodes
    #     nhid=3 # nb of hidden layers? TODO verify
    #     nclass=data.num_classes
    #     dropout=0.3 # drop 30% of irrelevant features? TODO verify
    #     alpha=0.01
    #     nheads=2

    #     self.dropout = dropout

    #     self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
    #     for i, attention in enumerate(self.attentions):
    #         self.add_module('attention_{}'.format(i), attention)

    #     self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # def forward(self, list_data, device, skip_connection): 
    #     data = list_data[0] # for the moment, implemented only for one data graph TODO etxend if it works 

    #     data = data.to(device)
    #     x = data.x.float().to(device)
    #     edge_index=data.edge_index.to(device)

    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, edge_index))
    #     return F.log_softmax(x, dim=1)
