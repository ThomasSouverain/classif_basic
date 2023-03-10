import torch


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
