import itertools
import time

import numpy as np
import torch
from sklearn.utils import class_weight

from classif_basic.graph.evaluate import get_auc
from classif_basic.graph.evaluate import get_loss
from classif_basic.graph.evaluate import plot_metrics
from classif_basic.graph.loader import get_loader
from classif_basic.graph.models import GCN_ancestor
from classif_basic.graph.utils import check_attributes_graph_data
from classif_basic.graph.utils import check_batch_info

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

def train_GNN_ancestor(
    list_data_total:torch,
    loader_method:str,
    loss_name:str="CrossEntropyLoss",
    batch_size:int = None,#150,
    nb_batches:int = None, #300,
    epoch_nb:int = 5,
    learning_rate:float = 0.001,
    num_neighbors:int = 30,
    nb_iterations_per_neighbors:int = 2,
    plot_loss:bool=True,
    plot_auc:bool=True,
    unique_data_graph:bool=False,
    skip_connection:bool=False, # to overweight ancestors conv layers
    )->GCN_ancestor:
    """Train a GNN adapted to progressive integration of causal descendants, i.e. new graph-data through GCN_ancestor() layers

    TODO doc if it works -> Train a basic GNN over data_total (which must contain a train mask and a valid mask), with the chosen batch method.
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
    
    I foresee 2 methods: 
        (1) get a loader_i for all graph-data -> split batches into successive loaders (e.g. loader_parent: batches 1-15, loader_child: batches 15-30...)
            Each time, get the 15 "best" batches (couples of neighbors)? May lead to promising results (as not constituted with the same edges)
            Then, pass them to "normal" GCN
        (2) OR iterate over several loaders -> pass them into GNC_ancestor... But is it possible to iterate over a n-uplet of loaders? To be, or not to be...

    Here, we try the method (2)...
        
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
    #loss=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    print("\n'''Training Start '''\n")
    classifier.train()

    # get errors in list -> plot them across epochs
    list_train_epoch_errors = []
    list_valid_epoch_errors = []

    list_train_epoch_roc = []
    list_valid_epoch_roc = []

    list_train_epoch_pr = []
    list_valid_epoch_pr = []

    list_train_epoch_fpr_ratio = []
    list_valid_epoch_fpr_ratio = []

    list_train_epoch_tpr_ratio = []
    list_valid_epoch_tpr_ratio = []

    list_train_epoch_accuracy = []
    list_valid_epoch_accuracy = []

    for epoch in range(epoch_nb):
        
        epoch_metrics_train = 0
        epoch_metrics_valid = 0

        epoch_roc_train = 0
        epoch_roc_valid = 0

        epoch_pr_train = 0
        epoch_pr_valid = 0

        epoch_fpr_ratio_train = 0
        epoch_fpr_ratio_valid = 0

        epoch_tpr_ratio_train = 0
        epoch_tpr_ratio_valid = 0

        total_train = 0
        correct_train = 0
        total_valid = 0
        correct_valid = 0

        classifier.train()

        i=0

        # TODO set in a function get_list_loader()
        if unique_data_graph==True:
            list_full_data_graphs = list_loader
        elif unique_data_graph==False: # by default, when multiple data-graphs (=> directed edges, i.e. causal paths) are passed
            list_full_data_graphs = itertools.zip_longest(*tuple(list_loader))

        # for batch in list_loader: # TODO delete -> temporary to train a 1-graph "classic" GCN on all features for comparison
        #     print(f"Training Batch {i+1} of unique graph\n")
        #     list_data = list(batch)
        for batch in list_full_data_graphs:#itertools.zip_longest(*tuple(list_loader)): 
            # print(f"Training Batch {i+1} ")
            list_data = list(batch)
            # (!) TODO think about the training target: 
            # as the last layer is evaluated on the last batch (here, batch_child)
            # and batch_child had the more complete set of information (nb_features, causal messages), we choose its targets and masks for error measurement
            batch_child = list_data[-1]
            target = batch_child.y.to(device)
            target_train = target[batch_child.train_mask]
            target_valid = target[batch_child.valid_mask]
            optimizer.zero_grad()
            preds = classifier(list_data=list_data, device=device, skip_connection=skip_connection)
            probas_pred_train = preds[batch_child.train_mask]
            probas_pred_valid = preds[batch_child.valid_mask]
            # instantiates a balanced loss (depending on classes imbalance inside the batch)
            class_weights=class_weight.compute_class_weight(class_weight='balanced',
                                                            classes=np.unique(target.cpu()),
                                                            y=target.cpu().numpy())
            class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
            # compute train metrics
            error_train = loss(probas_pred_train, target_train)
            epoch_metrics_train = epoch_metrics_train + error_train.item()
            # compute valid metrics
            error_valid = loss(probas_pred_valid, target_valid)
            epoch_metrics_valid = epoch_metrics_valid + error_valid.item()
            # compute (and aggregate over batches) ROC and PR AUC -> TODO? function to get all measurements per batch (loss, pr, roc)
            # add here (temporary TODO delete?) false and true positive ratios
            roc_auc_train, pr_auc_train, fpr_ratio_train, tpr_ratio_train = get_auc(
                                    probas_pred=probas_pred_train, 
                                    y_true=target_train)
            roc_auc_valid, pr_auc_valid, fpr_ratio_valid, tpr_ratio_valid = get_auc(
                                    probas_pred=probas_pred_valid, 
                                    y_true=target_valid)
            epoch_roc_train = epoch_roc_train + roc_auc_train.item()
            epoch_roc_valid = epoch_roc_valid + roc_auc_valid.item()
            epoch_pr_train = epoch_pr_train + pr_auc_train.item()
            epoch_pr_valid = epoch_pr_valid + pr_auc_valid.item()
            epoch_fpr_ratio_train = epoch_fpr_ratio_train + fpr_ratio_train.item()
            epoch_fpr_ratio_valid = epoch_fpr_ratio_valid + fpr_ratio_valid.item()
            epoch_tpr_ratio_train = epoch_tpr_ratio_train + tpr_ratio_train.item()
            epoch_tpr_ratio_valid = epoch_tpr_ratio_valid + tpr_ratio_valid.item()
            # compute train&valid accuracy TODO compute accuracy in get_accuracy()
            _, preds_temp_train = torch.max(probas_pred_train.data, 1)
            total_train = total_train + len(target_train)
            correct_train = correct_train + (preds_temp_train == target_train).sum().item()

            _, preds_temp_valid = torch.max(probas_pred_valid.data, 1)
            total_valid = total_valid + len(target_valid)
            correct_valid = correct_valid + (preds_temp_valid == target_valid).sum().item()
            # integrate false positives (inverse because <1) in loss (coeff)
            error_train = error_train/epoch_fpr_ratio_train
            error_train.backward()
            optimizer.step()
            i=i+1

        error_metrics_train = round(epoch_metrics_train/(i+1),2)
        error_metrics_valid = round(epoch_metrics_valid/(i+1),2)

        list_train_epoch_errors.append(error_metrics_train)
        list_valid_epoch_errors.append(error_metrics_valid)

        # TODO definitely a function to not repeat the codes (measurements per batch)

        roc_train = round(epoch_roc_train/(i+1),2)
        roc_valid = round(epoch_roc_valid/(i+1),2)

        list_train_epoch_roc.append(roc_train)
        list_valid_epoch_roc.append(roc_valid)

        pr_train = round(epoch_pr_train/(i+1),2)
        pr_valid = round(epoch_pr_valid/(i+1),2)

        list_train_epoch_pr.append(pr_train)
        list_valid_epoch_pr.append(pr_valid)

        list_train_epoch_fpr_ratio.append(fpr_ratio_train)
        list_valid_epoch_fpr_ratio.append(fpr_ratio_valid)

        list_train_epoch_tpr_ratio.append(tpr_ratio_train)
        list_valid_epoch_tpr_ratio.append(tpr_ratio_valid)

        accuracy_train = round(correct_train / total_train, 2)
        accuracy_valid = round(correct_valid / total_valid, 2)

        list_train_epoch_accuracy.append(accuracy_train)
        list_valid_epoch_accuracy.append(accuracy_valid)

        print(f"||| Epoch {epoch + 1} {loss_name}_train = {error_metrics_train} {loss_name}_valid = {error_metrics_valid}")
            #   f"\n accuracy_train: {accuracy_train} accuracy_valid: {accuracy_valid} \n" 
            #   f"\n roc_auc_train: {roc_train} roc_valid: {roc_valid} \n"
            #   f"\n pr_auc_train: {pr_train} pr_valid: {pr_valid} \n")

        # TODO when more stable models (see below) cross validation
        # min_train_epoch_error = min(min_train_epoch_error, error_metrics_train)
        # min_valid_epoch_error = min(min_valid_epoch_error, error_metrics_valid)

        # if min_train_epoch_error < error_metrics_train and min_valid_epoch_error < error_metrics_valid:
        #     print("Training improves no more across epochs")                                    
        #     print("Last Epoch Classifier on last batch")
        #     get_auc(probas_pred=probas_pred_train, 
        #             y_true=target_train,
        #             plot=True)
        #     break 

    t_basic_2 = time.time()  

    print(f"Training of the basic GCN on Census on {data_total.x.shape[0]} nodes and {data_total.edge_index.shape[1]} edges, \n with {nb_batches} batches each of {batch_size} individuals and {epoch_nb} epochs took {round((t_basic_2 - t_basic_1)/60)} mn")

    plot_metrics(metrics_name=loss_name, epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_errors, 
                list_valid_epoch_errors=list_valid_epoch_errors)

    plot_metrics(metrics_name="Accuracy", epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_accuracy, 
                list_valid_epoch_errors=list_valid_epoch_accuracy)

    plot_metrics(metrics_name="ROC AUC", epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_roc, 
                list_valid_epoch_errors=list_valid_epoch_roc)

    plot_metrics(metrics_name="PR AUC", epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_pr, 
                list_valid_epoch_errors=list_valid_epoch_pr)
    
    plot_metrics(metrics_name="False Positive Rate", epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_fpr_ratio, 
                list_valid_epoch_errors=list_valid_epoch_fpr_ratio)

    plot_metrics(metrics_name="True Positive Rate", epoch_nb=epoch_nb, 
                list_train_epoch_errors=list_train_epoch_tpr_ratio, 
                list_valid_epoch_errors=list_valid_epoch_tpr_ratio)

    # TODO set in a separate module the visualisation, as computing of metrics (utils.py?)

    return classifier
