import itertools
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
from xgboost import XGBClassifier

from classif_basic.graph.evaluate import compute_metric
from classif_basic.graph.evaluate import get_auc
from classif_basic.graph.evaluate import get_loss
from classif_basic.graph.loader import get_loader
from classif_basic.graph.models import GAT_ancestor
from classif_basic.graph.models import GAT_conv_ancestor
from classif_basic.graph.models import GCN_ancestor
from classif_basic.graph.models import GCN_ancestor_sequential
from classif_basic.graph.plot import plot_metric_epochs
from classif_basic.graph.plot import plot_perfs_gnn
from classif_basic.graph.utils import activate_gpu
from classif_basic.graph.utils import check_attributes_graph_data
from classif_basic.graph.utils import check_batch_info
from classif_basic.model import compute_best_fscore

def train_xgb_benchmark(X_train: pd.DataFrame, Y_train:pd.DataFrame, X_valid: pd.DataFrame, Y_valid:pd.DataFrame)->tuple:
    """Given DataFrames X and Y, performs train/valid/test split 
    and trains a baseline XGB classifier for comparison with other models.

    Args:
        X_train (pd.DataFrame): X of the train set used for other models training
        Y_train (pd.DataFrame): Target of the train set used for other models training
        X_valid (pd.DataFrame): X of the valid set used for other models training
        Y_valid (pd.DataFrame): Target of the valid set used for other models training

    Returns:
        tuple: performance metrics of the XGB on train&valid set
            ROC_AUC, PR_AUC, False Positive Ratio, True Positive Ratio
            Also plot ROC_AUC and PR_AUC
    """

    X_train_valid = X_train.append(X_valid)
    Y_train_valid = Y_train.append(Y_valid)

    SEED = 7
    early_stopping_rounds = 20
    verbose = 100
        
    xgb_classif_params = {
        "seed": SEED,
        "objective": "binary:logistic",
        "n_estimators": 1000,
        "max_depth": 3,
        "importance_type": "gain",
        "use_label_encoder": False,
    }

    eval_metric="auc"

    model = XGBClassifier(**xgb_classif_params)

    model.fit(
        X_train,
        Y_train,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=verbose,
    )

    probas_pred_train_valid = model.predict_proba(X_train_valid)
    probas_pred_valid = model.predict_proba(X_valid)

    ## set y predicted with optimised thresholds, to compute accuracy
    best_threshold, best_fscore = compute_best_fscore(Y_valid, probas_pred_valid[:,1])

    Y_pred_train_valid = (probas_pred_train_valid[:,1] >= best_threshold).astype(int)

    total_target = Y_train_valid.shape[0]
    total_exact=(Y_pred_train_valid==Y_train_valid).sum()#.all()

    xgb_accuracy = total_exact/total_target
    print(f"xgb_accuracy: {xgb_accuracy}")

    # compute AUCs and false / true positive ratios
    roc_auc, pr_auc, fpr_ratio, tpr_ratio = get_auc(y_true=Y_train_valid,
            probas_pred=probas_pred_train_valid,
            plot=True)

    return roc_auc, pr_auc, fpr_ratio, tpr_ratio

def train_GNN_ancestor(
    list_data_total:torch,
    model_type:str, 
    loader_method:str,
    loss_name:str="CrossEntropyLoss",
    batch_size:int = None,
    nb_batches:int = None,
    epoch_nb:int = 1000,
    cv_step:int=100,
    learning_rate:float = 0.001,
    num_neighbors:int = 30,
    nb_iterations_per_neighbors:int = 2,
    skip_connection:bool=False, # to overweight ancestors conv layers
    list_performance_metrics:list=["accuracy", "roc_auc", "pr_auc", "fpr_ratio", "tpr_ratio"],
    )->GCN_ancestor:
    """Train a GNN adapted to progressive integration of causal descendants, i.e. new graph-data through GCN_ancestor() layers
    
    I foresee 2 methods: 
        (1) get a loader_i for all graph-data -> split batches into successive loaders (e.g. loader_parent: batches 1-15, loader_child: batches 15-30...)
            Each time, get the 15 "best" batches (couples of neighbors)? May lead to promising results (as not constituted with the same edges)
            Then, pass them to "normal" GCN
        (2) OR iterate over several loaders -> pass them into GNC_ancestor... But is it possible to iterate over a n-uplet of loaders? To be, or not to be...

    Here, we try the method (2)...

    Args:
        list_data_total (List(torch_geometric.data.Data)): the concatenation of each data-graph, following the order of their causal links (edges)
            Each data-graph must have the attributes x, edge_index, y, num_classes, num_node_features, train_mask, valid_mask
        model_type (str): how to train the GNN to progressively integrate causal information (new data-graph edge = new layer of the GNN)
            Must be set to a value in {'conv', 'sequential', 'attention', 'conv_attention'}
        loader_method (str): how to split data in batches to train the GNN
            Must be set to a value in {'neighbor_nodes', 'index_groups'}
            if loader_method == 'neighbor_nodes':
                num_neighbors (int, optional): number of 'similar' nodes to join per batch. Defaults to 30.
                nb_iterations_per_neighbors (int, optional): number of iterations of the loader to find the 'similar' nodes. Defaults to 2.

        loss_name (str, optional): Measurement of the performance during GNN training, used for gradient descent. Defaults to "CrossEntropyLoss".
        batch_size (int, optional): number of individuals per batch, s.t. data will be split in nb_total_indivs/batch_size for faster GNN training. Defaults to 32.        
        nb_batches (int, optional): number of splits (i.e. batches) on which the GNN will be trained. Defaults to None.
        epoch_nb (int, optional): number of epochs, i.e. backs and forths, the GNN has to train. Defaults to 1000.
        cv_step (int, optional): for cross-validation, after how many epochs to stop training if train/valid loss does not decrease. Defaults to 100.
        learning_rate (float, optional): learning rate of the GNN for gradient descent. Defaults to 0.001.
        skip_connection (bool, optional): whether to stress the importance on past layers (i.e. "ancestor" edge knowledge) in GNN architecture. Defaults to False.
        list_performance_metrics (list, optional): names of the metrics to assess GNN performances
            By default ["accuracy", "roc_auc", "pr_auc", "fpr_ratio", "tpr_ratio"]
            Each element must be set to a value in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy}

        Returns:
            GCN: model of our class GCN(torch.nn.Module), trained on the train set of data_total
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
    print(f"\n GNN of {model_type} type")
    if model_type=='conv':
        classifier = GCN_ancestor(list_data_total).to(device) 
    elif model_type=='sequential':
        classifier = GCN_ancestor_sequential(list_data_total).to(device)
    elif model_type=='attention':
        classifier = GAT_ancestor(list_data_total).to(device) 
    elif model_type=='conv_attention':
        classifier = GAT_conv_ancestor(list_data_total).to(device)
    else:
        raise NotImplementedError("Your ancestor GNN 'model_type' is not implemented. "
                                  "Must be set to a value in {'conv', 'sequential', 'attention', 'conv_attention'}")
    #loss=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    print("\n'''Training Start '''\n")
    classifier.train()

    # get performance metrics in lists -> plot them across epochs
    list_performance_metrics.append(loss_name)
    dict_train_epochs_metrics = {} # dict_train_epochs_metrics[metric_name] == list with metrics across epochs 
    dict_valid_epochs_metrics = {}

    for metric_name in list_performance_metrics:
        dict_train_epochs_metrics[metric_name] = []
        dict_valid_epochs_metrics[metric_name] = []

    # let's start the training by epochs 
    for epoch in range(epoch_nb):

        dict_train_new_epoch_metrics_value = {}
        dict_valid_new_epoch_metrics_value = {}

        for metric_name in list_performance_metrics:
            dict_train_new_epoch_metrics_value[metric_name] = 0
            dict_valid_new_epoch_metrics_value[metric_name] = 0

        classifier.train()

        i=0

        iter_full_data_graphs = itertools.zip_longest(*tuple(list_loader))
        for batch in iter_full_data_graphs: 
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
            probas_pred_train = preds[batch_child.train_mask].to(device) 
            probas_pred_valid = preds[batch_child.valid_mask].to(device) 
            # instantiates a balanced loss (depending on classes imbalance inside the batch)
            dict_class_weights = {0:1, 1:5} 
            class_weights=class_weight.compute_class_weight(class_weight='balanced', # alternative: dict_class_weights for large imbalance?
                                                            classes=np.unique(target.cpu()),
                                                            y=target.cpu().numpy())
            class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')  
            # compute train metrics
            error_train = loss(probas_pred_train, target_train)
            dict_train_new_epoch_metrics_value[loss_name] = dict_train_new_epoch_metrics_value[loss_name] + error_train.item()
            # compute valid metrics
            error_valid = loss(probas_pred_valid, target_valid)
            dict_valid_new_epoch_metrics_value[loss_name] = dict_valid_new_epoch_metrics_value[loss_name] + error_valid.item()

            for metric_name in [metric_name for metric_name in list_performance_metrics if metric_name!=loss_name]: # loss handled apart for gradients
                new_epoch_metrics_train = compute_metric(metric_name, probas_pred_train, target_train)
                new_epoch_metrics_valid = compute_metric(metric_name, probas_pred_valid, target_valid)

                dict_train_new_epoch_metrics_value[metric_name] = dict_train_new_epoch_metrics_value[metric_name] + new_epoch_metrics_train.item() 
                dict_valid_new_epoch_metrics_value[metric_name] = dict_valid_new_epoch_metrics_value[metric_name] + new_epoch_metrics_valid.item() 
            
            # loss handled apart to back-propagate gradients
            error_train.backward()
            optimizer.step()
            i=i+1

        for metric_name in list_performance_metrics:
            metrics_train = round(dict_train_new_epoch_metrics_value[metric_name]/i,2)
            metrics_valid = round(dict_valid_new_epoch_metrics_value[metric_name]/i,2)

            dict_train_epochs_metrics[metric_name].append(metrics_train)
            dict_valid_epochs_metrics[metric_name].append(metrics_valid)

        list_train_losses = dict_train_epochs_metrics[loss_name]
        list_valid_losses = dict_valid_epochs_metrics[loss_name]
        print(f"||| Epoch {epoch + 1} {loss_name}_train = {list_train_losses[-1]} {loss_name}_valid = {list_valid_losses[-1]}")

        # cross-validation
        if len(list_train_losses) > cv_step: # enable initialisation of the losses 
            min_loss_train_registered = min(list_train_losses[:-cv_step])
            min_loss_train_current = min(list_train_losses)
            min_loss_valid_registered = min(list_valid_losses[:-cv_step])
            min_loss_valid_current = min(list_valid_losses)

            if min_loss_train_current >= min_loss_train_registered or min_loss_valid_current >= min_loss_valid_registered:
                print(f"Training no more improved over the past {cv_step} epochs")
                epoch_nb = epoch + 1
                break 

    t_basic_2 = time.time()  

    str_multiple_data_graphs = "unique graph-data" if len(list_data_total)==1 else "multiple graph-data"

    print(f"\n{model_type} GNN model, Loader method {loader_method} on {str_multiple_data_graphs}")
    print(f"Training of the basic GCN on Census on {data_total.x.shape[0]} nodes and {data_total.edge_index.shape[1]} edges, \n with {nb_batches} batches each of {batch_size} individuals and {epoch_nb} epochs took {round((t_basic_2 - t_basic_1)/60)} mn")

    print("\n||| Results of the GNN across epochs |||")

    for metric_name in list_performance_metrics:

        plot_metric_epochs(metric_name=metric_name, 
                    list_train_epoch_metrics=dict_train_epochs_metrics[metric_name], 
                    list_valid_epoch_metrics=dict_valid_epochs_metrics[metric_name])

    print("\n||| Results of the GNN on the last epoch |||")
    # even if it is arbitrary, plot the results on the last batch (small => biased predictions...) -> how TODO else? 
    plot_perfs_gnn(classifier=classifier,
                 list_data_test=list_data) # as it is the last => child list_data in memory

    return classifier
