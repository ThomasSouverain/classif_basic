from typing import List
from typing import Union

import numpy as np
import torch
from matplotlib.pylab import plt

from classif_basic.graph.evaluate import compute_metric
from classif_basic.graph.utils import activate_gpu


def plot_metric_epochs(metric_name:str, list_train_epoch_metrics:list, list_valid_epoch_metrics:list):
    """Plots the performances for one metric in train and valid sets, across epochs.

    Args:
        metric_name (str): name of the performance metrics to be computed 
        list_train_epoch_metrics (list): list with the metrics values for all epoch (train set)
        list_valid_epoch_metrics (list): list with the metrics values for all epoch (valid set)
    """
    epoch_nb = len(list_train_epoch_metrics)
    epochs=range(epoch_nb)
    # Plot and label the training and validation loss values
    plt.plot(epochs, list_train_epoch_metrics, label=f'Training {metric_name}')
    plt.plot(epochs, list_valid_epoch_metrics, label=f'Validation {metric_name}')
    
    # Add in a title and axes labels
    plt.title(f'Training and Validation {metric_name} across epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric_name}')
    
    # Set the tick locations
    plt.xticks(np.arange(0, epoch_nb, 2))
    
    # Display the plot
    plt.legend(loc='best')
    plt.show()

def plot_perfs_gnn(classifier:torch.nn.Module, 
                 list_data_test:list,
                 list_performance_metrics:list=["accuracy", "roc_auc", "pr_auc", "fpr_ratio", "tpr_ratio"]
    )->torch.tensor:
    """Computes the desired performance metrics to evaluate a given GNN classifier on test data.

    Args:
        classifier (GCN): already trained GCN classifier (!) must not have been trained on data_test
        list_data_test (torch_geometric.data.Data): new graph-data to evaluate the GCN
            Must have the same attributes (feature nodes, edges...) than data used for GNN training
        list_performance_metrics:
            List of the metrics of performance to plot on test data
            Each element must be set to a value in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy}

    """

    # activate and signal the use of GPU for faster processing
    device = activate_gpu()

    print("\n'''Evaluation Start '''\n")
    classifier = classifier.to(device)
    classifier.eval()

    data_child = list_data_test[-1] # get data.x and data.y from the last graph-data (as it remains the same for all data, with GCN_ancestor)
    print(data_child.x.shape)
    target = data_child.y.to(device)
    target_train = target[data_child.train_mask]
    target_valid = target[data_child.valid_mask]

    skip_connection=True # TODO pass it as internal parameter of the classifier (directly in GCN_ancestor)
    preds = classifier(list_data=list_data_test, device=device, skip_connection=skip_connection)
    probas_pred_train = preds[data_child.train_mask].to(device) 
    probas_pred_valid = preds[data_child.valid_mask].to(device) 

    for metric_name in list_performance_metrics: 

        if metric_name not in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy'}:
            raise NotImplementedError(f"The metric {metric_name} you want to compute is not implemented. "
                                    "Must be set to a value in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy}")

        metrics_train = compute_metric(metric_name, probas_pred_train, target_train, plot=True)
        metrics_valid = compute_metric(metric_name, probas_pred_valid, target_valid, plot=True)

        print(f"{metric_name}: {metrics_train} (train) || {metrics_valid} (valid)")

    return preds
