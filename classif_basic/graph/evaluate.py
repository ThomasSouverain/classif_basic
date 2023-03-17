from typing import List
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.pylab import plt
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from torch_geometric.nn import GCNConv

from classif_basic.graph.utils import activate_gpu
from classif_basic.graph.utils import check_attributes_graph_data
from classif_basic.graph.utils import tensor_to_numpy
from classif_basic.model import compute_best_fscore

def get_loss(loss_name:str, probas_pred: torch.tensor, y_true: torch.tensor):
    """Computes the loss according to the chosen metrics for GNN training.
     Uses only CPU to avoid GPU wasting during training.

    Args:
        error_metrics (str): Measurement of the performance during GNN training.
            Must be set to a value in {'CrossEntropyLoss', 'L1Loss'}
        y (Union(np.array, torch.tensor)):
            The vector of targets with values in classes_values
            Of shape (nb_indivs,)
        probas_pred: Union(np.array, torch.tensor)
            The vector of probabilities for each class, predicted by the GNN
            Of shape (nb_indivs, nb_classes)
        device: either GPU or CPU must be activated, to enable faster computation of error metrics
    """
    y_true=y_true.cpu()
    probas_pred=probas_pred.cpu()

    if loss_name=="CrossEntropyLoss":
        loss=torch.nn.CrossEntropyLoss()
        error_metrics=loss(probas_pred, y_true)
    
    elif loss_name=="L1Loss":
        loss=torch.nn.L1Loss()
        error_metrics=loss(probas_pred, y_true)

    else:
        raise NotImplementedError("The loss_name you want to implement for GNN training is not implemented"
                                  "Must be set to a value in {'CrossEntropyLoss', 'L1Loss'}")
    
    return error_metrics

def get_auc(y_true:Union[torch.tensor, np.array], probas_pred:Union[torch.tensor, np.array], plot=False)->tuple:
    """Computes the ROC & PR AUCs, and false and true positive ratios, 
    given a vector of probabilities for 2 classes and true labels (binary classification).

    Args:
        probas_pred (Union[torch.tensor, np.array]): vector of probabilities for the 2 classes
            shape(n_indivs, 2)
        y_true (Union[torch.tensor, np.array]): true labels 
            Must be set to values in {0,1}
            shape (n_indivs, )
        plot (bool, optional):if True, plots the ROC AUC and PR AUC curves. Defaults to False.

    Returns:
        tuple: roc_auc, pr_auc, fpr_ratio, tpr_ratio
    """
    # ensure to use CPU and arrays -> accuracy and AUC computing need to separate values from gradients
    probas_pred = tensor_to_numpy(probas_pred) if torch.is_tensor(probas_pred) else probas_pred
    y_true = tensor_to_numpy(y_true) if torch.is_tensor(y_true) else y_true
    # in case of np.array or pandas objects, already in the expected format to plot AUC
    # only get probability of class 1, for AUCs computing
    probas_pred_class1 = probas_pred[:,1]
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=probas_pred_class1)
    fpr, tpr, _ = roc_curve(y_true, probas_pred_class1)
    roc_auc = round(auc(fpr, tpr),2)
    pr_auc = round(auc(recall, precision),2)

    if plot==True:
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y_true[y_true == 1]) / len(y_true)
        # plot the no skill precision-recall curve
        plt.plot(
            [0, 1], [no_skill, no_skill], linestyle="--", label="No Skill PR AUC"
        )
        plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill ROC AUC")
        # plot model precision-recall curve
        plt.plot(recall, precision, marker=".", label=f"PR AUC ({pr_auc})")
        plt.plot(fpr, tpr, marker=".", label=f"ROC AUC ({roc_auc})")
        # Add in a title and axes labels
        plt.title("Last Epoch Classifier - PR AUC and ROC AUC")
        plt.xlabel("Recall/TPR")
        plt.ylabel("Precision/FPR")
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    # also return false positive and true positive mean ratios TODO other aggregation than mean?
    fpr_ratio = round(fpr.mean(),2)
    tpr_ratio = round(tpr.mean(),2)
    return roc_auc, pr_auc, fpr_ratio, tpr_ratio

def compute_metric(metric_name:str, probas_pred:Union[torch.tensor, np.array], y_true:Union[torch.tensor, np.array], plot:bool=False)->float:
    """Computes the desired metric for binary classification, given a vector of probabilities for 2 classes and true labels.

    Args:
        metric_name (str): metric to be computed
            Must be set to a value in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy}
        probas_pred (Union[torch.tensor, np.array]): vector of probabilities for the 2 classes
            shape(n_indivs, 2)
        y_true (Union[torch.tensor, np.array]): true labels 
            Must be set to values in {0,1}
            shape (n_indivs, )
        plot (bool): by default False. If True, plots:
            - PR AUC & ROC AUC for metrics_name in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio'}
            - precision / recall curve (used to choose the best threshold with Fscore) if metrics_name == "accuracy"

    Raises:
        NotImplementedError: when metric's computing is not implemented 

    Returns:
        float: value of the metric
    """
    # ensure to use CPU and arrays -> accuracy and AUC computing need to separate values from gradients    
    probas_pred = tensor_to_numpy(probas_pred) if torch.is_tensor(probas_pred) else probas_pred
    y_true = tensor_to_numpy(y_true) if torch.is_tensor(y_true) else y_true

    if metric_name in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio'}:
        roc_auc, pr_auc, fpr_ratio, tpr_ratio = get_auc(
                                    probas_pred=probas_pred, 
                                    y_true=y_true)
        if metric_name == 'roc_auc':
            metric = roc_auc
        elif metric_name == 'pr_auc':
            metric = pr_auc
        elif metric_name == 'fpr_ratio':
            metric = fpr_ratio
        elif metric_name == 'tpr_ratio':
            metric = tpr_ratio
    
    elif metric_name == "accuracy":
        ## set y predicted with optimised thresholds, to compute accuracy
        best_threshold, best_fscore = compute_best_fscore(y_true, probas_pred[:,1], plot=plot)

        Y_pred = (probas_pred[:,1] >= best_threshold).astype(int) # threshold optimized with data, by default 
        #Y_pred = Y_pred.long() if torch.is_tensor(Y_pred) else Y_pred.astype(int) # convert bool to int 
        # _, Y_pred = torch.max(probas_pred.data, 1) # or with torch.max
        total = len(y_true) 
        correct = (Y_pred==y_true).sum()
        metric = round(correct/total, 2)
    
    else:
        raise NotImplementedError("The metric you want to compute is not implemented. "
                                  "Must be set to a value in {'roc_auc', 'pr_auc', 'fpr_ratio', 'tpr_ratio', 'accuracy}")

    return metric
