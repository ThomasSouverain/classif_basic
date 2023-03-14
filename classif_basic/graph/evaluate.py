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

from classif_basic.graph.utils import check_attributes_graph_data

def evaluate_gnn(classifier:GCNConv, data_test:torch, device, loss_name:str="cross_entropy"):
    # TODO improve the function 
    """Computes the loss and accuracy of a given GCN classifier on test data

    Args:
        classifier (GCN): already trained GCN classifier (!) must not have been trained on data_test
        data_test (torch): new graph-data to evaluate the GCN
            Must have the same attributes (feature nodes, edges...) than data used for GNN training
        loss_name (str, optional): type of loss the user wants to compute on test data. Defaults to "cross_entropy"
        device: either GPU of CPU, to compute the classifier predictions
    """

    # first of all, check if data_test shares the attributes of the data-graph (data_total) used for our GNN batch training
    check_attributes_graph_data(data_test)

    # set the loss (for the moment, cross_entropy by default) TODO extend to other losses 
    if loss_name == "cross_entropy":
        print("Loss is measured through cross entropy")
        loss = torch.nn.CrossEntropyLoss()

    #classifier = classifier.to(device)
    classifier.eval()

    data_test = data_test.to(device)
    target = data_test.y.to(device)

    preds = classifier(data=data_test, device=device)

    error_test = loss(preds, target)
    print(f"\nError on test: {error_test:.4f} \n")

    # compute overall train&valid accuracy
    _, preds_temp = torch.max(preds.data, 1)
    total = len(target)
    correct = (preds_temp == target).sum().item()
    print(f"Test Accuracy = {round(correct / total, 2)}") 

    # TODO compute and plot ROC-AUC

    return

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

def get_auc(y_true:torch.tensor, probas_pred:torch.tensor, plot=False)->tuple:
    # for AUC computing, separate gradients from Y and probabilities vectors
    y_true_np = y_true.cpu().detach().numpy()
    probas_pred_np = probas_pred.cpu().detach().numpy()
    # only get probability of class 1, for AUCs computing
    probas_pred_class1 = probas_pred_np[:,1]
    precision, recall, _ = precision_recall_curve(y_true=y_true_np, probas_pred=probas_pred_class1)
    fpr, tpr, _ = roc_curve(y_true_np, probas_pred_class1)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision) 

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

    # also return false positive and true positive mean ratios 
    fpr_ratio = fpr.mean()
    tpr_ratio = tpr.mean()
    return roc_auc, pr_auc, fpr_ratio, tpr_ratio

def plot_metrics(metrics_name:str, epoch_nb:int, list_train_epoch_errors:list, list_valid_epoch_errors:list):
    epochs=range(epoch_nb)
    # Plot and label the training and validation loss values
    plt.plot(epochs, list_train_epoch_errors, label=f'Training {metrics_name}')
    plt.plot(epochs, list_valid_epoch_errors, label=f'Validation {metrics_name}')
    
    # Add in a title and axes labels
    plt.title(f'Training and Validation {metrics_name} across epochs')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metrics_name}')
    
    # Set the tick locations
    plt.xticks(np.arange(0, epoch_nb, 2))
    
    # Display the plot
    plt.legend(loc='best')
    plt.show()
