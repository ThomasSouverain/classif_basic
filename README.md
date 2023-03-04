# classif_basic with GPU

Here, the package accesses GPU (if present) to make the computation of models (e.g. graphs, see "notebooks/data_census_graph_tests.ipynb").
To activate the GPU on your computer with docker:

## Install NVIDIA
(!) implies to already have installed NVIDIA'. If you have a GPU and must install NVIDIA, there is a [process](https://www.linuxcapable.com/install-nvidia-drivers-on-ubuntu-linux/):

    sudo apt update
    sudo apt upgrade
    sudo apt autoremove && sudo apt autoclean
    sudo apt-get remove --purge '^nvidia-.*'
    lspci | grep -e VGA
    sudo apt install nvidia-driver-my_version (e.g. nvidia-driver-525)
    reboot
    
*If* a package blocks the installation, make:

    sudo dpkg --remove --force-all package_name
    
And, if the package can still not be removed:

    sudo rm /var/lib/dpkg/info/package_name.prerm
    
*If* problem of "unmet dependencies", make:

    sudo apt-add-repository -r ppa:graphics-drivers/ppa

*.. And do again the steps above*

## First activation of GPU with the package
If it is the first time you activate the GPU (and you have installed NVIDIA):

    cd project_name (e.g. classif_basic)
    
create / replace your Dockerfile_gpu with [this content](https://pastebin.com/fD0VQrUD)

    sudo rm -rf .cache
    make install-gpu
    make start-gpu
    make notebook

## Usual launching of notebook with GPU 
For the other times, normal launching of a jupyter notebook with your GPU activated:

    cd project_name (e.g. classif_basic)
    make start-gpu
    make notebook

On basic binary and multi-classification, and on regression (see the notebooks for application), this package provides scripts to accelerate:

## The basic processing of data

General preparations:

- "train_valid_test_split": splits the data in train / valid / test samples to train the model using cross-validation, according to the data scientist's preferences. Also returns a "train_valid" sample to inspect the model's functioning on these both sets. 

- "set_target_if_feature": selects a sub-sample of the dataset according to desired effect of a feature (percentages of positive target, e.g. of people selected for a loan, inside a feature's group). Used to control if the learning of a feature effect by the algorithm is efficient.


Preparations specific to a dataset:

- "automatic_preprocessing": prepares the Census public dataset (https://www.kaggle.com/datasets/uciml/adult-census-income) for binary classification of people whose income > $50_000

- "new_dataset_column": merges the individuals' and families' data by request_id for the DreamQuark "housing_nights_dataset"


## The training of a tree-based model in case of binary or multi-classification, or regression ##

- "train_naive_xgb": trains a basic tree-based algorithm from XGBoost (with defaults settings of hyper-parameters)

- "pickle_save_model": stores the model

- "prediction_train_valid_by_task": provides a predicted target

- "compute_best_fscore": specific to the binary classification cases, to optimise the transition from scores -> to labels (0, 1) according to the F-score.

## First analysis of the model ## 

- "features_importances_from_pickle": basically computes the features' importance attribution (using the Shapley values approximated by SHAP library)

- "select_important_features": from initial features list (by default from Census dataset), selects the one with the highest SHAP importance (by default >1%)

- "augment_train_valid_set_with_results": adds predictions and statistics to a model to the dataset train&valid,
    to enable further comparison and selection between the initial and fair train models.
 
