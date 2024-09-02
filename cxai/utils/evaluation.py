import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch

from cxai.utils.constants import CLASS_IDX_MAPPER


###################################### fucntions for accuracy model evaluation 

def get_acc(model, testloader=None, device=torch.device('cpu'), is_toy=False):
    '''
    Calculates accuracy on test set
    -----
    Args:
        is_toy (bool): Case toy model and toydataset
    Returns:
        acc     (float): mean accuracy across classes
        ypred   (list): predicted labels on test set
        ytrue   (list): true labels of test set
    '''

    model.eval()

    ytrue = []
    ypred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in testloader:
            if not is_toy:
                b, chunks, c, f, t = xb.size() # b=batch
                xb = xb.view(-1, c, f, t)
                yb = yb.repeat_interleave(chunks)

            xb, yb = xb.to(device).squeeze(0), yb.to(device)
            #yb = yb.repeat_interleave(xb.size(0))

            outputs = model(xb)
            _, predicted = torch.max(outputs.detach(), 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            ytrue.extend(yb.cpu().numpy())
            ypred.extend(predicted.cpu().numpy())

    acc = correct / total * 100
    print(f'Accuracy on test set is: {round(acc, 2)}%')

    return acc, np.asarray(ytrue), np.asarray(ypred)


def get_cm(ytrue, ypred, valid_fold=1, plot=True):
    '''
    Calculates and plots confusion matrix.
    -----
    Args:
        ypred   (list): predicted labels on test set
        ytrue   (list): true labels of test set
        plot    (bool): Controls if confusion matrix has to be plotted
    Returns:
        cm (np.ndarray): confusion matrix of shape [num_classes, num_classes]
    '''

    cm = confusion_matrix(ytrue, ypred)
    # convert to percentage
    cm = cm / cm.sum(axis=1) * 100
    if plot:
        plot_cm(cm, valid_fold=valid_fold)

    return cm

def plot_cm(cm, valid_fold=None, genredict=CLASS_IDX_MAPPER):
    ax = sns.heatmap(cm, annot=True, xticklabels=genredict.keys(), yticklabels=genredict.keys(), cmap='YlGnBu', fmt='.1f')

    # Setting the title, x-axis label, and y-axis label
    ax.set_title('Confusion Matrix across 10 folds [%]' if valid_fold==None else f'Confusion Matrix [%], Validation fold: {valid_fold}')  # Main title
    ax.set_xlabel('Predicted label')  # X-axis label
    ax.set_ylabel('True label')  # Y-axis label
    plt.show()


def class_accs(cm, genredict=CLASS_IDX_MAPPER):
    r"""
    Calculates and prints prediction accuaries of each class.
    -----
    Args:
        cm (np.ndarray): confusion matrix of shape [num_classes, num_classes]
    Returns:
        confusion_matrix_dict (dict): prediction accuracy on each class 
    """

    confusion_matrix_dict = {}
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1) * 100
    for i, class_acc in enumerate(class_accuracies):
        print(f'Acc of {list(genredict.keys())[i]:>10s}: {round(class_acc, 2):.2f}%')
        confusion_matrix_dict[list(genredict.keys())[i]] = round(class_acc, 2)
    return confusion_matrix_dict


def get_train_stats(path):
    
    if path.endswith('.csv'):
        csv_files = [path]
    else:
        # case training was interrrupted and carried on at some later point (several .csv files for train stats)
        csv_files = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.csv')]
        csv_files.sort()
    
    all_dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        all_dfs.append(df[['train_loss', 'train_acc', 'valid_losses', 'valid_acc']])

    return pd.concat(all_dfs, ignore_index=True)



###################################### DRSA train stat loading

def get_best_run(path):

    best_loss = 0
    concept_relevances = None

    for dir_level1 in [d for d in os.listdir(path) if not d.startswith('.')]:
        run = int(dir_level1[-1])
        loss, concept_relevances, train_losses = get_run_stats(os.path.join(path, dir_level1, 'train_stats.csv'))

        if loss > best_loss:
            best_loss = loss
            concept_relevances = concept_relevances
            best_run = run
            path_to_best_run = os.path.join(path, dir_level1)

    return best_run, best_loss, concept_relevances, path_to_best_run, train_losses


def get_run_stats(path):

    stats = pd.read_csv(path)
    final_loss = list(stats['loss'])[-1]

    concept_relevances = []
    for key in stats.keys():
        if key.startswith('R'):
            concept_relevances.append(list(stats[key])[-1])
    
    return final_loss, concept_relevances, list(stats['loss'])
