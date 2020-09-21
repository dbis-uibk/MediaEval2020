import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate


def print_confusion_matrix(confusion_matrix,
                           labels=None,
                           normalize_axis=None,
                           figsize=(10, 7),
                           fontsize=11):
    """
    Prints a confusion matrix, as returned by sklearn, as a heatmap.

    Args:
        confusion_matrix: returned from sklearn.metrics.confusion_matrix or
            with a matching format.
        labels: list of class names, in the order they index the given
            confusion matrix.
        normalize_axis: configures the axis that is used of normalization. Set
            to None to disable normalization.
        figsize:  of the resulting image where the first value determins the
            horizontal size and the second determins the vertical size.
        fontsize: of the axes labels.

    Returns: The resulting confusion matrix figure.
    """
    confusion_matrix = pd.DataFrame(
        confusion_matrix,
        index=labels,
        columns=labels,
    )
    if normalize_axis is not None:
        if normalize_axis < 0 or normalize_axis > 1:
            raise ValueError('The normalize axis needs to be 0 or 1.')
        denominator = confusion_matrix.astype(float).sum(axis=normalize_axis)
        if normalize_axis == 1:
            denominator = denominator[:, np.newaxis]
        confusion_matrix = confusion_matrix / denominator
    fig = plt.figure(figsize=figsize)
    fmt = '.2f' if normalize_axis is not None else 'd'
    try:
        heatmap = sns.heatmap(confusion_matrix, annot=True, fmt=fmt)
    except ValueError:
        raise ValueError('Confusion matrix values must be integers.')
    plt.setp(heatmap.yaxis.get_ticklabels(),
             rotation=0,
             ha='right',
             fontsize=fontsize)
    plt.setp(heatmap.xaxis.get_ticklabels(),
             rotation=0,
             ha='right',
             fontsize=fontsize)
    plt.title('Confution Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return fig


def print_per_label(data, figsize=(10, 7), fontsize=11):
    sns.set(style='whitegrid')
    fig = plt.figure(figsize=figsize)
    lineplot = sns.lineplot(
        data=data,
        markers=True,
        dashes=False,
        palette=sns.color_palette('tab10', n_colors=data.shape[1]),
    )
    plt.setp(lineplot.get_xticklabels(), rotation=90, fontsize=fontsize)
    plt.title('Scores per Label')
    plt.xlabel('Label')
    plt.ylabel('Metric score')

    return fig


def plot_confusion_matrix(database_entry,
                          normalize_axis=None,
                          multilabel=None):
    confusion_matrix = np.array(database_entry.outcome['confusion_matrix'])
    if multilabel:
        try:
            idx = int(multilabel)
        except ValueError:
            labels = _strip_labels(database_entry.dataloader['classes'])
            idx = labels.index(multilabel)
        confusion_matrix = confusion_matrix[idx]
    if confusion_matrix.shape == (2, 2):
        print_confusion_matrix(confusion_matrix,
                               labels=['false', 'true'],
                               normalize_axis=normalize_axis)
    else:
        print_confusion_matrix(confusion_matrix,
                               labels=database_entry.dataloader['classes'],
                               normalize_axis=normalize_axis)
    plt.show()


def extract_metrics(database_entry):
    # TODO: Select metrics dynamic
    accuracy_all = []
    precision_all = []
    recall_all = []
    f1_all = []

    outcome = extract_final_outcome(database_entry.outcome)

    for label_cf in outcome['confusion_matrix']:
        tn, fp, fn, tp = np.array(label_cf).ravel()
        accuracy_all.append((tp + tn) / (tp + tn + fp + fn))
        precision_all.append(tp / (tp + fp))
        recall_all.append(tp / (tp + fn))
        f1_all.append(2 * tp / (2 * tp + fp + fn))

    data = zip(
        outcome['roc_auc_all'],
        outcome['average_precision_all'],
        accuracy_all,
        precision_all,
        recall_all,
        f1_all,
    )
    try:
        labels = _strip_labels(database_entry.dataloader['classes'])
    except KeyError:
        labels = None
    return pd.DataFrame(
        data,
        index=labels,
        columns=['roc-auc', 'pr-auc', 'accuracy', 'precision', 'recall', 'f1'])


def plot_per_label(database_entry):
    data = extract_metrics(database_entry)
    print_per_label(data)
    plt.show()


def _strip_labels(labels):
    return list(map(lambda name: str(name[13:]), labels))


def extract_final_outcome(outcome):
    try:
        key = max([int(e) for e in outcome.keys()])
        return outcome[str(key)]
    except ValueError:
        return outcome
