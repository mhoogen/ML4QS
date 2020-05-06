import re
import scipy
import copy
import math
import numpy as np

# Not a class, just a bunch of useful functions.

def get_chapter(module_path):
    return re.search('_ch._', 'crowdsignals_ch3_outliers.py').group(0).strip('_')

def normalize_dataset(data_table, columns):
    dt_norm = copy.deepcopy(data_table)
    for col in columns:
        dt_norm[col] = (data_table[col] - data_table[col].mean()) / (data_table[col].max() - data_table[col].min())
    return dt_norm

# Calculate the distance between rows.
def distance(rows, d_function='euclidean'):
    if d_function == 'euclidean':
        # Assumes m rows and n columns (attributes), returns and array where each row represents
        # the distances to the other rows (except the own row).
        return scipy.spatial.distance.pdist(rows, 'euclidean') # todo: replace with numpy?
    else:
        raise ValueError("Unknown distance value '" + d_function + "'")

def print_statistics(dataset, describe=True):

    if describe:
        # .describe() gives number of values, mean, standard deviation, min and max for each column in one table.
        print(dataset.describe().round(3).to_string())
        return

    print('\ncolumn \t\t % missing \t\t mean \t\t standard deviation \t\t min \t\t max')
    dataset_length = len(dataset.index)
    for col in dataset.columns:
        print('\t\t'.join([f'{col}',
                           f'{(dataset_length - dataset[col].count()) / dataset_length * 100:3.1f}%',
                           f'{dataset[col].mean():6.3f}',
                           f'{dataset[col].std():6.3f}',
                           f'{dataset[col].min():6.3f}',
                           f'{dataset[col].max():6.3f}']))

def print_table_cell(value1, value2):
    print("{0:.2f}".format(value1), ' / ', "{0:.2f}".format(value2), end='')

def print_latex_table_statistics_two_datasets(dataset1, dataset2):
    print('attribute, fraction missing values, mean, standard deviation, min, max')
    dataset1_length = len(dataset1.index)
    dataset2_length = len(dataset2.index)
    for col in dataset1.columns:
        print(col, '& ', end='')
        print_table_cell((float((dataset1_length - dataset1[col].count()))/dataset1_length)*100, (float((dataset2_length - dataset2[col].count()))/dataset2_length)*100)
        print(' & ', end='')
        print_table_cell(dataset1[col].mean(), dataset2[col].mean())
        print(' & ', end='')
        print_table_cell(dataset1[col].std(), dataset2[col].std())
        print(' & ', end='')
        print_table_cell(dataset1[col].min(), dataset2[col].min())
        print(' & ', end='')
        print_table_cell(dataset1[col].max(), dataset2[col].max())
        print('\\\\')

def print_latex_statistics_clusters(dataset, cluster_col, input_cols, label_col):
    label_cols = [c for c in dataset.columns if label_col == c[0:len(label_col)]]

    clusters = dataset[cluster_col].unique()

    for c in input_cols:
        print('\multirow{2}{*}{', c, '} & mean ', end='')
        for cluster in clusters:
            print(' & ', "{0:.2f}".format(dataset.loc[dataset[cluster_col] == cluster, c].mean()), end='')
        print('\\\\')
        print(' & std ', end='')
        for cluster in clusters:
            print(' & ', "{0:.2f}".format(dataset.loc[dataset[cluster_col] == cluster, c].std()), end='')
        print('\\\\')

    for l in label_cols:
        print(l, ' & percentage ', end='')
        for cluster in clusters:
            print(' & ', "{0:.2f}".format((float(dataset.loc[dataset[cluster_col] == cluster, l].sum())/len(dataset[dataset[l] == 1].index) * 100)), '\%', end='')
        print('\\\\')

def print_table_row_performances(row_name, training_len, test_len, values):
    scores_over_sd = []
    print(row_name, end='')

    for val in values:
        print(' & ', end='')
        sd_train = math.sqrt((val[0]*(1-val[0]))/training_len)
        print("{0:.4f}".format(val[0]), end='')
        print('\\emph{(', "{0:.4f}".format(val[0]-2*sd_train), '-', "{0:.4f}".format(val[0]+2*sd_train), ')}', ' & ', end='')
        sd_test = math.sqrt((val[1]*(1-val[1]))/test_len)
        print("{0:.4f}".format(val[1]), end='')
        print('\\emph{(', "{0:.4f}".format(val[1]-2*sd_test), '-', "{0:.4f}".format(val[1]+2*sd_test), ')}', end='')
        scores_over_sd.append([val[0], sd_train, val[1], sd_test])
    print('\\\\\\hline')
    return scores_over_sd

def print_table_row_performances_regression(row_name, training_len, test_len, values):
    print(row_name),

    for val in values:
        print(' & ', end='')
        print("{0:.4f}".format(val[0]), end='')
        print('\\emph{(', "{0:.4f}".format(val[1]), ')}', ' & ', end='')
        print("{0:.4f}".format(val[2]), end='')
        print('\\emph{(', "{0:.4f}".format(val[3]), ')}', end='')
    print('\\\\\\hline')

def print_pearson_correlations(correlations):
    for i in range(0, len(correlations)):
        if np.isfinite(correlations[i][1]):
            print(correlations[i][0], ' & ', "{0:.4f}".format(correlations[i][1]), '\\\\\\hline')
