##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5: Clustering                                   #
#                                                            #
##############################################################

import argparse
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from Chapter5.Clustering import HierarchicalClustering
from Chapter5.Clustering import NonHierarchicalClustering
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up the file names and locations
DATA_PATH = Path('./intermediate_datafiles/')
DATASET_FNAME = 'chapter4_result.csv'
RESULT_FNAME = 'chapter5_result.csv'


def print_flags():
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    # Read the result from the previous chapter convert the index to datetime
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Create an instance of visualization class to plot the results
    DataViz = VisualizeDataset(__file__)

    # Create objects for clustering
    clusteringNH = NonHierarchicalClustering()
    clusteringH = HierarchicalClustering()

    if FLAGS.mode == 'kmeans':
        # Do some initial runs to determine the right number for k
        k_values = range(2, 10)
        silhouette_values = []

        print('Running k-means clustering')
        for k in k_values:
            print(f'k = {k}')
            dataset_cluster = clusteringNH.k_means_over_instances(dataset=copy.deepcopy(dataset),
                                                                  cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'],
                                                                  k=k, distance_metric='default',
                                                                  max_iters=20, n_inits=10)
            silhouette_score = dataset_cluster['silhouette'].mean()
            print(f'silhouette = {silhouette_score}')
            silhouette_values.append(silhouette_score)

        DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                        ylim=[0, 1], line_styles=['b-'])

        # Run the knn with the highest silhouette score
        k = k_values[np.argmax(silhouette_values)]
        print(f'Highest K-Means silhouette score: k = {k}')
        print('Use this value of k to run the --mode=final --k=?')

    if FLAGS.mode == 'kmediods':
        # Do some initial runs to determine the right number for k
        k_values = range(2, 10)
        silhouette_values = []
        print('Running k-medoids clustering')

        for k in k_values:
            print(f'k = {k}')
            dataset_cluster = clusteringNH.k_medoids_over_instances(dataset=copy.deepcopy(dataset),
                                                                    cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'],
                                                                    k=k, distance_metric='default',
                                                                    max_iters=20, n_inits=10)
            silhouette_score = dataset_cluster['silhouette'].mean()
            print(f'silhouette = {silhouette_score}')
            silhouette_values.append(silhouette_score)

        DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                        ylim=[0, 1], line_styles=['b-'])

        # Run k medoids with the highest silhouette score
        k = k_values[np.argmax(silhouette_values)]
        print(f'Highest K-Medoids silhouette score: k = {k}')

        dataset_kmed = clusteringNH.k_medoids_over_instances(dataset=copy.deepcopy(dataset),
                                                             cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'],
                                                             k=k, distance_metric='default',
                                                             max_iters=20, n_inits=50)
        DataViz.plot_clusters_3d(data_table=dataset_kmed, data_cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'],
                                 cluster_col='cluster', label_cols=['label'])
        DataViz.plot_silhouette(data_table=dataset_kmed, cluster_col='cluster', silhouette_col='silhouette')
        util.print_latex_statistics_clusters(dataset=dataset_kmed, cluster_col='cluster',
                                             input_cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'],
                                             label_col='label')

    # Run hierarchical clustering
    if FLAGS.mode == 'agglomerative':
        k_values = range(2, 10)
        silhouette_values = []

        # Do some initial runs to determine the right number for the maximum number of clusters
        print('Running agglomerative clustering')
        for k in k_values:
            print(f'k = {k}')
            dataset_cluster, link = clusteringH.agglomerative_over_instances(
                dataset=dataset, cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], max_clusters=k,
                distance_metric='euclidean', use_prev_linkage=True, link_function='ward')
            silhouette_score = dataset_cluster['silhouette'].mean()
            print(f'silhouette = {silhouette_score}')
            silhouette_values.append(silhouette_score)
            if k == k_values[0]:
                DataViz.plot_dendrogram(dataset_cluster, link)

        # Plot the clustering results
        DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
                        ylim=[0, 1], line_styles=['b-'])

    if FLAGS.mode == 'final':
        # Select the outcome dataset of the knn clustering
        clusteringNH = NonHierarchicalClustering()
        dataset = clusteringNH.k_means_over_instances(
            dataset=dataset, cols=['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], k=FLAGS.k,
            distance_metric='default', max_iters=50, n_inits=50)
        # Plot the results
        DataViz.plot_clusters_3d(dataset, ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], 'cluster', ['label'])
        DataViz.plot_silhouette(dataset, 'cluster', 'silhouette')
        # Print table statistics
        util.print_latex_statistics_clusters(dataset, 'cluster', ['acc_phone_x', 'acc_phone_y', 'acc_phone_z'], 'label')
        del dataset['silhouette']

        # Store the final dataset
        dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, kmeans, kmediods, hierarchical or aggloromative. \
                        'kmeans' to study the effect of kmeans on a selection of variables \
                        'kmediods' to study the effect of kmediods on a selection of variables \
                        'agglomerative' to study the effect of agglomerative clustering on a selection of variables  \
                        'final' kmeans with an optimal level of k is used for the next chapter",
                        choices=['kmeans', 'kmediods', 'agglomerative', 'final'])

    parser.add_argument('--k', type=int, default=6,
                        help="The selected k number of means used in 'final' mode of this chapter")
    FLAGS, unparsed = parser.parse_known_args()

    # Print args and run main script
    print_flags()
    main()
