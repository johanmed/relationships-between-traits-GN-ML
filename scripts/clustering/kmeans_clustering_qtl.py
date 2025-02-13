#!/usr/bin/env python

"""
Script 15b
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies:
- vector_data_pre.py -> data, preprocessing_qtl
- general_clustering -> ModellingKMeans
KMeans is run twice:
1. Identify the best number of clusters for the data using the training data
2. Proceed to actual training and validation on respective data
Modelling by QTL (chromosome number)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

import os

from vector_data_pre import training_set as X_train
from vector_data_pre import validation_set as X_valid
from vector_data_pre import test_set as X_test

from vector_data_pre import preprocessing_qtl

import pandas as pd
import numpy as np

X_train=X_train[['p_lrt', 'chr_num']]

X_valid=X_valid[['p_lrt', 'chr_num']]

X_test=X_test[['p_lrt', 'chr_num']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import MiniBatchKMeans # import MiniBatchKMeans class
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt # import plot manager
from sklearn.pipeline import Pipeline

from general_clustering import ModellingKMeans


out_dir=os.path.abspath('../../output/') # define directory to save plots to




class Columns2Clustering(ModellingKMeans):

    """
    Represent clustering task on only 2 features extracted from dimensionality reduction
    """
    
    def get_features(self):
        """
        Extract 2 PCA from transformation
        Try different number of clusters and select the one with the best silhouette score
        Return pipeline: QTL transformation + clustering with the best number of clusters
        """
        preprocessed_training=preprocessing_qtl.fit_transform(self.training)
        preprocessed_validation=preprocessing_qtl.transform(self.validation)
        preprocessed_test=preprocessing_qtl.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_kmeans_clustering(self, reduced_features_valid, n_clusters=10):
        """
        Run KMeans for number of clusters on training
        """
        n_cluster_sil=[]
        for num in range(2, n_clusters):
            kmeans_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('kmeans', MiniBatchKMeans(random_state=2024, n_init=10, n_clusters=num))])
            kmeans_clustering.fit(self.training)
            y_pred=kmeans_clustering.predict(self.validation)
            
            sil=silhouette_score(reduced_features_valid, y_pred)
            print('The silhouette score obtained as clustering performance measure is:', sil)
            n_cluster_sil.append([num, sil])
            
            
        def sort_second(arr): # utility function to sort based on second element
            return arr[1]
            
        sorted_n_cluster_sil=sorted(n_cluster_sil, key=sort_second, reverse=True) # sort according to the highest silhouette score
        
        kmeans_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('kmeans', MiniBatchKMeans(random_state=2024, n_init=10, n_clusters=sorted_n_cluster_sil[0][0]))]) # reperform clustering with the best number of clusters
        kmeans_clustering.fit(self.training)
        
        return kmeans_clustering



    def plot_kmeans(clusterer, X, plot_decision_boundaries):
        """
        Plot clusters extracted by KMeans
        """
                
        plot_decision_boundaries(clusterer, X)
        

    def visualize_plot(plot_kmeans, clusterer, X_train):
        """
        Generate actual visualization of clusters
        Save figure
        """
        
        plt.figure(figsize=(10, 10))
        plot_kmeans(clusterer, X_train, Columns2Clustering.plot_decision_boundaries)
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_clustering_result_by_qtl"), dpi=500)
        



# Main

import joblib

def main():
    
    if os.path.exists('kmeans_clustering/kmeans_clustering_qtl.pkl'): # check if this has already been saved
        
        print('The model has already been trained and saved on disk!')
        
    else: # Proceed to clustering and  save model if not yet done
    
        clustering_task=Columns2Clustering(X_train, X_valid, X_test)

        X_train_features, X_valid_features, X_test_features=clustering_task.get_features()

        actual_clustering=clustering_task.perform_kmeans_clustering(X_valid_features)

        joblib.dump(actual_clustering[1], 'kmeans_clustering/kmeans_clustering_qtl.pkl')
        
        #Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[1], X_train_features)

        Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[1], X_valid_features) # Plot only validation data (more manageable than training data for plotting)

        


main()
