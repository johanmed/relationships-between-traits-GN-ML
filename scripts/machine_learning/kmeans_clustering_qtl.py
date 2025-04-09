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

from vector_data_pre import train_set as X_train
from vector_data_pre import valid_set as X_valid
from vector_data_pre import test_set as X_test

from vector_data_pre import preprocessing_qtl

import pandas as pd
import numpy as np

desc_train = X_train['full_desc']

X_train=X_train[['lod', 'chr_num']]

desc_valid = X_valid['full_desc']

X_valid=X_valid[['lod', 'chr_num']]

desc_test = X_test['full_desc']

X_test=X_test[['lod', 'chr_num']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import MiniBatchKMeans # import MiniBatchKMeans class
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt # import plot manager
import matplotlib
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
        
        print(preprocessed_training)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_kmeans_clustering(self, reduced_features_valid, n_clusters):
        """
        Run KMeans for number of clusters on training
        """
       
        kmeans_clustering=Pipeline([('preprocessing_qtl', preprocessing_qtl), ('kmeans', MiniBatchKMeans(random_state=2024, n_init=10, n_clusters=n_clusters))]) # reperform clustering with the best number of clusters
        kmeans_clustering.fit(self.training)
        
        return kmeans_clustering



    def plot_kmeans(clusterer, X, plot_decision_boundaries):
        """
        Plot clusters extracted by KMeans
        """
                
        plot_decision_boundaries(clusterer, X)
        

    def visualize_plot(plot_kmeans, clusterer, X_train, n_clusters):
        """
        Generate actual visualization of clusters
        Save figure
        """
        
        plt.figure(figsize=(10, 10))
        plot_kmeans(clusterer, X_train, Columns2Clustering.plot_decision_boundaries)
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_clustering_result_by_qtl_{n_clusters}_clusters"), dpi=500)
        
        

    def annotate_plot(X, anno, type_anno, size=500):
        """
        Annotate plot with trait information
        Save figure
        """
        
        colors = matplotlib.colormaps['tab20b'].colors +  matplotlib.colormaps['tab20c'].colors # define possible colors
        
        plt.figure(figsize=(10, 10))
            
            
        if type_anno == 'trait':
        
            dic={} # Get numeric values for anno that can be used for color
        
            start=0
        
            for ind, trait in enumerate(anno):
                trait = ' '.join(trait.split(' ')[:2]) # select only first 2 words of description, do away of dataset name for simplicity, as long as description is the same, consider the same
                if trait in dic.keys():
                    continue
                else:
                    dic[trait] = start # associate numeric values to colors
                    start += 1
        
            labels=[dic[' '.join(trait.split(' ')[:2])] for trait in anno] # apply previous formatting to trait to get corresponding value
            
            unique_labels = list(dic.values())
            unique_names = list(dic.keys())
    
    
            for ind, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(X[mask, 0], X[mask, 1], c=colors[ind], label=f'Trait {unique_names[ind][:50]}', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        elif type_anno == 'chromo':
        
            dic={} 
        
            start=0
        
            for ind, chromo in enumerate(anno):
                
                if chromo in dic.keys():
                    continue
                else:
                    dic[chromo] = start # associate numeric values to colors
                    start += 1
                    
            labels=[dic[chromo] for chromo in anno] # apply previous formatting to trait to get corresponding value
            
            unique_labels = list(dic.values())
            unique_names = list(dic.keys())
            
            for ind, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(X[mask, 0], X[mask, 1], c=colors[ind], label=label, alpha=0.7, edgecolors='black', linewidth=0.5)
                
            
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_qtl_clustering_data_annotated_{type_anno}"), dpi=500)
        
        

# Main

import joblib

def main():
    
    if os.path.exists('kmeans_clustering/kmeans_clustering_qtl.pkl'): # check if this has already been saved
        
        print('The model has already been trained and saved on disk!')
        
    else: # Proceed to clustering and  save model if not yet done
    
        clustering_task=Columns2Clustering(X_train, X_valid, X_test)

        X_train_features, X_valid_features, X_test_features=clustering_task.get_features()
        
        for n_clusters in range(2, 11): # try values between 2 and 10

            actual_clustering=clustering_task.perform_kmeans_clustering(X_valid_features, n_clusters)

            Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[1], X_valid_features, n_clusters) # Plot only validation data (more manageable than training data for plotting)

        
        Columns2Clustering.annotate_plot(X_valid_features, list(desc_valid), 'trait')
        
        Columns2Clustering.annotate_plot(X_valid_features, list(X_valid['chr_num']), 'chromo')
    
    

main()
