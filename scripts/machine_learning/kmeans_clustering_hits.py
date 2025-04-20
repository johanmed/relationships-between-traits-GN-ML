#!/usr/bin/env python

"""
Script 15a
Summary:
This script contains code to run KMeans clustering algorithm on data
Dependencies:
- vector_data_pre.py -> data, preprocessing_hits
- general_clustering -> ModellingKMeans
Modelling by hits (chromosome number + chromosomal position)
"""



# 1. Import X from vector_data script, select relevant columns and transform in appropriate format

import os

from vector_data_pre import train_set as X_train
from vector_data_pre import valid_set as X_valid
from vector_data_pre import test_set as X_test

from vector_data_pre import preprocessing_hits

import pandas as pd
import numpy as np

desc_train = X_train['full_desc']

X_train=X_train[['lod', 'chr_num', 'pos']]

desc_valid = X_valid['full_desc']

X_valid=X_valid[['lod', 'chr_num', 'pos']]

desc_test = X_test['full_desc']

X_test=X_test[['lod', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set


# 2. Select the 2 columns, do clustering and plot

from sklearn.cluster import MiniBatchKMeans # import MiniBatchKMeans class
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
        Extract 2 PCA from preprocessing_hits pipeline
        """
        preprocessed_training=preprocessing_hits.fit_transform(self.training)
        preprocessed_validation=preprocessing_hits.transform(self.validation)
        preprocessed_test=preprocessing_hits.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_kmeans_clustering(self, reduced_features_valid, n_clusters):
        """
        Run KMeans for number of clusters on training
        Return pipeline: hits transformation + clustering
        """
        
        kmeans_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('kmeans', MiniBatchKMeans(random_state=2024, n_init=10, n_clusters=n_clusters))]) # perform clustering with specified number of clusters
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
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_clustering_result_by_hits_{n_clusters}_clusters"), dpi=500)
        
    
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
        
            for chromo in sorted(anno):
                
                if chromo in dic.keys():
                    continue
                else:
                    dic[chromo] = start # associate numeric values to colors
                    start += 1
                    
            labels=[dic[chromo] for chromo in anno] # apply previous formatting to trait to get corresponding value
            
            unique_labels = list(dic.values())
            unique_names = list(dic.keys())
            
             # Rename chromo 88 to X
            
            final_names = []
            for name in unique_names:
                if name == 88:
                    final_names.append('X')
                final_names.append(name)
            
            
            for ind, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(X[mask, 0], X[mask, 1], c=colors[ind], label=final_names[ind], alpha=0.7, edgecolors='black', linewidth=0.5)
                
            
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(out_dir, f"MiniBatchKMeans_hits_clustering_data_annotated_{type_anno}"), dpi=500)
        
  

# Main

import joblib

def main():
        
    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()
        
    for n_clusters in range(2, 11): # try values between 2 and 10

        actual_clustering=clustering_task.perform_kmeans_clustering(X_valid_features, n_clusters)

        Columns2Clustering.visualize_plot(Columns2Clustering.plot_kmeans, actual_clustering[1], X_valid_features, n_clusters) # Plot only validation data (more manageable than training data for plotting)

        
    Columns2Clustering.annotate_plot(X_valid_features, list(desc_valid), 'trait')
        
    Columns2Clustering.annotate_plot(X_valid_features, list(X_valid['chr_num']), 'chromo')
    

main()
