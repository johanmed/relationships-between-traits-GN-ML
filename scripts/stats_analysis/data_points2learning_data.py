#!/usr/bin/env python

"""
Script 19

This script uses best clustering model previously identified to predict data index, cluster assigned and distance to its centroid for the training set + validation set (whole dataset without test set)
This will then further be analyzed
Dependencies: vector_data_pre.py -> preprocessing_hits, vector_data_post.py -> training_validation_set
"""

import os
import pandas as pd
import tensorflow as tf
from random import choice

from vector_data_post import training_validation_set as processed_X # use concatenation of training and validation sets

from vector_data_pre import preprocessing_hits # use transformation pipeline of the best clustering model found

# Define features

clusters_hits_full = processed_X['clusters_hits']

distances_hits_full = processed_X['distances_hits']

X_full=processed_X[['p_lrt', 'chr_num', 'pos']]


class NewColumns2Clustering:

    """
    Represent supervised learning task on 2 columns extracted from transformation pipeline
    """
    
    def __init__(self, data):
        self.data=data # use whole dataset without test set

    def get_features(self):
        """
        Extract 2 PCA from preprocessing_hits pipeline
        """
        return preprocessing_hits.fit_transform(self.data)
        
    def get_clusters_labels(raw_predictions_proba):
    
        """
        Loop through output dimensions and select the cluster with the highest probability
        """
        clusters=[]
        for i in raw_predictions_proba:
            temp=[]
            for (x,y) in enumerate(i):
                if y==max(i):
                    temp.append(x)
            clusters.append(choice(temp))
        
        return clusters
        
    def predict_neural_clustering(neural_clustering, data, get_clusters_labels):
        """
        Use neural networks to predict cluster and distance of each data in the set
        """
        clusters_pred, distances_pred = neural_clustering.predict(data)
        
        final_clusters = get_clusters_labels(clusters_pred)
        
        return final_clusters, distances_pred


def main():

    clustering_task = NewColumns2Clustering(X_full)

    X_full_features = clustering_task.get_features() # get preprocessed features for the whole set
    
    type=input('Please enter the type of model you want to use for extraction of deep learning results: ')
    
    if os.path.exists(f'../clustering/deep_learning_clustering_{type}/best_clustering_model_by_hits.keras'): # check first existence of best_clustering_model_by hits
        
        best_model=tf.keras.models.load_model(f'../clustering/deep_learning_clustering_{type}/best_clustering_model_by_hits.keras')
        
        prediction_clusters, prediction_distances = NewColumns2Clustering.predict_neural_clustering(best_model, X_full_features, NewColumns2Clustering.get_clusters_labels) # get predictions
        
        container=[]
        for (i, (j, k)) in enumerate(zip(prediction_clusters, prediction_distances)):
            container.append([i, j, k[0]]) # save data index, cluster assigned and distance to centroid for each observation, because k is saved in array despite having only one element, need to index 
            
        to_save=pd.DataFrame(container) # convert to a dataframe
        to_save.to_csv(f'../../../data_indices_learning_data_{type}.csv', index=False, header=False) # save as csv
            
    elif os.path.exists(f'../clustering/deep_learning_clustering_{type}/best_checkpoint.keras'): # if fails, check existence of best_checkpoint
        
        best_model=tf.keras.models.load_model(f'../clustering/deep_learning_clustering_{type}/best_checkpoint.keras')
        
        prediction_clusters, prediction_distances = NewColumns2Clustering.predict_neural_clustering(best_model, X_full_features, NewColumns2Clustering.get_clusters_labels)
        
        container=[]
        for (i, (j, k)) in enumerate(zip(prediction_clusters, prediction_distances)):
            container.append([i, j, k[0]]) # same
            
        to_save=pd.DataFrame(container)
        to_save.to_csv(f'../../../data_indices_learning_data_{type}.csv', index=False, header=False)
    
    else: # handle gracefully when none of the above exists
    
        print('Best model not found, sorry :)')
    


main()
