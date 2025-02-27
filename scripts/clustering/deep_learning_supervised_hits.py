#!/usr/bin/env python
"""
Script 18a

This script contains code to performed supervised deep learning using clustering and distance information from the best classical clustering
Neural architecture built to predict the right cluster and distance of each data point to its centroid 
Modelling by hits (chromosome number + marker position)
Dependencies: vector_data_post.py -> data, vector_data_pre.py -> preprocessing_hits

NB: Need to make sure the number of neurons in output_layer1 of general_deep_learning.py is set to the right number of clusters found by classical models for hits modeling
"""


# 1. Import data and select relevant columns

import os

from vector_data_post import training_set as X_train
from vector_data_post import validation_set as X_valid
from vector_data_post import test_set as X_test

from vector_data_pre import preprocessing_hits

import numpy as np
import pandas as pd

clusters_train=X_train['clusters_hits']

distances_train=X_train['distances_hits']

desc_train=X_train['full_desc'] # for data point annotation according to trait later

X_train=X_train[['p_lrt', 'chr_num', 'pos']]


clusters_valid=X_valid['clusters_hits']

distances_valid=X_valid['distances_hits']

desc_valid=X_valid['full_desc']

X_valid=X_valid[['p_lrt', 'chr_num', 'pos']]


clusters_test=X_test['clusters_hits']

distances_test=X_test['distances_hits']

desc_test=X_test['full_desc']

X_test=X_test[['p_lrt', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set

clusters_train_full=pd.concat([clusters_train, clusters_valid])

distances_train_full=pd.concat([distances_train, distances_valid])


# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf
import matplotlib.pyplot as plt # import plot manager
import matplotlib
from sklearn.pipeline import Pipeline
from random import choice
from pathlib import Path
from time import strftime
import keras_tuner as kt

tf.keras.utils.set_random_seed(2024) # set random seed for tf, np and python

out_dir=os.path.abspath('../../output/') # define directory to save plots in


from general_deep_learning import MyClusteringTaskTuning # import MyClusteringTaskTuning and its utilities


class Columns2Clustering:

    """
    Represent clustering task on 2 columns extracted from transformation pipeline
    """
    
    def __init__(self, training, validation, test):
        self.training=training
        self.validation=validation
        self.test=test
    
    
    def get_features(self):
        """
        Extract 2 PC from preprocessing_hits pipeline
        """
        preprocessed_training=preprocessing_hits.fit_transform(self.training)
        preprocessed_validation=preprocessing_hits.transform(self.validation)
        preprocessed_test=preprocessing_hits.transform(self.test)
        
        return preprocessed_training, preprocessed_validation, preprocessed_test
        

    def perform_neural_clustering(self, best_clustering_model, X_valid_features, clusters_train, clusters_valid, distances_train, distances_valid):
        """
        Perform neural clustering on 2 features columns using best model extracted from tuning
        """
        neural_clustering=Pipeline([('preprocessing_hits', preprocessing_hits), ('best_clustering_model', best_clustering_model)])
        neural_clustering.fit(self.training, (clusters_train, distances_train), best_clustering_model__validation_data=(X_valid_features, (clusters_valid, distances_valid)))
        
        return neural_clustering
    
    
    
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
    
    
    
    def visualize_plot(neural_clustering, X, get_clusters_labels, size=500):
        """
        Generate actual visualization of clusters
        Save figure
        """
        pred_clusters, pred_distances = neural_clustering.predict(X)
                
        final_clusters=get_clusters_labels(pred_clusters)
        
        
        plt.figure(figsize=(10, 10))
        plt.scatter(X[:, 0], X[:, 1], c=final_clusters)
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.savefig(os.path.join(out_dir, f"Deep_learning_clustering_result_by_hits"), dpi=500)
        
        
        
    def annotate_plot(neural_clustering, X, desc, size=500):
        """
        Annotate plot with trait categories
        Save figure
        """
        dic={} # Get numeric values for desc that can be used for color
        
        start=0
        
        for ind, trait in enumerate(desc):
            trait = ' '.join(trait.split(' ')[:2]) # select only first 2 words of description, do away of dataset name for simplicity, as long as description is the same, consider the same
            if trait in dic.keys():
                continue
            else:
                dic[trait] = start # associate numeric values to colors
                start += 1
        
        labels=[dic[' '.join(trait.split(' ')[:2])] for trait in desc] # apply previous formatting to trait to get corresponding value
        
        unique_labels = list(dic.values())
        unique_names = list(dic.keys())
    
        colors = matplotlib.colormaps['tab20'].colors # define possible colors
    
        plt.figure(figsize=(10, 10))
    
        for ind, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(X[mask, 0], X[mask, 1], c=colors[ind], label=f'Trait {unique_names[ind][:50]}', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        plt.xlabel("PC 1", fontsize=10)
        plt.ylabel("PC 2", fontsize=10, rotation=90)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(out_dir, f"Deep_learning_clustering_annotated_result_by_hits"), dpi=500)
        
     
    def predict_neural_clustering(neural_clustering, X_valid, get_clusters_labels):
        """
        Use neural networks to predict clustering on validation set
        """
        pred_clusters, pred_distances = neural_clustering.predict(X_valid)
        
        final_clusters=get_clusters_labels(pred_clusters)
        
        return final_clusters, pred_distances
     


# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()
    
    if os.path.exists('deep_learning_clustering_hits/best_clustering_model_by_hits.keras'): # check for existence of the best clustering model
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_clustering_hits/best_clustering_model_by_hits.keras')
    
    elif os.path.exists('deep_learning_clustering_hits/best_checkpoint.keras'): # check for existence of the best checkpoint
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_clustering_hits/best_checkpoint.keras')

    else:
        
        # Perform search of the best hyperparameters
        
        hyperband_tuner=kt.Hyperband(MyClusteringTaskTuning(), objective=[kt.Objective('val_output_clusters_accuracy', 'max'), kt.Objective('val_output_distances_RootMeanSquaredError', 'min')], seed=2024, max_epochs=10, factor=2, hyperband_iterations=2, overwrite=True, directory='deep_learning_clustering_hits', project_name='hyperband')
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('deep_learning_clustering_hits/best_checkpoint.keras', save_best_only=True)

        early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2) # callback to prevent overfitting
    
        tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S")) # callback for tensorboard visualization
    
        hyperband_tuner.search(X_train_features, (clusters_train, distances_train), epochs=10, validation_data=(X_valid_features, (clusters_valid, distances_valid)), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    
        top3_models=hyperband_tuner.get_best_models(num_models=3)
    
        best_model=top3_models[0] # select the best model
        
        best_model.save('deep_learning_clustering_hits/best_clustering_model_by_hits.keras') # save it
        
        
        
    actual_clustering=clustering_task.perform_neural_clustering(best_model, X_valid_features, clusters_train, clusters_valid, distances_train, distances_valid) # reperform learning using the best hyperparameters

    #Columns2Clustering.visualize_plot(actual_clustering[1], X_train_features, Columns2Clustering.get_clusters_labels)

    Columns2Clustering.visualize_plot(actual_clustering[1], X_valid_features, Columns2Clustering.get_clusters_labels) # plot validation data and label according to cluster assigned
    
    Columns2Clustering.annotate_plot(actual_clustering[1], X_valid_features, desc_valid) # plot validation data and label according to traits to which each hit pertain to

    final_clusters, pred_distances = Columns2Clustering.predict_neural_clustering(actual_clustering[1], X_valid_features, Columns2Clustering.get_clusters_labels) # extract clusters and distances to centroid



main()
