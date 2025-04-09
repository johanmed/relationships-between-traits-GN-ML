#!/usr/bin/env python
"""
Script 16a

This script contains code to performed unsupervised deep learning 
Modelling by hits (chromosome number + marker position)
Dependencies: vector_data_pre.py -> data
Need to check that max number of neurons, input shape of encoder and output shape of decoder in general_deep_learning.py are set to 3 before running this script
"""


# 1. Import data and select relevant columns

import os

from vector_data_pre import train_set as X_train
from vector_data_pre import valid_set as X_valid
from vector_data_pre import test_set as X_test

import numpy as np
import pandas as pd


X_train=X_train[['lod', 'chr_num', 'pos']]


X_valid=X_valid[['lod', 'chr_num', 'pos']]


X_test=X_test[['lod', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set



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


from general_deep_learning import MyUnsupervisedTaskTuning 


class Columns2Clustering:

    """
    Represent clustering task on 2 columns extracted from transformation pipeline
    """
    
    def __init__(self, training, validation, test):
        self.training=training
        self.validation=validation
        self.test=test
    
    
    def get_features(self):
        
        return self.training, self.validation, self.test

    def perform_neural_clustering(self, best_unsup_model):
        """
        Perform neural clustering on features columns using best model extracted from tuning
        """
        
        best_unsup_model.fit(self.training, self.training, validation_data=(self.validation, self.validation))
        
        return best_unsup_model
        
        
    def predict_repr(best_unsup_model):
        """
        Use neural networks to predict clustering on validation set
        """
        return best_unsup_model.predict(self.validation)
        
    

# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)
    
    X_train_features, X_valid_features, X_test_features = clustering_task.get_features()
    
    if os.path.exists('deep_learning_unsupervised_hits/best_unsup_model_by_hits.keras'): # check for existence of the best clustering model
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_unsupervised_hits/best_unsup_model_by_hits.keras')
    
    elif os.path.exists('deep_learning_unsupervised_hits/best_checkpoint.keras'): # check for existence of the best checkpoint
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_unsupervised_hits/best_checkpoint.keras')

    else:
        
        # Perform search of the best hyperparameters
        
        hyperband_tuner=kt.Hyperband(MyUnsupervisedTaskTuning(), objective=kt.Objective('val_accuracy', 'max'), seed=2024, max_epochs=10, factor=2, hyperband_iterations=2, overwrite=True, directory='deep_learning_unsupervised_hits', project_name='hyperband')
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('deep_learning_unsupervised_hits/best_checkpoint.keras', save_best_only=True)

        early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2) # callback to prevent overfitting
    
        tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S")) # callback for tensorboard visualization
    
        hyperband_tuner.search(X_train_features, X_train_features, epochs=10, validation_data=(X_valid_features, X_valid_features), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    
        top3_models=hyperband_tuner.get_best_models(num_models=3)
    
        best_model=top3_models[0] # select the best model
        
        best_model.save('deep_learning_unsupervised_hits/best_unsup_model_by_hits.keras') # save it
        
        
        
    actual_clustering=clustering_task.perform_neural_clustering(best_model) # reperform learning using the best hyperparameters
    
    

main()
