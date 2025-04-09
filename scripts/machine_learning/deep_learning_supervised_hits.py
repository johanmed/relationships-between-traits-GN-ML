#!/usr/bin/env python
"""
Script 19a

This script contains code to perform supervised deep learning based on label extracted from autoencoder
Modelling by hits (chromosome number + marker position)
Dependencies: vector_data_post.py -> data

NB: Need to make sure input shape of ANN in general_deep_learning.py is set to the right number of dimensions expected for hits modeling -> 3
"""


# 1. Import data and select relevant columns

import os

from vector_data_post import training_set as X_train
from vector_data_post import validation_set as X_valid
from vector_data_post import test_set as X_test


import numpy as np
import pandas as pd

y_train=X_train['label']

X_train=X_train[['lod', 'chr_num', 'pos']]


y_valid=X_valid['label']

X_valid=X_valid[['lod', 'chr_num', 'pos']]


y_test=X_test['label']

X_test=X_test[['lod', 'chr_num', 'pos']]


X_train_full= pd.concat([X_train, X_valid]) # define bigger training set to train model on before going to test set

y_train_full=pd.concat([y_train, y_valid])


# 2. Select the 2 columns, do clustering and plot

import tensorflow as tf
import matplotlib.pyplot as plt # import plot manager
import matplotlib
from pathlib import Path
from time import strftime
import keras_tuner as kt

tf.keras.utils.set_random_seed(2024) # set random seed for tf, np and python

out_dir=os.path.abspath('../../output/') # define directory to save plots in


from general_deep_learning import MySupervisedTaskTuning 


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
        

    def perform_supervised_learning(self, best_model, y_train, y_valid):
        """
        Perform supervised deep learning using label extracted from autoencoder
        """
        best_model.fit(self.training, y_train, validation_data=(self.validation, y_valid))
        
        return best_model

     
    def predict_labels(best_model):
        """
        Use neural networks to predict labels on validation set
        """
        return best_model.predict(self.validation)
        
     


# Main

def main():

    clustering_task=Columns2Clustering(X_train, X_valid, X_test)

    X_train_features, X_valid_features, X_test_features=clustering_task.get_features()
    
    if os.path.exists('deep_learning_supervised_hits/best_sup_model_by_hits.keras'): # check for existence of the best clustering model
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_supervised_hits/best_sup_model_by_hits.keras')
    
    elif os.path.exists('deep_learning_supervised_hits/best_checkpoint.keras'): # check for existence of the best checkpoint
        
        print('The model has already been trained and saved on disk!')
        
        best_model=tf.keras.models.load_model('deep_learning_supervised_hits/best_checkpoint.keras')

    else:
        
        # Perform search of the best hyperparameters
        
        hyperband_tuner=kt.Hyperband(MySupervisedTaskTuning(), objective=kt.Objective('val_RootMeanSquaredError', 'min'), seed=2024, max_epochs=10, factor=2, hyperband_iterations=2, overwrite=True, directory='deep_learning_supervised_hits', project_name='hyperband')
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('deep_learning_supervised_hits/best_checkpoint.keras', save_best_only=True)

        early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=2) # callback to prevent overfitting
    
        tensorboard_cb=tf.keras.callbacks.TensorBoard(Path(hyperband_tuner.project_dir)/'tensorflow'/strftime("run_%Y_%m_%d_%H_%M_%S")) # callback for tensorboard visualization
    
        hyperband_tuner.search(X_train_features, y_train, epochs=10, validation_data=(X_valid_features, y_valid), callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
    
        top3_models=hyperband_tuner.get_best_models(num_models=3)
    
        best_model=top3_models[0] # select the best model
        
        best_model.save('deep_learning_supervised_hits/best_sup_model_by_hits.keras') # save it
        
        
        
    actual_clustering=clustering_task.perform_supervised_learning(best_model, y_train, y_valid) # reperform learning using the best hyperparameters



main()
