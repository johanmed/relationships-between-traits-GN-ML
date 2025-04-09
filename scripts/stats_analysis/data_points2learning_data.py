#!/usr/bin/env python

"""
Script 20

This script uses best deep learning supervised model previously identified to predict representation label for the training set + validation set (whole dataset without test set)
This will then further be analyzed
Dependencies: vector_data_post.py -> training_validation_set
"""

import os
import pandas as pd
import tensorflow as tf
from random import choice

from vector_data_post import training_validation_set as processed_X # use concatenation of training and validation sets


# Define preprocessing pipeline and features to use

type=input('Please enter the type of model you want to use for extraction of deep learning results: ')
      
if type=='hits':
           
    X_full=processed_X[['lod', 'chr_num', 'pos']]
        
elif type=='qtl':
          
    X_full=processed_X[['lod', 'chr_num']]



class NewColumns2Clustering:

    """
    Represent supervised learning task on columns extracted
    """
    
    def __init__(self, data):
        self.data=data # use whole dataset without test set


    def get_features(self):
        
        return self.data
        
        
    def predict_labels(best_model, X):
        """
        Use neural networks to predict label of each data in the set
        """
        
        return best_model.predict(X)


def main():

    clustering_task = NewColumns2Clustering(X_full)
    
    X_features = clustering_task.get_features()
    
    type=input('Please enter the type of model you want to use for extraction of deep learning results: ')
    
    if os.path.exists(f'../machine_learning/deep_learning_supervised_{type}/best_sup_model_by_{type}.keras'): # check first existence of best_clustering_model_by hits
        
        best_model=tf.keras.models.load_model(f'../machine_learning/deep_learning_supervised_{type}/best_sup_model_by_{type}.keras')
        
        predictions = NewColumns2Clustering.predict_labels(best_model, X_features) # get predictions
        
        container=[]
        
        for (i, j) in enumerate(predictions):
        
            container.append([i, j[0]]) # save data index and label where label is a single value array
            
        to_save=pd.DataFrame(container) # convert to a dataframe
        
        to_save.to_csv(f'../../../data_indices_learning_data_{type}.csv', index=False, header=False) # save as csv
            
    elif os.path.exists(f'../machine_learning/deep_learning_supervised_{type}/best_checkpoint.keras'): # if fails, check existence of best_checkpoint
        
        best_model=tf.keras.models.load_model(f'../machine_learning/deep_learning_supervised_{type}/best_checkpoint.keras')
        
        predictions = NewColumns2Clustering.predict_labels(best_model, X_features)
        
        container=[]
        
        for (i, j) in enumerate(predictions):
        
            container.append([i, j[0]]) # save data index and label where label is a single value array
            
        to_save=pd.DataFrame(container) # convert to a dataframe
        
        to_save.to_csv(f'../../../data_indices_learning_data_{type}.csv', index=False, header=False) # save as csv
          
          
    else: # handle gracefully when none of the above exists
    
        print('Best model not found, sorry :)')
    


main()
