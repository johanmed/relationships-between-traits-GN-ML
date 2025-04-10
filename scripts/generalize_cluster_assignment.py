#!/usr/bin/env python
"""
Script 17

Add encoder final representation to dataset respectively for hits and qtl modeling
"""

import pandas as pd

import tensorflow as tf

import os

# 1. Read in original data again

full_X_rec=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['chr_num', 'lod']) # take only chr_num and lod as best autoencoder models using qtl

#print('full_X_rec looks like: \n', full_X_rec.head())

def predict_new_repr(full_X_rec, best_model):
    """
    Predict the new representation of each observation for whole dataset
    """
    
    predictions = best_model.predict(full_X_rec) # get clusters prediction
    
    return predictions


# 2. Use best deep learning model to extract new representation


if os.path.exists('deep_learning_unsupervised_qtl/best_unsup_model_by_qtl.keras'): # check the best model for QTL
    
    best_model=tf.keras.models.load_model('deep_learning_unsupervised_qtl/best_unsup_model_by_qtl.keras') # load the best model for hits
    
    stacked_encoder = best_model.layers[0] # extract encoder part
    
    predictions=predict_new_repr(full_X_rec, stacked_encoder)
    
    full_X_rec['label']=predictions # add encoder's predictions to dataframe for hits modeling


#print('new full_X_rec:\n', full_X_rec.head())


rem_X=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['pos', 'full_desc'])

to_save=pd.concat([full_X_rec, rem_X], axis=1) # add description information 

#print('to save looks like:\n', to_save)


to_save.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_clustering_data.csv', index=False) # save the new dataset for further processing
