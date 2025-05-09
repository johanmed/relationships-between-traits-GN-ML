#!/usr/bin/env python

"""
Script 13a
Summary:
This file contains code to load data from dataset BEFORE unsupervised machine learning
Data of each column are plotted in histogram to assess quality
Training, validation and test sets are defined from X data randomly
Features of the training set are scaled and validation and test set are transformed accordingly
Data of each column of the training are plotted in histogram to confirm quality
"""


# 1. Read data by chunks

import numpy as np
import pandas as pd

full_X=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['chr_num', 'pos', 'lod', 'full_desc'])

print('full_X looks like: \n', full_X)


# 2. Define training, validation and test sets

from sklearn.model_selection import train_test_split # import utility for splitting

def define_sets(X):
    X_train_valid, X_test=train_test_split(X, test_size=0.1, random_state=2024)
    X_train, X_valid=train_test_split(X_train_valid, test_size=0.1, random_state=2024)
    return X_train, X_valid, X_test
    
train_set, valid_set, test_set=define_sets(full_X)

# Get training + validation with full_desc

training_validation_set=pd.concat([train_set, valid_set]) # useful later clustering results analysis

training_set= train_set[['chr_num', 'pos', 'lod']] # select features of interest
validation_set= valid_set[['chr_num', 'pos', 'lod']]
testing_set= test_set[['chr_num', 'pos', 'lod']]

# 3. Plot histogram of training features and assess quality

import matplotlib.pyplot as plt # import plot manager
import os

out_dir=os.path.abspath('../../output/')

fig, ax=plt.subplots(figsize=(20, 10))
training_set.hist(ax=ax, color='black', alpha=0.7)
plt.show()
fig.savefig(os.path.join(out_dir, "Project_Quality_Check_Before_Transformation_v1"), dpi=500)



# 4. Extract clusters using chr_num and chr_pos just for transformation quality assessment

from sklearn.cluster import MiniBatchKMeans

def perform_clustering(X_train, X_valid, X_test):
    prelim_clustering=MiniBatchKMeans(n_clusters=5, random_state=2024, n_init=10)

    prelim_clustering1=prelim_clustering.fit_transform(np.array(X_train[['chr_num', 'pos']]).reshape(-1, 2))
    X_train[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim_clustering1

    prelim_clustering2=prelim_clustering.transform(np.array(X_valid[['chr_num', 'pos']]).reshape(-1, 2))
    X_valid[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim_clustering2
    
    prelim_clustering3=prelim_clustering.transform(np.array(X_test[['chr_num', 'pos']]).reshape(-1, 2))
    X_test[['combined_chr_num_pos1', 'combined_chr_num_pos2', 'combined_chr_num_pos3', 'combined_chr_num_pos4', 'combined_chr_num_pos5']]=prelim_clustering3


    return X_train, X_valid, X_test



clustered_training_set, clustered_validation_set, clustered_test_set=perform_clustering(training_set, validation_set, test_set)


# 5. Perform feature engineering just for transformation quality assessment

from sklearn.preprocessing import StandardScaler # import transformer

def scale(X_train, X_valid, X_test):
    std_scaler=StandardScaler()
    for i in X_train.columns:
        if i=='desc' or 'full_desc':
            continue
        else:
            std_scaler1=std_scaler.fit_transform((np.array(X_train[i])).reshape(-1, 1)) # fit transformer on training set
            X_train['transformed_'+ i]=std_scaler1
            
            std_scaler2=std_scaler.transform((np.array(X_valid[i])).reshape(-1, 1)) # fit transformer on training set
            X_valid['transformed_'+ i]=std_scaler2
            
            std_scaler3=std_scaler.transform((np.array(X_test[i])).reshape(-1, 1)) # fit transformer on training set
            X_test['transformed_'+ i]=std_scaler3
    
    return X_train, X_valid, X_test


scaled_training_set, scaled_validation_set, scaled_test_set=scale(clustered_training_set, clustered_validation_set, clustered_test_set)


# 6. Plot histogram of transformed training features and confirm quality

fig, ax=plt.subplots(figsize=(20, 20))
scaled_training_set.hist(ax=ax, color='black', alpha=0.7)
plt.show()
fig.savefig(os.path.join(out_dir, "Project_Quality_Check_After_Transformation_v1"), dpi=500)


# 7. Wrap up all transformations in a Transformer and use PCA to get 2d

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

custom_preprocessing=Pipeline([('standardize', StandardScaler()), ('reduce', PCA(n_components=2, random_state=2024))])

preprocessing_hits=ColumnTransformer([('lod_chr_num_pos', custom_preprocessing, ['lod', 'chr_num', 'pos'])], remainder=StandardScaler())

preprocessing_qtl=ColumnTransformer([('lod_chr_num', custom_preprocessing, ['lod', 'chr_num'])], remainder=StandardScaler())
