#!/usr/bin/env python
"""
Script 16
Add columns clusters_hits and clusters_qtl (cluster assigned) to dataset using the best classical model between Birch and Kmeans, respectively for hits and qtl modeling
Add columns distances_hits and distances_qtl (distance to the centroid of the cluster) to dataset using best classical model, respectively for hits and qtl modeling
"""

from vector_data_pre import preprocessing_hits, preprocessing_qtl # import transformation pipelines previously defined

import pandas as pd

from sklearn.pipeline import Pipeline

import joblib

import os

# 1. Read in original data again

full_X_rec=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['chr_num', 'pos', 'p_lrt'])

#print('full_X_rec looks like: \n', full_X_rec)

def predict_cluster_distance(full_X_rec, preprocessing_type, model):
    """
    Fit the same pipeline to the complete dataset
    Predict the cluster of each observation
    """
    clustering_model=Pipeline([('preprocessing_type', preprocessing_type), ('model', model)])
    
    clustering_model.fit(full_X_rec)
    
    clusters_predicted=clustering_model.predict(full_X_rec) # get clusters prediction
    
    raw_distances=clustering_model.transform(full_X_rec) # get distances to all centroids
    
    processed_distances=[max(arr_dist) for arr_dist in raw_distances] # select the distance to the closest centroid (centroid of the assigned cluster) only
    
    return clusters_predicted, processed_distances


# 2. Use best model to extract clusters

best_model=input('Please type the abbreviation of the best model you have got: ') # Get best model information from keyboard for dynamism

if os.path.exists(f'{best_model}_clustering/{best_model}_clustering_hits.pkl'): # check best model for hits

    best_model_clust=joblib.load('birch_clustering/birch_clustering_hits.pkl') # load the best model for hits
    
    clusters_hits, distances_hits=predict_cluster_distance(full_X_rec, preprocessing_hits, best_model_clust)
    
    full_X_rec['clusters_hits']=clusters_hits # add cluster information to dataframe for hits modeling
    full_X_rec['distances_hits']=distances_hits # add distance to closest centroid to dataframe for hits modeling


if os.path.exists(f'{best_model}_clustering/{best_model}_clustering_qtl.pkl'): # check the best model for QTL
    
    best_model_clust=joblib.load('birch_clustering/birch_clustering_qtl.pkl') # load the best model for QTL
    
    clusters_qtl, distances_qtl=predict_cluster_distance(full_X_rec, preprocessing_qtl, best_model_clust)
    
    full_X_rec['clusters_qtl']=clusters_qtl # add cluster information to dataframe for QTL modeling
    full_X_rec['distances_qtl']=distances_qtl # add distance to closest centroid to dataframe for QTL modeling


desc_X=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['full_desc'])

to_save=pd.concat([full_X_rec, desc_X], axis=1) # add description information 

#print('to save looks like:\n', to_save)

to_save.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_clustering_data.csv', index=False) # save the new dataset for further processing
