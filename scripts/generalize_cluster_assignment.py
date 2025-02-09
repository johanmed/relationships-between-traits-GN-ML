#!/usr/bin/env python
"""
Add a new column y to dataset for the cluster using the best classical model for clustering
"""

from vector_data_pre import preprocessing_hits, preprocessing_qtl

import pandas as pd

from sklearn.pipeline import Pipeline

import joblib

import os

# 1. Read in original data again as full_desc was dropped

full_X_rec=pd.read_csv('../../../diabetes_gemma_association_data_plrt_filtered.csv', usecols=['chr_num', 'pos', 'p_lrt', 'full_desc'])

#print('full_X_rec looks like: \n', full_X_rec)

def predict_cluster_distance(full_X_rec, preprocessing_type, model):
    """
    Fit the same pipeline to the complete dataset
    Predict the cluster of each observation
    """
    clustering_model=Pipeline([('preprocessing_type', preprocessing_type), ('model', model)])
    
    clustering_model.fit(full_X_rec)
    
    clusters_predicted=clustering_model.predict(full_X_rec)
    
    raw_distances=clustering_model.transform(full_X_rec)
    
    processed_distances=[max(arr_dist) for arr_dist in raw_distances]
    
    return clusters_predicted, processed_distances


# 2. Use best model to extract clusters

best_model=input('Please type the abbreviation of the best model you have got: ') # Get best model information from keyboard for dynamism

if os.path.exists(f'birch_clustering/{best_model}_clustering_hits.pkl'):

    best_model_clust=joblib.load('birch_clustering/birch_clustering_hits.pkl')
    
    clusters_hits, distances_hits=predict_cluster(full_X_rec, preprocessing_hits, best_model_clust)
    
    full_X_rec['clusters_hits']=clusters_hits
    full_X_rec['distances_hits']=distances_hits


if os.path.exists(f'birch_clustering/{best_model}_clustering_qtl.pkl'):
    
    best_model_clust=joblib.load('birch_clustering/birch_clustering_qtl.pkl')
    
    clusters_qtl, distances_qtl=predict_cluster(full_X_rec, preprocessing_hits, best_model_clust)
    
    full_X_rec['clusters_qtl']=clusters_qtl
    full_X_rec['distances_qtl']=distances_qtl


#print('Extended full_X_rec looks like:\n', full_X_rec)

full_X_rec.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_clustering_data.csv', index=False)
