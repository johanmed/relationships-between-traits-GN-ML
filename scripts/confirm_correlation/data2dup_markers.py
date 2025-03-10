#!/usr/bin/env python3

"""
Script 24

This script takes all association data extracted from select_data_traits_interest.py

It looks for duplicates in markers column

It outputs them for search in literature

"""

import os
import pandas as pd

def extract_markers_lod(data_file):

    """
    Extract markers when found for 
    """
    
    sel = pd.read_csv(data_file, usecols=['-logP', 'marker', 'full_desc'])
    
    result_dup = sel['marker'].duplicated(keep=False)
    
    markers_lods =[]
    
    for ind, dup in enumerate(list(result_dup)):
        if dup: # check if dup is True
            extract = list(sel.iloc[ind, :]) # extract marker and LOD
            markers_lods.append(extract) # add marker value to cont
            
    return markers_lods
    

selected_files = [os.path.join('../../../diabetes_gemma_association_data_plrt_filtered_selected/', file) for file in os.listdir('../../../diabetes_gemma_association_data_plrt_filtered_selected/')]


data_dict = {}

data_dict['marker']=[]
data_dict['LOD']=[] 
data_dict['trait']=[]

for file in selected_files:

    final = extract_markers_lod(file)
    
    for el in final:
    
        trait, lod, marker = el
        
        data_dict['marker'].append(marker)
        data_dict['LOD'].append(lod)
        data_dict['trait'].append(trait)

data_dataf = pd.DataFrame(data_dict)
print(data_dataf.head())

new_data_dataf = data_dataf.drop_duplicates()

new_data_dataf.to_csv('../../output/result_duplicated_markers.csv', index=False)
