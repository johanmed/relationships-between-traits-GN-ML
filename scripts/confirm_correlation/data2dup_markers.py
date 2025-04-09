#!/usr/bin/env python3

"""
Script 24

This script takes association data extracted for a specific pair of traits

eg -> ../../../diabetes_gemma_association_data_plrt_filtered_selected_traits_modified.csv

It looks for duplicates (shared hits) in markers column

It outputs them in a file

"""

import os
import pandas as pd
import argparse


def extract_markers_lod(data_file):

    """
    Extract markers when found for 
    """
    
    sel = pd.read_csv(data_file, usecols=['lod', 'marker', 'full_desc'])
    
    result_dup = sel['marker'].duplicated(keep=False)
    
    markers_lods =[]
    
    for ind, dup in enumerate(list(result_dup)):
        if dup: # check if dup is True
            extract = list(sel.iloc[ind, :]) # extract marker, LOD and desc
            markers_lods.append(extract) # add values to cont
            
    return markers_lods



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract overlapping markers for a specific selection of traits of interest')
    
    parser.add_argument('file', type=str, help='Path to association file for the pair of traits to process')    
    
    args = parser.parse_args()

    data_dict = {}

    data_dict['marker']=[]
    data_dict['LOD']=[] 
    data_dict['trait']=[]

    final = extract_markers_lod(args.file)
    
    for el in final:
    
        trait, lod, marker = el
        
        data_dict['marker'].append(marker)
        data_dict['LOD'].append(lod)
        data_dict['trait'].append(trait)

    data_dataf = pd.DataFrame(data_dict)
    #print(data_dataf.head())

    new_data_dataf = data_dataf.drop_duplicates()
    
  
    new_data_dataf.to_csv(f'../../output/result_markers_shared_hits.csv', index=False)
