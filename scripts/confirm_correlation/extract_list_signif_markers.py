#!/usr/bin/env python3

"""
Script 25

This script takes duplicated markers for pair of traits of interest

It saves markers if LOD >= 3 or p <= 0.05 for both traits

"""

import pandas as pd

processed_data = pd.read_csv('../../output/result_duplicated_markers_selection_traits.csv')

# Get markers

markers = processed_data['marker']
uniq_markers = set(markers)


# Get LODs and traits

lods = processed_data['LOD']
traits = processed_data['trait']

new_file = open('../../output/markers_interest.csv', 'a')

for marker in uniq_markers:
    
    indices = [ind for ind, mark in enumerate(list(markers)) if mark == marker] # get indices of rows where marker appears
    
    sel_lods = list(lods.iloc[indices]) # select corresponding lods
    sel_traits = list(traits.iloc[indices]) # select corresponding traits
    
    final_traits = [sel_traits[ind] for (ind, lod) in enumerate(sel_lods) if lod >= 3] # select traits with LODs >= 3, same traits can be repeated
    
    if len(set(final_traits)) == 2: # confirm that 2 different traits of interest met previous condition
    
        new_file.write(f'{marker}\n') # write markers uniquely in file for manual search in database
    
new_file.close()

