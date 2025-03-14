#!/usr/bin/env python3

"""
Script 25

This script takes markers of interest

It extracts their traits if LOD >= 3 or p <= 0.05

Saves markers and (markers + trait) incrementally in different files

"""

import pandas as pd

processed_data = pd.read_csv('../../output/result_duplicated_markers.csv')

# Get markers

markers = processed_data['marker']
uniq_markers = set(markers)

new_file1 = open('../../output/markers_interest.csv', 'a')

for marker in uniq_markers:
    
    new_file1.write(f'{marker}\n') # write markers uniquely in file for manual search in database
    
new_file1.close()

# Get traits

traits = processed_data['trait']
lods = processed_data['LOD']

new_file2 = open('../../output/duplicated_markers_traits.csv', 'a')

for marker in uniq_markers:

    indices = [ind for ind, mark in enumerate(list(markers)) if mark==marker] # get indices of rows where marker appears
    
    sel_lods = list(lods.iloc[indices]) # select corresponding lods
    sel_traits = list(traits.iloc[indices]) # select corresponding traits
    
    final_traits = [sel_traits[ind] for (ind, lod) in enumerate(sel_lods) if lod >= 3] # select only traits with LODs >= 3
    final_traits = set(final_traits)
    
    new_file2.write(f'{marker}, {final_traits}\n') # write marker and related traits found overlapping

new_file2.close()
