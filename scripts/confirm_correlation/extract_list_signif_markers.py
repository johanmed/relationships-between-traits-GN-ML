#!/usr/bin/env python3

"""
Script 25

This script takes duplicated markers for pair of traits of interest

It saves markers in file if LOD >= 3 or p <= 0.05 for both traits

"""

import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract list of markers of interest for specific selection of traits')
    
    parser.add_argument('file', type=str, help='Path to the file containing duplicated markers for selected traits')
    
    args = parser.parse_args()

    processed_data = pd.read_csv(args.file)

    # Get markers

    markers = processed_data['marker']
    uniq_markers = set(markers)


    # Get LODs and traits

    lods = processed_data['LOD']
    traits = processed_data['trait']
    
    trait1=input("Please confirm trait 1:")
    trait2=input("Please confirm trait 2:")
    
    with open(f'../../output/markers_interest_{trait1}_{trait2}.csv', 'a') as new_file:

        for marker in uniq_markers:
    
            indices = [ind for ind, mark in enumerate(list(markers)) if mark == marker] # get indices of rows where marker appears
    
            sel_lods = list(lods.iloc[indices]) # select corresponding lods
            sel_traits = list(traits.iloc[indices]) # select corresponding traits
    
            final_traits = [sel_traits[ind] for (ind, lod) in enumerate(sel_lods) if lod >= 3] # select traits with LODs >= 3, same traits can be repeated
    
            if len(set(final_traits)) == 2: # confirm that 2 different traits of interest met previous condition
    
                new_file.write(f'{marker}\n') # write markers uniquely in file for manual search in database
    
    
