#!/usr/bin/env python3

"""
Script 26

This script takes markers of interest

And narrow down the list to the markers applicable for all traits
"""

store ={}

with open('../../output/markers_interest.csv') as markers_file:
    
    all_markers = markers_file.readlines()
    
    # Store traits associated to markers uniquely
    
    for pair in all_markers:
        
        marker, trait = pair.split(',')
        
        trait = trait.strip('\n')
        
        if marker in store.keys():
            
            store[marker].add(trait)
            
        else:
            
            store[marker]= set([trait])
            
with open('../../output/final_markers_interest.csv', 'a') as new_file:

    for key in store.keys():

        if len(store[key]) == 4:
        
            new_file.write(f'{key}\n')
