#!/usr/bin/env python3

"""
Script 32

Improves preliminary matching

Extracts oldnames of traits and dataset names from matches

"""

import os
import re

matches_files = [os.path.join('../../output/', file) for file in os.listdir('../../output') if 'match' in file] # select files with match in name

# Enter full names of datasets
dataset1 = input('Please confirm dataset 1:')
dataset2 = input('Please confirm dataset 2:')

collection = [] # store trait and dataset for complete matches

for match_file in matches_files:

    with open(match_file) as file:
        
        contents = file.read().split('--') # only 1 line in file and matches are separated by --
        
        for content in contents:
        
            new = content.split(':')
            
            new_dataset = new[-1].strip(' ')
            processed_dataset = ''.join(element for element in new_dataset if element.isalnum())
            
            new_trait = new[-2]
            
            if dataset1 in processed_dataset or dataset2 in processed_dataset: # check match between passed dataset and new

                new_trait = new_trait.strip('Dataset id:') # remove Trait id:
                collection.append([new_trait, new_dataset])

#print(collection)
