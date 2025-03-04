#!/usr/bin/env python3

"""
Script 22

This script takes:
- filtered dataset: ../../../diabetes_gemma_association_data_plrt_filtered.csv
- trait-initials specifying traits
- dataset-initials specifying datasets

Can get hints for initials in file ../../processed_data/priori_list_traits.csv

It returns the lines of the dataset where any trait in the list is present and saves in a csv file

"""

import argparse
import pandas as pd
from math import log

def extract_data(data_file, column_name, traits):
    
    # Extract data from data with a specific value in a given column
    
    data=pd.read_csv(data_file)
    
    cont=[]
    
    for trait in traits:
        dataset_name = trait.split(' ')[-1]
        initial_ori = trait.split(' ')[0][0]
        initial_pro = trait.split(' ')[0][0].lower()
        
        indices=[(dataset_name in desc) and (initial_ori == desc[0] or initial_pro == desc[0]) for desc in list(data[column_name])] # select lines related to traits of interest based on appearance of both dataset name and initial in trait desc
        cont.append(data[indices])

    return pd.concat(cont) # return only one general dataframe

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Extract data related to traits of interest')
    
    parser.add_argument('file', type=str, help='Path to the dataset file to process')    
    
    parser.add_argument('--tinitials', type=str, help='Commma separated strings of trait initials to look for in dataset')
    
    parser.add_argument('--dinitials', type=str, help='Commma separated strings of dataset initials to look for in dataset')
    
    
    args = parser.parse_args()
    
    # Get full traits names
    
    trait_initials=args.tinitials.split(',')
    
    dataset_initials = args.dinitials.split(',')
    
    full_traits = []
    
    for trait, dataset in zip(trait_initials, dataset_initials):
        full_traits.append(f'{trait} {dataset}')
        
    #print('Names of traits selected are: ', full_traits)
    
    
    new_data = extract_data(args.file, 'full_desc', full_traits)
    
    
    pvals = list(new_data['p_lrt'])
    
    logps = [-log(p) for p in pvals] # compute -logP
    
    new_data['-logP']=logps
    
    
    ori_chr=list(new_data['chr_num'])
    
    trans_chr=[]
    
    for num in ori_chr:
        if num == 88: # encode chromosome 88 back to X
            num='X'
        trans_chr.append(num)
        
    new_data['chr']=trans_chr # add column for chromosome number with the right name
    
    new_data.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_traits_selected.csv', index=False)
