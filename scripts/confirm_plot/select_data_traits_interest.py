#!/usr/bin/env python3

"""
Script 22

This script takes:
- filtered dataset: ../../../diabetes_gemma_association_data_plrt_filtered.csv
- trait list: ../../processed_data/priori_list_traits.csv
- integers specifying position of traits of interest from command line; when not provided, '5,6' is the default used

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
        initial = trait.split(' ')[0][0].lower()
        
        indices=[(dataset_name in desc) and (initial == desc[0]) for desc in list(data[column_name])] # select lines related to traits of interest based on appearance of both dataset name and initial in trait desc
        cont.append(data[indices])

    return pd.concat(cont) # return only one general dataframe

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Extract data related to traits of interest')
    
    parser.add_argument('file', type=str, help='Path to the dataset file to process')    
    
    parser.add_argument('--traitfile', type=str, help='Path to trait list file')
    
    parser.add_argument('--ids', type=str, default='5,6', help='Commma separated strings of trait index to look for in dataset') # file generated using defaults trait ids
    
    args = parser.parse_args()
    
    # Get traits names
    
    priori_traits=open(args.traitfile)
    trait_lists=priori_traits.read().split(',')
    priori_traits.close()

    mapping = {}

    for ind, val in enumerate(trait_lists):
        mapping[ind]=val # create mapping index to trait desc
    
    
    ids=args.ids.split(',')
    
    traits=[mapping[int(ind)] for ind in ids] # get trait desc corresponding to index provided
    #print('Names of traits selected are: ', traits)
    
    new_data = extract_data(args.file, 'full_desc', traits)
    
    pvals = list(new_data['p_lrt'])
    
    logps = [-log(p) for p in pvals] # compute -logP
    
    new_data['-logP']=logps
    
    new_data['chr']=new_data['chr_num'] # column for chromosome number with the right name
    
    new_data.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_traits_selected.csv', index=False)
