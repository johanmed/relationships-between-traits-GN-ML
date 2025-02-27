#!/usr/bin/env python3

"""
Script 22

This script takes:
- filtered dataset: ../../../diabetes_gemma_association_data_plrt_filtered.csv
- trait list: ../../processed_data/priori_list_traits.csv
- integers specifying position of traits of interest from command line

It returns the lines of the dataset where any trait in the list is present and saves in a csv file

"""

import argparse
import pandas as pd
from math import log

def extract_data(data_file, column_name, traits):
    
    # Extract data from data with a specific value in a given column
    
    data=pd.read(data_file)
    
    cont=[]
    
    for trait in traits:
        indices=[trait.split(' ')[-1] in desc for desc in list(data[column_name])] # select lines related to traits of interest based on dataset name
        cont.append(data[indices])

    return pd.concat(cont)

if __name__ = 'main':

    priori_traits=open('../../processed_data/priori_list_traits.csv')
    trait_lists=priori_traits.read().split(',')
    priori_traits.close()

    mapping = {}

    for ind, val in enumerate(trait_lists):
        mapping[ind]=val # create mapping index to trait desc
    
    parser = argparse.ArgumentParser(description='Extract data related to traits of interest')
    
    parser.add_argument('file', type=str, help='Path to the dataset file to process')    

    parser.add_argument('--ids', type=str, help='Commma separated strings of trait index to look for in dataset')
    
    args = parser.parse_args()
    
    ids=args.trait.ids.split(',')
    
    traits=[mapping[ind] for ind in ids] # get trait desc corresponding to index provided
    
    new_data = extract_data('../../../diabetes_gemma_association_data_plrt_filtered.csv', 'full_desc', traits)
    
    pvals = list(new_data['p_lrt'])
    
    logps = [-log(p) for p in pvals] # compute -logP
    
    new_data['-logP']=logps
    
    new_data['chr']=new_data['chr_num'] # column for chromosome number with the right name
    
    new_data.to_csv('../../../diabetes_gemma_association_data_plrt_filtered_traits_selected.csv', index=False)
