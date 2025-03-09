#!/usr/bin/env python3

"""
Script 22

This script takes:
- filtered dataset: ../../../diabetes_gemma_association_data_plrt_filtered.csv
- selected traits: ../../processed_data/priori_list_traits.csv
- marker annotation file: ../../processed_data/BXD_snps.txt

It returns the rows of the dataset containing pair of traits in selection 
Saves in respective csv file
"""

import argparse
import pandas as pd
from math import log


def extract_data(data_file, column_name, pair):
    
    # Extract data from data with a specific value in a given column
    
    data=pd.read_csv(data_file)
    
    cont=[]
    
    for trait in pair:
        dataset_name = trait.split(' ')[0]
        initial_ori1 = trait.split(' ')[-1][0]
        initial_pro1 = initial_ori1.lower()
        initial_ori2 = trait.split(' ')[-1][1]
        initial_pro2 = initial_ori2.lower()
        
        indices=[(dataset_name in desc) and (initial_ori1 == desc.split(' ')[0][0] or initial_pro1 == desc.split(' ')[0][0]) and (initial_ori2 == desc.split(' ')[1][0] or initial_pro2 == desc.split(' ')[1][0]) for desc in list(data[column_name])] # select lines related to traits of interest based on appearance of both dataset name and initial in trait desc
        cont.append(data[indices])
        

    return pd.concat(cont) # return only one general dataframe
    
    
    
def process_data(new_data, markers_info):
    
    # Make a series of transformations
    
    # Compute -logP
    
    pvals = list(new_data['p_lrt'])
    
    logps = [-log(p) for p in pvals] 
    
    new_data['-logP']=logps
    
    
    markers_dict={}
    
    for line in markers_info:
        marker, pos, chromo = line.split('\t')    
        chromo=chromo.strip('\n')
        if chromo=='X':
            markers_dict[(ord(chromo), int(pos))] = marker
        else:
            markers_dict[(int(chromo), int(pos))] = marker
    
    final_markers=[]
    
    for ind in new_data.index:
        row = new_data.loc[ind]
        chromo = row['chr_num']
        pos = row['pos']
        final_markers.append(markers_dict[(chromo, pos)])
    
    new_data['marker'] = final_markers
    
    
    # Add column for chromosome number with the right name
    
    ori_chr=list(new_data['chr_num'])
    
    trans_chr=[]
    
    for num in ori_chr:
        if num == 88: # encode chromosome 88 back to X
            num='X'
        trans_chr.append(num)
        
    new_data['chr']=trans_chr 
    
    
    
    return new_data
    
    
    

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Extract data related to traits of interest')
    
    parser.add_argument('file', type=str, help='Path to the dataset file to process')    
    
    parser.add_argument('--traits', type=str, help='Path to file with selected traits')
    
    parser.add_argument('--anno', type=str, help='Path to file with marker annotations')
    
    
    args = parser.parse_args()
    
    
    # Get selected traits
    
    trait_file = open(args.traits)
    full_traits = trait_file.read().split(',')
    full_traits = full_traits[:-1] # leave empty string at end
    trait_file.close()
    
    # Get markers annotations
    
    file_markers = open(args.anno)
    markers_info = file_markers.readlines()
    file_markers.close()
    
    
    for trait1 in full_traits[:]:
        for trait2 in full_traits[1:]:
            if trait1 != trait2:
                
                print(f'Processing {trait1} and {trait2}')
                
                # Extract rows of data file related to the 2 traits
                pair = [trait1, trait2]
                new_data = extract_data(args.file, 'full_desc', pair)
                
                # process the dataframe obtained
                final_data = process_data(new_data, markers_info)
                
                # Remove spacing in trait names
                new_trait1 = ''.join(word for word in trait1.split(' '))
                new_trait2 = ''.join(word for word in trait2.split(' '))
                
                # Save in corresponding csv file
                final_data.to_csv(f'../../../diabetes_gemma_association_data_plrt_filtered_selected/{new_trait1}_{new_trait2}.csv', index=False)
    
    
  
