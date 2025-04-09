#!/usr/bin/env python3

"""
Script 22

This script takes:
- filtered dataset: ../../../diabetes_gemma_association_data_plrt_filtered.csv
- marker annotation file: ../../processed_data/BXD_snps.txt
- comma separated string of traits of interest: eg -> 'UTHSCGutExL0414 LR, UTILMBXDhippNON1112 DE1, EPFLMouseLiverHFCEx0413 HNF4ADNDMTI, UMCG0907Myeloid ITFLD'

It returns the rows of the dataset containing pair of traits in selection

It saves in respective csv file

"""

import argparse
import pandas as pd


def extract_data(data_file, column_name, list_interest):
    
    # Extract data from data with a specific value in a given column
    
    data=pd.read_csv(data_file)
    
    cont=[]
    
    for trait in list_interest:
        dataset_name = trait.split(' ')[0]
        initial_ori1 = trait.split(' ')[-1][0]
        initial_pro1 = initial_ori1.lower()
        initial_ori2 = trait.split(' ')[-1][1]
        initial_pro2 = initial_ori2.lower()
        
        indices=[(dataset_name in desc) and 
        (initial_ori1 == desc.split(' ')[0][0].lower() or initial_pro1 == desc.split(' ')[0][0].lower()) and
        (initial_ori2 == desc.split(' ')[1][0].lower() or initial_pro2 == desc.split(' ')[1][0].lower()) 
        for desc in list(data[column_name])] # select lines related to traits of interest based on appearance of both dataset name and initial in trait desc
        
        cont.append(data[indices])
        

    return pd.concat(cont) # return only one general dataframe
    
    
    
def process_data(new_data, markers_info):
    
    # Make a series of transformations
    
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
        
        trans_chr.append(num)
        
    new_data['chr']=trans_chr 
    
    
    # Remove rows where lod > 50
    
    outliers = [lod>50 for lod in new_data['lod']]
    
    indices = new_data['lod'].index
    
    outliers_indices = [indices[ind] for (ind, val) in enumerate(outliers) if val == True]
    
    new_data.drop(outliers_indices, axis=0, inplace=True)
    
    return new_data.sort_values(by=['chr']) # sort chromosome numbers
    
    
    

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Extract data related to traits of interest')
    
    parser.add_argument('file', type=str, help='Path to the dataset file to process')    
    
    parser.add_argument('--anno', type=str, help='Path to file with marker annotations')
    
    parser.add_argument('--traits', type=str, help='comma separated string of traits of interest')
    
    
    args = parser.parse_args()
    
    
    # Get markers annotations
    
    file_markers = open(args.anno)
    markers_info = file_markers.readlines()
    file_markers.close()
    
    
    # Get traits of interest
    
    list_interest = args.traits.split(', ')
    
                
    # Extract rows of data file related to traits of interest
    new_data = extract_data(args.file, 'full_desc', list_interest)
                
    # process the dataframe obtained
    final_data = process_data(new_data, markers_info)
                          
    # Save in corresponding csv file
    final_data.to_csv(f'../../../diabetes_gemma_association_data_plrt_filtered_selected_traits.csv', index=False)
    
    
  
