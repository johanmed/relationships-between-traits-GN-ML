#!/usr/bin/env python3

"""
Script 33

It extracts phenotype data for pair of traits of interest

Inputs:
- file with trait names used for study and in order -> ../../processed_data/modified_order_trait_names_phenotype_file.csv
- file with imputed phenotype data -> ../../processed_data/diabetes_imputed_phenotype_file.bimbam
- names of traits of interest
- pairs of old trait names and dataset names extracted

"""

import pandas as pd

from extract_oldnames_pheno import collection # import collection extracted


with open('../../processed_data/modified_order_trait_names_phenotype_file.csv') as file:

    # Read traits
    
    trait_list = file.read() # file has only 1 line
    traits = trait_list.split(',')
    
    # Extract trait full names and their indices
    
    final_names = []
    indices = []
    
    for ind, trait in enumerate(traits):
        
        for pair in collection:
        
            full_name = '_'.join(element for element in pair)
        
            if full_name in trait:
            
                indices.append(ind)
                
                final_names.append(full_name)
            
    
    # Read phenotype data and only select those corresponding to columns of interest
    
    data = pd.read_csv('../../processed_data/diabetes_imputed_phenotype_file.bimbam', header=None, index_col=False, skipfooter=1, engine='python') # leave out last BXD individual to make size match the genotype size
    
    extracted_data = pd.concat([data[ind] for ind in indices], axis=1)
    
    name_map = {old:new for (old, new) in zip(indices, final_names)}
    
    extracted_data.rename(columns=name_map, inplace=True)
    
    extracted_data.to_csv('../../output/bnw_phenotypes.csv', index=False)
