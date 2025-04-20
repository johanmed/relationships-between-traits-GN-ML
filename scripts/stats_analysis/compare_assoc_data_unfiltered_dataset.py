#!/usr/bin/env python

"""
Script 13b
Store association data from non-filtered dataset in dictionary
Takes dictionary of association data
Zoom in chromosome number and marker position
Assuming they are identical, compare the p-values stored
"""

import numpy as np

def compare_info_trait(trait_pos):
    """
    Look for outliers in p_lrt values for each combination of chr_num and pos
    Return diff
    """
    
    diff=[] # store data with outlier
    
    for loc in trait_pos.keys(): # iterates through all marker positions of a given trait
        data = trait_pos[loc]
        non_valid=np.isnan(data) # check for nan values
        data=[j for i,j in enumerate(data) if non_valid[i]==False] # remove nan values
        med=np.median(data) # compute median
        q25, q75=np.percentile(data, [25, 75]) # compute Q1 and Q3
        q_range = q75 - q25 # compute interquartile range
        for datum in data:
            if not (q25 - (1.5 * q_range) <= datum <= q75 + (1.5 * q_range)): # check if datum is an outlier
                #print(data) # confirm outlier on printout
                diff.append([loc, datum]) # add to diff if outlier
    return diff
    
    

def analyze_traits(container, compare_info_trait):
    """
    Scrutinize and perform analysis of all the traits in container using compare_info_trait
    """
    assoc_diff={}
    
    for trait in container.keys(): # take one trait
        print(f'Analyzing trait {trait}...')
        trait_diff=compare_info_trait(container[trait]) # process it
    
        if len(trait_diff)>=1:
            print(f'The trait {trait} shows differences in p-lrt in at least 1 genomic position that could be statistically meaningful') # warn finding of a problematic trait
    
            assoc_diff[trait]=trait_diff # add trait and problematic locations to dictionary assoc_diff
    
    return assoc_diff
    
    


def store_assoc_data(file):
    """
    Read association info from dataset 'file' not filtered
    Store each trait and association data in dictionary 'container' for efficient lookup and replicate accumulation
    Return container
    """
    
    container={}
    
    f=open(file) # for demo
    assoc_info=f.readlines()
    f.close()
    
    to_process=assoc_info[1:] # skip first line that is file header
    
    num_processing=len(to_process)
    
    for row in to_process:
        num_processing -= 1
        print(f'{num_processing} remaining lines to be stored')
        chr_num, pos, af, beta, se, l_mle, p_lrt, full_desc=row.split(',')
        full_desc=full_desc.strip('\n')
        chr_num_pos= chr_num + ' ' + pos
        
        if full_desc in container.keys(): # check existence of trait in container
            if chr_num_pos in container[full_desc].keys(): # check existence of the location for the trait
                container[full_desc][chr_num_pos].append(float(p_lrt)) # add the p-value if yes
            else:
                container[full_desc][chr_num_pos]=[float(p_lrt)] # add a new location and initialize the array accumulating the p-values
                
        else:
            container[full_desc] = {chr_num_pos: [float(p_lrt)]} # add the trait, the location and initialize the array accumulating the p-values
            
        #print('The container is: \n', container)
    #print('The length of the container is: ', len(container))
    print('The traits in original container are: ', container.keys())
    
    return container
    
    
    
# 1. Proceed to actual storage of data from non-filtered dataset in dictionary

import os
import json

# Improve memory usage by saving results of association data processing on disk

if os.path.exists('../../../data_compare_assoc_data.json'): # check if already saved on disk

    f1=open('../../../data_compare_assoc_data.json')
    json_content=f1.read() # read data
    f1.close()
    
    dict_data=json.loads(json_content) # transform back to dictionary

else:

    dict_data=store_assoc_data('../../../diabetes_gemma_association_data.csv') # process traits and association data

    f2=open('../../../data_compare_assoc_data.json', 'w')
    dict2json_content=json.dumps(dict_data) # save dictionary as json
    f2.write(dict2json_content) # write to a file
    f2.close()

# 2. Separate non-duplicated and duplicated traits from dict_data

dup_traits={}
non_dup_traits={}

for trait in dict_data.keys():

    num_loc=len(dict_data[trait]) # get number of genomic locations saved for trait
    
    for loc in dict_data[trait]:
        num_loc -= 1 # remove 1 for each location examined
        if len(dict_data[trait][loc]) > 1: # check if location has many p-values -> duplicated trait
            dup_traits[trait] = dict_data[trait] # if yes, save the duplicated trait and all association data in dup_traits at once
            break # no need to go to another genomic position for the trait
            
    if num_loc == 0: # check if all genomic locations have been processed for the trait. If yes, and no break -> non-duplicated trait
        non_dup_traits[trait] = dict_data[trait] # save trait and all association data in non_dup_traits at once

print('The duplicated traits are: ', dup_traits.keys())

print('The non-duplicated traits are: ', non_dup_traits.keys())


# 3. Proceed to actual analysis of each duplicated trait and search for differences in p_lrt that might be relevant statistically

results=analyze_traits(dup_traits, compare_info_trait) # look for outliers in hits p-values only for duplicated traits

#print('The number of duplicated traits with dichotomy in association data: ', len(results))

print('The duplicated traits with dichotomy in association data are: ', results.keys())


# 4. Output summary of results

# Summarize
data_total=[len(dup_traits), len(dict_data)-len(dup_traits)] # get number of duplicated_traits out of the total traits
labels_total=['Traits with many occurences', 'Traits with only 1 occurence']

data_dup = [len(results), len(dup_traits) - len(results)] # get number of traits with outliers in hits p-values out of all the duplicated traits 
labels_dup = ['Traits with outliers for p-values hits', 'Traits with no outlier']

gwas_data = [(trait, len(results[trait])) for trait in results.keys()]

data_only = [len(results[trait]) for trait in results.keys()]


def display(data, labels):
    # Format printout
    for datum, label in zip(data, labels):
        print(f'Number of {label}: {datum}')

# Proceed to display

display(data_total, labels_total)

display(data_dup, labels_dup)

for trait, num_loci in gwas_data:
    print(f'The trait {trait} has {num_loci} problematic loci')


# 5. Build histogram of number of problematic markers

import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20, 20))

data_only = pd.DataFrame(data_only)

data_only[0].hist(ax=ax, color='black', alpha=0.3, grid=False, cumulative=True, density=True)

ax.set_ylabel('Frequency', fontsize=15)

ax.set_xlabel('Number of problematic markers', fontsize=15)

fig.savefig('../../output/distribution_problematic_markers.png', dpi=500)
