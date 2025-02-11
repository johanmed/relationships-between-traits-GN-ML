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
    Compare p_lrt for each combination of chr_num and pos
    Return diff
    """
    
    diff=[]
    
    for loc in trait_pos.keys():
        data = trait_pos[loc]
        non_valid=np.isnan(data)
        data=[j for i,j in enumerate(data) if non_valid[i]==False]
        med=np.median(data)
        q25, q75=np.percentile(data, [25, 75])
        q_range = q75 - q25
        for datum in data:
            if not (q25 - (1.5 * q_range) <= datum <= q75 + (1.5 * q_range)):
                diff.append([loc, datum])     
    return diff
    
    

def analyze_traits(container, compare_info_trait):
    """
    Scrutinize and perform analysis of all the traits in container using compare_info_trait
    """
    assoc_diff={}
    
    for trait in container.keys():
        print(f'Analyzing trait {trait}...')
        trait_diff=compare_info_trait(container[trait])
    
        if len(trait_diff)>=1:
            print(f'The trait {trait} shows differences in p-lrt in at least 1 genomic position that could be statistically meaningful')
    
            assoc_diff[trait]=trait_diff
    
    return assoc_diff
    
    


def store_assoc_data(file):
    """
    Read association info from dataset 'file' not filtered
    Store each trait and association data in dictionary 'container' for efficient lookup
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
        
        if full_desc in container.keys():
            if chr_num_pos in container[full_desc].keys():
                container[full_desc][chr_num_pos].append(float(p_lrt))
            else:
                container[full_desc][chr_num_pos]=[float(p_lrt)]
                
        else:
            container[full_desc] = {chr_num_pos: [float(p_lrt)]}
            
        #print('The container is: \n', container)
    #print('The length of the container is: ', len(container))
    print('The traits in original container are: ', container.keys())
    
    return container
    
    
    
# 1. Proceed to actual storage of data from non-filtered dataset in dictionary

import os
import json

if os.path.exists('../../../data_compare_assoc_data.json'):

    f1=open('../../../data_compare_assoc_data.json')
    json_content=f1.read()
    f1.close()
    
    dict_data=json.loads(json_content)

else:

    dict_data=store_assoc_data('../../../diabetes_gemma_association_data.csv')

    f2=open('../../../data_compare_assoc_data.json', 'w')
    dict2json_content=json.dumps(dict_data)
    f2.write(dict2json_content)
    f2.close()

# 2. Separate non-duplicated and duplicated traits from dict_data

dup_traits={}
non_dup_traits={}

for trait in dict_data.keys():

    num_loc=len(dict_data[trait])
    
    for loc in dict_data[trait]:
        num_loc -= 1
        if len(dict_data[trait][loc]) > 1:
            dup_traits[trait] = dict_data[trait]
            break
            
    if num_loc == 0:
        non_dup_traits[trait] = dict_data[trait]

print('The duplicated traits are: ', dup_traits.keys())

print('The non-duplicated traits are: ', non_dup_traits.keys())


# 3. Proceed to actual analysis of each duplicated trait and search for differences in p_lrt that might be relevant statistically

results=analyze_traits(dup_traits, compare_info_trait)

#print('The number of duplicated traits with dichotomy in association data: ', len(results))

print('The duplicated traits with dichotomy in association data are: ', results.keys())


# 4. Plot pie chart of proportion of duplicated traits with problematic GWAS results

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
import numpy as np
from collections import Counter

np.random.seed(2024)

fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(20, 10))

# pie chart parameters

data_total=[len(dup_traits), len(dict_data)-len(dup_traits)]
labels_total=['Traits with many occurences', 'Traits with only 1 occurence']

data_dup = [len(results), len(dup_traits) - len(results)]
labels_dup = ['Traits with outliers for p-values hits', 'Traits with no outlier']

gwas_data = [(trait, len(results[trait])) for trait in results.keys()]

for trait, num_loci in gwas_data:
    print(f'The trait {trait} has {num_loci} problematic loci')


explode=(0.1, 0)


ax1.pie(data_total, labels=labels_total, colors=['b', 'g'], autopct='%1.2f%%', explode=explode, startangle=-90, textprops={'fontsize':15})

ax1.set_title('Replicated vs non-replicated traits', fontsize=20)

ax2.pie(data_dup, labels=labels_dup, colors=['r', 'g'], autopct='%1.2f%%', explode=explode, startangle=35, textprops={'fontsize':15})

ax2.set_title('GWAS results comparison locus by locus of replicated traits', fontsize=20)


fig.suptitle('Proportion of replicated traits with contradictory hits at the same locus', fontsize=25)

plt.show()

fig.savefig('../../output/Proportion_traits_with_contradictory_gwas_results.png', dpi=500)




"""
5. Plot vertical bar plot of proportion of differences for traits with contradictory gwas results

def sort_second_el_len(seq):
    return len(seq[1])

sorted_results=sorted(results.items(), key=sort_second_el_len, reverse=True)

top20_results=[pair[0] for pair in sorted_results[1:21] if type(pair[0])==str and len(pair[0]) >= 5] # omit nan

#print(top20_results)

freq_top20_results=[len(results[key]) for key in top20_results]

top20_results = [' '.join(desc.split(' ')[1:6]) for desc in top20_results]



fig, ax=plt.subplots(figsize=(20, 20))

data=pd.DataFrame(freq_top20_results, index=top20_results) # use the number of differences in each trait to determine the width of each bar

data.plot.barh(ax=ax, color='black', alpha=0.7, legend=False)

ax.set_xlabel('Number of problematic loci', fontsize=15)

ax.set_title('Number of problematic loci for the top 20 traits with contradictory results', fontsize=20)

plt.show()

fig.savefig('../../output/Number_problematic_loci_top20.png', dpi=500)

"""
