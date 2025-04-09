#!/usr/bin/env python

"""
Script 21

This script links association data to predictions

Select hits found correlated at a given difference threshold

Only one hit is allowed for a trait at a given position

Compute number of hits found correlated for each pair of traits

Output: top 50 pairs of traits with more correlated hits

Dependencies: 
vector_data_post.py -> training_validation_set
../../../data_indices_learning_data{type}.csv

"""


# 1. Read file and import training_validation_set

type=input('Please enter the type of model you want to use for extraction of deep learning results: ')

f=open(f'../../../data_indices_learning_data_{type}.csv')
f_read=f.readlines()
f.close()

from vector_data_post import training_validation_set


# 2. Link index to label

labels = {}

for line in f_read:
    
    ind, lab = line.split(',')
    lab = lab.strip('\n')
    labels[int(ind)] = float(lab)


# 3. Extract genomic features

import numpy as np

genomic_features={}

tv_data=np.array(training_validation_set) # convert to a numpy array

len_processing=len(tv_data)

for (index, line) in enumerate(tv_data):

    print(f'{len_processing} more data to process')
    
    len_processing -= 1

    chr_num, lod, label, pos, trait = line
    
    if (chr_num, pos) in genomic_features.keys():
        
        genomic_features[(chr_num, pos)].append([labels[index], trait, lod]) # use label assigned by deep learning supervised
        
    else:
        
        genomic_features[(chr_num, pos)] = [[labels[index], trait, lod]] # store genomic features per chromosomal position



# 4. Compare labels per position

threshold = float(input('Please enter difference threshold you want to use for analysis: ')) # define difference threshold

comp_results={}

all_traits = set() # keep track of traits without duplicates

n_pos = len(genomic_features)

for pos in genomic_features.keys():
    
    print(f'{n_pos} positions to go')
        
    n_pos -= 1
        
    all_features = genomic_features[pos] # get all features
    
    for element1 in all_features:
        
        label1 = element1[0] # get label of specific
        trait1 = element1[1] # get trait of specific
    
        for element2 in all_features[1:]:
        
            label2 = element2[0]
            trait2 = element2[1]
            
            if (element1 != element2) and (abs(label1 - label2) <= threshold): # labels are float

                # Rename trait1
            
                splitted1= trait1.split(' ')
        
                splitted1_part1 = splitted1[:-1]
                splitted1_new_part1 = ''.join(word[0].upper() for word in splitted1_part1 if len(word)>=1) # build trait initials
                splitted1_part2 = splitted1[-1] # get dataset name
        
                new_trait1= splitted1_part2 + ' ' + splitted1_new_part1 # new naming
            
                all_traits.add(new_trait1)
                
                # Rename trait2
            
                splitted2= trait2.split(' ')
        
                splitted2_part1 = splitted2[:-1]
                splitted2_new_part1 = ''.join(word[0].upper() for word in splitted2_part1 if len(word)>=1) # build trait initials
                splitted2_part2 = splitted2[-1] # get dataset name
        
                new_trait2= splitted2_part2 + ' ' + splitted2_new_part1 # new naming
                
                all_traits.add(new_trait2)
                
                
                if pos in comp_results.keys():
                    
                    comp_results[pos].add(new_trait1) # append the trait for the GWAS hit
                    comp_results[pos].add(new_trait2)
                    
                else:
                    comp_results[pos] = set([new_trait1, new_trait2]) # allow only one instance
        


# 5. Extract number of shared hits for each pair of traits

shared_hits = {}

all_traits = list(all_traits)

n_traits = len(all_traits)

for trait1 in all_traits:
    
    print(f'{n_traits} more traits to go')
    
    n_traits -= 1
    
    for trait2 in all_traits[1:]:
    
        if trait1 != trait2:
        
            for pos in comp_results.keys():
            
                traits_pos = comp_results[pos]
            
                if trait1 in traits_pos and trait2 in traits_pos:
                
                    # Sort trait1 and trait2 alphabetically to prevent duplicates in plots
                    
                    if trait1 < trait2:
                
                        if (trait1, trait2) in shared_hits.keys():
                
                            shared_hits[(trait1, trait2)] += 1
                
                        else:
                
                            shared_hits[(trait1, trait2)] = 1
                            
                    else: 
                        
                        if (trait2, trait1) in shared_hits.keys():
                
                            shared_hits[(trait2, trait1)] += 1
                
                        else:
                
                            shared_hits[(trait2, trait1)] = 1
                            

# 6. Sort pairs of traits in descending order and plot number of shared hits for top 50 pairs


import pandas as pd
import matplotlib.pyplot as plt


def sort_second_el(seq): # utility function for sorting according to second element
    return seq[1]
            

sorted_shared_hits=sorted(shared_hits.items(), key=sort_second_el, reverse=True)

sorted_traits, sorted_num_hits = [], []

for (key, value) in sorted_shared_hits[:50]: # select first 50 pairs only

    trait1, trait2 = key

    sorted_traits.append(f'{trait1}\n{trait2}')
    
    sorted_num_hits.append(value)


final = pd.DataFrame(sorted_num_hits, index=sorted_traits)

fig, ax = plt.subplots(figsize=(20, 20))

final.plot.barh(ax=ax, color='black', alpha=0.7, legend=False) # plot number of hits for traits

ax.set_xlabel('Number of hits', fontsize=15)

ax.set_title(f'Number of shared hits at difference threshold of {threshold}', fontsize=20)

plt.show()

fig.savefig(f'../../output/Results_analysis_deep_learning_supervised_{type}_{threshold}.png', dpi=500)

            

