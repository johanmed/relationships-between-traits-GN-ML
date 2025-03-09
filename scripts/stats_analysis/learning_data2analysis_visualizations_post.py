#!/usr/bin/env python

"""
Script 21

This script is similar to script 20 except that only one instance of the trait is taken

Duplicates removed

Output: top 50 traits with more hits in the cluster
"""


# 1. Read file and import training_validation_set

type=input('Please enter the type of model you want to use for extraction of deep learning results: ')

f=open(f'../../../data_indices_learning_data_{type}.csv')
f_read=f.readlines()
f.close()

from vector_data_post import training_validation_set



# 2. Extract clusters and their elements

clusters={}

for line in f_read:

    ind, clust, dist = line.split(',')
    clust=int(clust)
    dist=float(dist.strip('\n')) # remove and of line character from dist
    
    if clust in clusters.keys():
        clusters[clust].append([ind, dist]) # add new element using its index and the distance to centroid
    else:
        clusters[clust]=[[ind, dist]] # store for each element of cluster the index and the distance to centroid
        
#print('The clusters and elements are: \n', clusters)



# 3. Link index to trait

import numpy as np
import json
import os

traits={}
seen=[]

tv_data=np.array(training_validation_set) # convert to a numpy array

len_processing=len(tv_data)

for (index, line) in enumerate(tv_data):
    len_processing -= 1
    print(f'{len_processing} more to process')

    chr_num, pos, p_lrt, clusters_hits, distances_hits, clusters_qtl, distances_qtl, trait = line
        
    if [chr_num, pos, trait] not in seen: # strategy to take care of duplicates
    
        traits[str(index)] = trait # store only trait as prediction of cluster and distance information are already available
        
        seen.append([chr_num, pos, trait]) # update seen


#print('The indices and corresponding traits are: \n', traits)



# 4. Use relationship between index and observation to get traits and predictions of distances

clust_trait_dist={}


for cluster in clusters.keys():

    clust_trait_dist[cluster]=[] # register the cluster in clust_dist
    
    for ind, dist in clusters[cluster]:

        if ind not in traits.keys():
            continue
            
        else:
        
            trait = traits[str(ind)]
            splitted= trait.split(' ')
        
            part1 = splitted[:-1]
            new_part1 = ''.join(word[0].upper() for word in part1 if len(word)>=1) # build trait initials
            part2 = splitted[-1] # get dataset name
        
            new_trait= part2 + ' ' + new_part1 # new naming
            
            clust_trait_dist[cluster].append([new_trait, dist]) # append the trait for the GWAS hit and the distance to the centroid
        
            
#print('The clusters and corresponding trait hits with distance to centroid are: \n', clust_trait_dist)




# 5. Extract traits for which hits associated at a given distance from the centroid, compute number of hits and plot results

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

type_option_levels = {'hits':{1:4, 2:5, 3:6, 4:7}, 'qtl':{1:9.0, 2:9.2, 3:9.4, 4:9.6, 5:9.8, 6:10.0}} # define mapping between type of model, options and distance thresholds
 

def sort_second_el(seq): # utility function for sorting according to second element
    return seq[1]
            
            
def analyze_association(clust_trait_dist, level, sort_second_el): # level set by default to the one saved

    """
    Analyze association results at a specified level
    The level represents the distance of the point to centroid used to defined association
    Extract traits with hits found associated at the specified level
    """
    
    results={}
    
    for cluster in clust_trait_dist.keys():
        
        assoc_trait_hits=[] # save trait of hits found associated
        traits_distances = clust_trait_dist[cluster]
        
        for (trait, dist) in traits_distances:
            
            if dist <= level: # check if distance is inferior or equal to the level set
                assoc_trait_hits.append(trait) # every occurence of trait appended represents a different hit
        
        freq_assoc_traits=Counter(assoc_trait_hits) # get number of hits for each trait
        
        sorted_freq_assoc_traits=sorted(freq_assoc_traits.items(), key=sort_second_el, reverse=True)
        
        new_freq_assoc_traits={}
        
        for key, value in sorted_freq_assoc_traits:
            new_freq_assoc_traits[key]=value
            
            
        results[cluster]=new_freq_assoc_traits
        
        results=pd.DataFrame(results) # convert to dataframe where traits are indices and clusters columns
        
    return results.iloc[:50, :] # select first 50 traits



def plot_results(results, ax, level):
    
    """
    Plot number of hits found associated for each trait per cluster at a specified level on specified axis
    """
    
    results_trans = results.transpose() # set traits to columns and clusters to indices instead

    sns.heatmap(results_trans, ax=ax, cbar=True, cmap='YlGnBu', xticklabels=True)
    
    ax.set_title(f'Number of correlated hits at distance threshold of {level}', fontsize=15)
    
    return ax
    

option_levels=type_option_levels[type]

num_levels=len(option_levels.keys())

fig, axes = plt.subplots(num_levels, 1, figsize=(20, 20), sharex=True) # define subplots

for (option, ax) in zip(option_levels.keys(), axes): # repeat analysis and results plotting for each level

    results = analyze_association(clust_trait_dist, option_levels[option], sort_second_el) # analysis
    
    plot_results(results, ax, option_levels[option]) # plot results on specified axis
    

fig.suptitle('Number of correlated hits for top 50 traits', fontsize=20)

plt.show()

fig.savefig(f'../../output/Results_analysis_learning_data_{type}_v2.png', dpi=500)



