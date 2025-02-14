#!/usr/bin/env python

"""
Script 20

This script links association data to predictions of cluster and centroid distance

For generalization of the method, predictions are used instead of clusters and centroid distances already in the dataset

Select hits found associated at a given distance level

Compute number of hits found associated for each trait per cluster

Dependencies: 
vector_data_post.py -> training_validation_set
../../../data_indices_learning_data.csv

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

tv_data=np.array(training_validation_set) # convert to a numpy array

len_processing=len(tv_data)

for (index, line) in enumerate(tv_data):
    len_processing -= 1
    print(f'{len_processing} more to process')

    chr_num, pos, p_lrt, clusters_hits, distances_hits, clusters_qtl, distances_qtl, trait = line
        
    traits[str(index)] = trait # store only trait as prediction of cluster and distance information are already available

#print('The indices and corresponding traits are: \n', traits)



# 4. Use relationship between index and observation to get traits and predictions of distances

clust_trait_dist={}

count_traits={}

for cluster in clusters.keys():
    clust_trait_dist[cluster]=[] # register the cluster in clust_dist
    
    for ind, dist in clusters[cluster]:
    
        trait = traits[str(ind)]
        words = trait.split(' ') # process trait name
        new_trait = ' '.join(words[:3]) # select only the 4 first words in trait name
        
        clust_trait_dist[cluster].append([new_trait, dist]) # append the trait for the GWAS hit and the distance to the centroid
        
        # Initialize and keep track of the number of hits for each trait all clusters included
            
        if new_trait in count_traits.keys():
            count_traits[new_trait] += 1
        else:
            count_traits[new_trait] = 1

#print('The clusters and corresponding trait hits with distance to centroid are: \n', clust_trait_dist)




# 5. Extract traits for which hits associated at a given distance from the centroid, compute number of hits and plot results

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

type_option_levels = {'hits':{1:5, 2:5.5, 3:6, 4:6.5}, 'qtl':{1:12.8, 2:13, 3:13.2, 4:13.4, 5:13.6, 6:13.8}} # define mapping between type of model, options and distance thresholds
 

def analyze_association(clust_trait_dist, level): # level set by default to the one saved

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
        
        assoc_trait_hits = sorted(assoc_trait_hits) # sort trait alphabetically to make plots more understandable
        
        freq_assoc_traits=Counter(assoc_trait_hits) # get number of hits for each trait
        
        results[cluster]=freq_assoc_traits
        
    return results


def plot_results(results, ax, level, loc):
    
    """
    Plot number of hits found associated for each trait per cluster at a specified level on specified axis
    """
    
    results=pd.DataFrame(results) # convert to dataframe where traits are indices and clusters columns
    
    results_trans = results.transpose() # set traits to columns and clusters to indices instead

    conts=results_trans.plot.bar(ax=ax)

    ax.set_ylabel('Number of hits', fontsize=15)

    ax.set_xlabel('Clusters', fontsize=15)

    ax.legend(loc=loc)
    
    ax.set_title(f'Number of associated hits at distance threshold of {level}', fontsize=15)
    
    
    return ax
    

option_levels=type_option_levels[type]

num_levels=len(option_levels.keys())

loc=input('Please enter the desired location where you want the legend to be placed: ')

fig, axes = plt.subplots(num_levels, 1, figsize=(20, 20), sharex=True) # define subplots

for (option, ax) in zip(option_levels.keys(), axes): # repeat analysis and results plotting for each level

    results = analyze_association(clust_trait_dist, option_levels[option]) # analysis
    
    plot_results(results, ax, option_levels[option], loc=loc) # plot results on specified axis



fig.suptitle('Number of associated hits at different thresholds', fontsize=20)

plt.show()

fig.savefig(f'../../output/Results_analysis_learning_data_{type}_v1.png', dpi=500)



# 6. Plot total number of hits for each trait on the next subplot -> reference for comparison

fig, ax = plt.subplots(figsize=(20, 20))

sorted_traits = sorted(count_traits) # get list of alphabetically-sorted traits

sorted_count_traits = pd.DataFrame([count_traits[trait] for trait in sorted_traits], index=sorted_traits) # get data related to number of hits in the order of the traits as dataframe

conts = sorted_count_traits.plot.bar(ax=ax, color='black', alpha=0.7, legend=False) # plot number of hits for traits

ax.set_ylabel('Number of hits', fontsize=15)

ax.set_title('Total number of hits for each trait', fontsize=15)

plt.show()

fig.savefig(f'../../output/Total_number_hits_traits_modeling_{type}.png', dpi=500)
