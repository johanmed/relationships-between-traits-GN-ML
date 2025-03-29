#!/usr/bin/env python

"""
Script 29

Extracts genotypes of lines for genes found -> BNW analysis

Input: 
- list of markers and genes -> ../../processed_data/markers_only_genes.csv
- genotypes for markers -> ../../processed_data/diabetes_genotype_file.bimbam

"""

import pandas as pd

collection = {}

with open('../../processed_data/markers_only_genes.csv') as file1:
    
    markers_genes = file1.readlines()
    
    markers, genes = [], []
    
    for ind, pair in enumerate(markers_genes):
        
        pair = pair.split(',')
        
        marker = pair[0]
        markers.append([ind, marker])
        
        gene = pair[1:]
    
        for g in gene:
            g = g.strip('\n')
            genes.append([ind, g])
        
    with open('../../processed_data/original_BXD_genotypes_project.bimbam') as file2:
    
        geno_data = file2.readlines()
        
        ri_lines = geno_data[0]
        elements = ri_lines.split(',')
        collection['RI_lines']= elements[1:]
        
        for ind1, marker in markers:
            marker = marker.strip('\n')
            
            for line in geno_data[1:]:
                if marker in line:
                    items = line.split(',')
                    genotypes = items[1:]
                    
                    for ind2, gene in genes:
                        if ind1 == ind2 and gene not in collection.keys():
                            collection[gene] = genotypes
        
final_data = pd.DataFrame(collection)

final_data.to_csv('../../output/bnw_genes_genotypes.csv', index=False)
