#!/usr/bin/env python3

"""
Script 27

This script takes:
- file containing list of overlapping markers for the pair of traits
- file containing list of markers and genes nearby

It returns the genes to use respectively for network building in a file

"""

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build tuples of traits and genes')
    
    parser.add_argument('file', type=str, help='Path to the file containing markers of interest for selected traits')
    
    args = parser.parse_args()

    markers_genes={} # initialize a dictionary to store nearby genes using markers

    # Process annotation data

    with open('../../raw_data/markers2genes_only.csv') as all_markers_file: # standard file
    
        markers_annot = all_markers_file.readlines()
    
        for annot in markers_annot:
            elements = annot.split(',')
            marker = elements[0] # get marker
            genes = elements[1:] # get genes
            str_genes = ','.join(gene.strip('\n') for gene in genes)
        
            markers_genes[marker] = str_genes # store in dictionary

    #print(markers_genes)

    list_genes=[] # initialize list for genes to save


    # Extract list of genes of interest

    with open(args.file) as extract_markers_file:
        
        extract_markers = extract_markers_file.readlines()
        
        #print(extract_markers)
    
        for the_marker in extract_markers:
            
            #print(the_marker)
        
            the_marker = the_marker.strip('\n')
            
            if the_marker in markers_genes.keys():
            
                #print('Marker found')
                
                unit = markers_genes[the_marker]
                
                if ',' in unit: # check if many genes in string
                    
                    processed_genes = unit.split(',')
                    
                    for gene in processed_genes:
                        
                        list_genes.append(gene)
                        
                else:
            
                    list_genes.append(unit)
            
    #print(list_genes)
    
    # Order used to pass the traits does not matter as they all for have LOD >= 5
    
    trait1 = input('Please enter the first trait:')
    trait2 = input('Please enter the second trait:')
    trait3 = input('Please enter the third trait:')
    trait4 = input('Please enter the fourth trait:')
    
    # Save pair of trait and gene in file for network building

    with open(f'../../output/genes_interest.csv', 'a') as traits_genes_file:
        
        list_genes = set(list_genes) # remove duplicates
        
        for gene in list_genes:
                gene = gene.strip('\n') # remove next line character
                traits_genes_file.write(f'{trait1},{gene}\n{trait2},{gene}\n{trait3},{gene}\n{trait4},{gene}\n') # write relationship between each trait of the pair and each gene of interest
            
