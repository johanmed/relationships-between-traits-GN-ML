#!/usr/bin/env python3

"""
Script 26

This script takes:
- file containing list of overlapping markers for the pair of traits
- file containing list of markers and genes nearby

It returns the genes with the 2 traits to use respectively for network building in a file

"""

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build tuples of traits and genes')
    
    parser.add_argument('file', type=str, help='Path to the file containing markers of interest for selected traits')
    
    args = parser.parse_args()

    markers_genes={} # initialize a dictionary to store nearby genes using markers

    # Process annotation data

    with open('../../raw_data/markers2genes.csv') as all_markers_file: # standard file
        markers_annot = all_markers_file.readlines()
    
        for annot in markers_annot:
            elements = annot.split(',')
            marker = elements[0] # get marker
            genes = elements[1:] # get genes
        
            markers_genes[marker] = genes # store in dictionary

    list_genes=[] # initialize list for genes to save


    # Extract list of genes of interest

    with open(args.file) as extract_markers_file:
        extract_markers = extract_markers_file.readlines()
    
        for marker in extract_markers:
            marker = marker.strip('\n')
            if marker in markers_genes.keys():
                list_genes.append(markers_genes[marker])
            
            
    trait1 = input('Please enter the first trait of the pair being studied:')
    trait2 = input('Please enter the second trait of the pair being studied:')


    # Save pair of trait and gene in file for network building

    with open(f'../../output/genes_{trait1}_{trait2}.csv', 'a') as traits_genes_file:
    
        for genes in list_genes:
            for gene in genes:
                gene = gene.strip('\n') # remove next line character
                traits_genes_file.write(f'{trait1},{gene}\n{trait2},{gene}\n') # write relationship between each trait of the pair and each gene of interest
            
