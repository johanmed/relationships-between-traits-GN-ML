#!/usr/bin/env python3


"""
Script 33

Apply series of modifications to phenotype_file.bimbam to make it contain the same lines as final_bnw_genes_genotypes.csv
"""

import pandas as pd
import numpy as np


# 1. Handle reading of files

# 1.1. Reformat phenotype file

f=open('new_phenotype_file.csv')

read_lines=f.readlines()

container={}

for x in read_lines:
    y=x.split(',')
    for z in y[1:]:
        u, v=z.split(':')
        container[(y[0], u)]=float(v)

row_names=set()
col_names=set()

for i in container.keys():
    a, b = i
    row_names.add(a)
    col_names.add(b)

#print('size of row_names is: ', len(row_names))
#print('size of col_names is: ', len(col_names))

ori_pheno=pd.DataFrame(np.full((len(row_names), len(col_names)), np.nan), index=list(row_names), columns=list(col_names)) 

for i in container.keys():
    a, b= i
    ori_pheno.loc[a, b]=container[i]


#print('Phenotype data looks like: \n', ori_pheno.head())


# 1.2. Read genotype file

ori_geno=pd.read_csv('final_bnw_genes_genotypes.csv', index_col=0)

#print('Column labels of transposed genotype are: \n', ori_geno.columns)




# 2. Remove lines in phenotype file not in genotype file

list_lines_pheno=ori_pheno.index # get labels
#print('Lines phenotype file: ', list_lines_pheno)

list_lines_geno=ori_geno.index # same
#print('Lines genotype file: ', list_lines_geno)

diff_to_remove=[]
for i in list_lines_pheno:
    if i not in list_lines_geno:
        diff_to_remove.append(i)
        
#print('Lines to remove: ', diff_to_remove) 

ori_pheno_trimmed=ori_pheno.drop(axis=0, labels=diff_to_remove) # remove lines not in genotype file
#print('Trimmed phenotype file: \n', ori_pheno_trimmed.head())




# 3. Add lines of genotype file missing in phenotype file

diff_to_add={}
for j in list_lines_geno:
    if j not in list_lines_pheno:
        diff_to_add[j]=[np.nan for rem in range(len(col_names))] # default number of columns to number of phenotypes in file
#print('Lines to add: ', diff_to_add.keys())
        
lines_to_add=pd.DataFrame(diff_to_add, columns=list(col_names), index=diff_to_add.keys())

total=pd.concat([ori_pheno_trimmed, lines_to_add]) # add new lines to phenotype data
#print('Complete phenotype file: \n', total.head())


# 4. Sort order of lines in phenotype file according to order in genotype file

final=pd.DataFrame()

for l in list_lines_geno:
    #print('l is: ', l)
    final[l]=total.loc[l, :] # need to select the row with the same line
    
final_transposed=final.transpose(copy=True) # need to transpose because lines on columns in dataframe final
#print('Final transposed: \n', final_transposed.head())
#print('Size of final', final_transposed.shape)


# 5. Save data

final_transposed.to_csv('final_phenotype_file.csv') # save traits and lines names with data

