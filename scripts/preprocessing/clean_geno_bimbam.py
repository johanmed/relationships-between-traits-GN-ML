#!/usr/bin/env python

# Script 5

# Apply some series of modifications to transpose genotype file and format it as rows=markers and columns=lines as expected by gemma

import pandas as pd
import numpy as np
import re

ori_geno=pd.read_csv('../../processed_data/original_BXD_genotypes_project.bimbam', index_col=0) # read in data from appropriate file
#print('ori geno looks like \n', ori_geno.head())
#print('shape before', ori_geno.shape)

ori_geno_no_lines=ori_geno.iloc[1:, :] # remove the row of RI lines
#print('shape after', transposed_geno_no_lines.shape)

al1=pd.DataFrame(np.full(ori_geno_no_lines.shape[0], 'X'), index=list(ori_geno_no_lines.index))
al2=pd.DataFrame(np.full(ori_geno_no_lines.shape[0], 'Y'), index=list(ori_geno_no_lines.index))

final_geno=pd.concat([al1, al2, ori_geno_no_lines], axis=1)

#print('Final genotype data is: \n', final_geno.head())


final_geno.to_csv('../../processed_data/diabetes_genotype_file.bimbam', header=False) # save dataframe



