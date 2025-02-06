#!/usr/bin/env python3

# Script 11

# Create full dataset after addition of trait category and full description

"""
Might need to process by batch if input files are too many for whole data to fit in memory at once. In this case temporary files need to be processed manually to obtain a common file by removing headers. Chromosome X need to be encoded using ord('X')=88 for further processing as dataframe
"""

import sys
import argparse

# Pass association files as arguments

parser = argparse.ArgumentParser(description='Turn GEMMA assoc output into a large dataset.')
parser.add_argument('files', nargs='*', help="GEMMA file(s)")
args = parser.parse_args()


# Store association data in array

container=[]

for fn in args.files:
    print(f"Processing {fn}...")
    with open(fn) as f:
        for line in f.readlines():
            chr_num,rs,pos,miss,a1,a0,af,beta,se,l_mle,p_lrt,desc,full_desc = line.rstrip('\n').split('\t')
            
            if chr_num=='chr':
                continue # ignore header
            elif (chr_num=='-9'):
                continue # ignore when chr_num=-9
                    
            container.append([chr_num, pos, af, beta, se, l_mle, p_lrt, desc, full_desc])
        
                 
                 
# Turn array into dataframe

import pandas as pd

df=pd.DataFrame(container, columns=['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'p_lrt', 'desc', 'full_desc'])

df.to_csv('../../../project_dataset_expression_traits_with_desc_full_desc.csv', index=False)
