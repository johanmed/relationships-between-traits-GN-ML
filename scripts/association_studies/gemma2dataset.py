#!/usr/bin/env python3

# Script 11

# Create full dataset after addition of trait category and full description

# Chromosome X need to be encoded using ord('X')=88 for further processing as dataframe

import sys
import argparse
from math import log

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
            chr_num,rs,pos,miss,a1,a0,af,beta,se,l_mle,p_lrt,full_desc = line.rstrip('\n').split('\t')
            
            lod = -log(p_lrt) # transform p-values to LOD
            
            if chr_num=='chr':
                continue # ignore header
            elif (chr_num=='-9'):
                continue # ignore when chr_num=-9
            elif chr_num=='X' or chr_num=='Y':
                chr_num=ord(chr_num)
                    
            container.append([chr_num, pos, af, beta, se, l_mle, lod, full_desc])
        
                 
                 
# Turn array into dataframe

import pandas as pd

df=pd.DataFrame(container, columns=['chr_num', 'pos', 'af', 'beta', 'se', 'l_mle', 'lod', 'full_desc'])

df.to_csv('../../../diabetes_gemma_association_data2.csv', index=False, header=False) # need to save the data in chunk and omit header accordingly as all the files cant be processed and turned into a DataFrame at once
