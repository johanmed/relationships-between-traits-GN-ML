#!/usr/bin/env python

# Script 10

# Add trait category and full description to association creating new files, after gemma files transformation


import pandas as pd
import os
import re

# 1. Read gemma files and sort

os.chdir('output/') # change to output directory for proper renaming

gemma_files=[ i for i in os.listdir() if ('assoc' in i) and ('relevant' in i) and ('new' not in i) and ('log' not in i)] # read files names from directory and store in array
gemma_files=sorted(gemma_files) # sort by names where the index after assoc is used
#print('gemma files sorted: ', gemma_files)

# 2. Read new order of trait names

order=open('../../../processed_data/modified_order_trait_names_phenotype_file.csv')
order_read=order.readline().split(',') # read contents of order file
order.close()

# 3. Read metadata

info=open('../../../processed_data/metadata_phenotype_info_file.json')
info_read=info.readlines() # read contents of metadata file
info.close()

# 4. Create a new metadata from the new order of trait names

ordered_info_read=[]

for traits in order_read:
    for ind in range(len(info_read)-7):
        data=info_read[ind]
        #print('data is: ', data)
        trait_found=re.search('Trait', data)
        if not (trait_found==None):
            trait_extract="".join(char for char in trait_found.string[9:] if char.isalnum())
            if trait_extract in traits:
                proba_desc=info_read[ind+5:ind+7] # slice where description is expected
                for desc in proba_desc:
                    description_found=re.search('description', desc)
                    if not (description_found==None):
                        ordered_info_read.append(desc)# add only the description line
                
            
#print('new metadata is: ', ordered_info_read)

# 5. Select the exact description

metadata=[] # metadata of phenotypes will be added in the order of the phenotypes in the project_imputed_phenotype_file.bimbam offhand, so no need to sort

for line in ordered_info_read:
    description_found=re.search('description', line)
    description_extract= "".join(char for char in description_found.string[15:] if char.isalnum() or char==' ')
    #print('description extract is ', description_extract)
    metadata.append(description_extract)
    
#print('final metadata is: ', metadata)


def add_desc_gemma_assoc(file, val1, val2):
        """
        Write to a new gemma file contents of old gemma file, val1 (trait category), val2(full description) tab separated
        """
        
        gemma_content=open(file).readlines()
        to_write=[]
        for u in gemma_content:
            to_write.append(f'{u.strip()}\t{val1}\t{val2}')
            
        ready_to_write='\n'.join(to_write)
        gemma_write=open(f'new_{file}', 'w')
        gemma_write.write(ready_to_write)
    
    
def process_file(metadata, gemma_files, add_desc_gemma_assoc):
    """
    Process all gemma files creating a new file for each using metadata information to infer category of trait and full trait description
    """
    for i, j in enumerate(metadata):
        for f in gemma_files:
            o, p, q, r = f.split('_')
            l, m, n = r.split('.')
            #print('num is: ', l[5:])
            if i==int(l[5:]) and ('diabetes' in j or 'diabet' in j or 'diabetic' in j or 'leptin' in j or 'insulin' in j or 'gluc' in j): # might need to add more keywords related to diabetes
                #print(f'Inferred diabetes trait for {f}')
                add_desc_gemma_assoc(f, 0, j)
            elif i==int(l[5:]) and ('immune' in j or 'immunity' in j or 'defensin' in j or 'innate' in j): # might need to add more keywords related to immune system
                #print(f'Inferred Immune system trait for {f}')
                add_desc_gemma_assoc(f, 1, j)
            elif i==int(l[5:]) and ('gut' in j or 'gastro' in j or 'gastric' in j or 'trypsin' in j): # might need to add more keywords related to gastrointestinal system
                #print(f'Inferred Gut microbiome trait for {f}')
                add_desc_gemma_assoc(f, 2, j)
            elif i==int(l[5:]): # for traits with undetermined category
                #print(f'Unable to infer category of trait for {f}')
                add_desc_gemma_assoc(f, 3, j)
                


process_file(metadata, gemma_files, add_desc_gemma_assoc)
