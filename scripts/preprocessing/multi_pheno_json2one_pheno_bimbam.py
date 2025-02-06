#!/usr/bin/env python

#Script 4

# Process json phenotype files and create a single bimbam phenotype file where phenotype values of a given trait are added as a different column

import os
import re

list_json_files=[os.path.join('../../processed_data/json_files', file) for file in os.listdir('../../processed_data/json_files') if 'pheno' in file]


def process_json_bimbam(json_filename, bimbam_filename):
    """
    - Extract for a given json file, the strain names and the phenotype values
    - Add a new line with strain name and phenotype value if strain not yet in bimbam strain column
    - Add a new phenotype column when strain name already listed (strain names are not yet listed in the correct order as they are continuously modified)
    """
    
    json=open(json_filename)
    json_contents=json.readlines()
    json.close()
    
    intro1, trait=json_contents[0].split(':')
    intro2, dataset=json_contents[1].split(':')
    trait=trait.strip(' ').strip('\n')
    dataset=dataset.strip(' ').strip('\n')
    trait_dataset=trait+'_'+dataset
    
    
    bimbam2=open(bimbam_filename)
    bimbam2_contents=bimbam2.read()
    bimbam2.close()
    
        
    container=[]
    
    for (x,y) in enumerate(json_contents):
    
        strain_found=re.search('sample_name_2', y)
        
        if not (strain_found==None):
        
            value_found1=re.search('value', json_contents[x+1]) # trait value expected to be on next line
            value_found2=re.search('value', json_contents[x+2]) # or trait value on second next line
            
            strain_extract= "".join(char for char in strain_found.string[-10:-1] if char.isalnum())
            #print('strain extract is ', strain_extract)
            
            if not (value_found1==None):
                value_extract="".join(char for char in value_found1.string[-10:-1] if char.isdigit() or char=='.')
                #print('value extract is ', value_extract)
                container.append([strain_extract, trait_dataset, value_extract])
                
            elif not(value_found2==None):
                value_extract="".join(char for char in value_found2.string[-10:-1] if char.isdigit() or char=='.')
                #print('value extract is ', value_extract)
                container.append([strain_extract, trait_dataset, value_extract])
    
    #print('container is ', container)
    
    for a in container:
        if a[0] in bimbam2_contents:
            
            bimbam2=open(bimbam_filename)
            bimbam2_contents=bimbam2.read()
            bimbam2.close()
    
            bimbam1=open(bimbam_filename, 'w')
            
            old_string=(re.search(f'{a[0]}.*\n', bimbam2_contents).group())[:-1]
            #print('old string is ', old_string)
            new_string=bimbam2_contents.replace(old_string, f'{old_string}, {a[1]}:{a[2]}')
            #print('new string is ', new_string)
            
            bimbam1.write(f'{new_string}')
            bimbam1.close()
            
        else:
            
            bimbam1=open(bimbam_filename, 'a')
            bimbam1.write(f'{a[0]}, {a[1]}:{a[2]}\n')
            bimbam1.close()
    
    
    
def contains_phenotype_data(json_filename):
    """
    Checks if a phenotype file contains phenotype data or not
    """
    
    f=open(json_filename)
    line1=f.readline()
    line2=f.readline()
    line3=f.readline()
    f.close()
    if line3.startswith('['):
        return True
    else:
        return False
        


count_remaining=len(list_json_files)

for jsonf in list_json_files:
    print(f'Processing {jsonf}')
    
    count_remaining -=1
    print(f'{count_remaining} still need to be processed')
    
    if contains_phenotype_data(jsonf):
        print(f'File {jsonf} contains data and can go for actual processing')
        process_json_bimbam(jsonf, '../../processed_data/project_phenotype_file.bimbam')
    
    
    
