#!/usr/bin/env python

# Script 2

# Remove duplicates from raw and concatenate dataset files into a single file, after making the file contents 100% comma separated

import os
import pandas as pd

list_csv_files=[os.path.join('../../raw_data/global_search_datasets', file) for file in os.listdir('../../raw_data/global_search_datasets') if 'trimmed' in file]

container=set()

for csv in list_csv_files:
    csv_data=pd.read_csv(csv, header=0, names=['dataset_id', 'trait_id'])
    for cols in zip(list(csv_data.iloc[:, 0]), list(csv_data.iloc[:, 1])):
        container.add(cols)

dataset_id=[]
trait_id=[]
for el in container:
    dataset_id.append(el[0])
    trait_id.append(el[1])
    
new_container={'dataset_id':dataset_id, 'trait_id':trait_id}
#print('new container is ', new_container)
#print('dataset id has ', len(new_container['dataset_id']), ' elements')
#print('trait id has ', len(new_container['trait_id']), ' elements')

final_container=pd.DataFrame(new_container)
#print(final_container.head())

final_container.to_csv('../../processed_data/list_dataset_name_trait_id.csv', header=False, index=False)


