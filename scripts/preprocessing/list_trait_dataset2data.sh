#!/usr/bin/env bash

# Script 3a

# Use trait and dataset ids and GNApi to fetch phenotype json files (phenotype data), after generation of concatenated with no duplicate BXD dataset

for i in {1..3887}; do
	dataset_id=$(cut -d, -f1 ../../processed_data/list_dataset_name_trait_id.csv | tail -n +3 | head -n $i | tail -n 1)
	trait_id=$(cut -d, -f2 ../../processed_data/list_dataset_name_trait_id.csv | tail -n +3 | head -n $i | tail -n 1)
	echo Trait id: ${trait_id} >> ../../processed_data/json_data/phenotype_file_${trait_id}_${dataset_id}.json
        echo Dataset id : ${dataset_id} >> ../../processed_data/json_data/phenotype_file_${trait_id}_${dataset_id}.json
	curl https://genenetwork.org/api/v_pre1/sample_data/${dataset_id}/${trait_id} >> ../../processed_data/json_data/phenotype_file_${trait_id}_${dataset_id}.json
done

