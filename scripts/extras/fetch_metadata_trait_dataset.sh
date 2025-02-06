#!/usr/bin/env bash

# Script 3b

# Extract metadata of each phenotype using trait id and dataset id and GNApi, after generation of concatenated with no duplicate BXD dataset

for i in {1..19544}; do
        dataset_id=$(cut -d, -f1 ../../processed_data/list_dataset_name_trait_id.csv | tail -n +3 | head -n $i | tail -n 1)
        trait_id=$(cut -d, -f2 ../../processed_data/list_dataset_name_trait_id.csv | tail -n +3 | head -n $i | tail -n 1)
        echo Trait id: ${trait_id} >> ../../processed_data/metadata_phenotype_info_file.json
        echo Dataset id : ${dataset_id} >> ../../processed_data/metadata_phenotype_info_file.json
        curl https://genenetwork.org/api/v_pre1/trait/${dataset_id}/${trait_id} >> ../../processed_data/metadata_phenotype_info_file.json
        echo >> ../../processed_data/metadata_phenotype_info_file.json
done

