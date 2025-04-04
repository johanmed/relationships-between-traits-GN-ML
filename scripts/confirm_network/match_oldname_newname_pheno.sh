#!/usr/bin/env bash

# Script 31

# Searches in metadata file : -> ../../processed_data/metadata_phenotype_info_file.json

# Extracts old name of trait based on preliminary keywords

# File argument 1: keyword related new name of trait
# File argument 2: keyword related to dataset


matches=$(grep -B 2 $1 ../../processed_data/metadata_phenotype_info_file.json | grep -B 1 $2)

echo ${matches} > ../../output/match_old_name_$1_$2.txt
    
