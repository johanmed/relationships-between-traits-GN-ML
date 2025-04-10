#!/usr/bin/env bash

# Script 31

# This script uses trait and dataset ids and GNApi to fetch phenotype json files (phenotype data)

# Need to get manually dataset ids and trait ids for 4 gene expression traits of interest and 1 classical trait

# The script will extract data for each phenotype -> BNW analysis



# For classical trait

# single trait id suffices, dataset always BXD

#echo Trait id: $1 >> json_data/phenotype_file_$1.json
#echo Dataset id : BXD >> json_data/phenotype_file_$1.json

#curl https://genenetwork.org/api/v_pre1/sample_data/bxd/$1 >> json_data/phenotype_file_$1.json



# For gene expression trait

# dataset id for arg 1 and trait id for arg 2

echo Trait id: $2 >> json_data/phenotype_file_$1_$2.json
echo Dataset id : $1 >> json_data/phenotype_file_$1_$2.json

curl https://genenetwork.org/api/v_pre1/sample_data/$1/$2 >> json_data/phenotype_file_$1_$2.json

