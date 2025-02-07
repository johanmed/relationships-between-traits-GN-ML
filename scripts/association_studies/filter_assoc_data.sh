#!/usr/bin/env bash

# Script 12

< ../../../diabetes_gemma_association_data.csv awk -F, 'NR==1 || ($7 < 0.1)' > ../../../diabetes_gemma_association_data_plrt_filtered.csv
