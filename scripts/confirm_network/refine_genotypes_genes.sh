#!/usr/bin/env bash

# Script 30

# Keep only the first 237 of extracted genotypes

less ../../output/bnw_genes_genotypes.csv | head -n 237  > ../../output/final_bnw_genes_genotypes.csv
