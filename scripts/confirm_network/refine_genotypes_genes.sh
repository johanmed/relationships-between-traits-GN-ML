#!/usr/bin/env bash

# Script 30

# Keep only the first 237 of extracted genotypes

# Take the columns related to genes of interest passed in script (might need to modify genes in script)

less bnw_genes_genotypes.csv | head -n 237 | csvcut -d, -c RI_lines,Slco5a1,Spag16,Hecw2,Mark1,Vwc2l,C130074G19Rik,Tns1,Slc4a3,Rab23,Khdrbs2,Cfap65  > final_bnw_genes_genotypes.csv
