#!/usr/bin/env python3

"""
Script 35

Concatenates genotypes and phenotypes data

"""

import pandas as pd

geno = pd.read_csv('final_bnw_genes_genotypes.csv', index_col=False)

mod_geno = geno.iloc[:, 1:]

pheno = pd.read_csv('final_bnw_phenotypes.csv', index_col=False)

new_data = pd.concat([mod_geno, pheno], axis=1)

new_data.to_csv('../../output/bnw_dataset.csv', sep='\t', index=False)
