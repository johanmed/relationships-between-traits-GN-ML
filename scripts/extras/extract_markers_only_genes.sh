#!/usr/bin/env bash


#Extract only markers and genes where genes are protein-coding

#Input : markers and genes annotation file -> ../../raw_data/markers2genes.csv


less ../../raw_data/markers2genes.csv | grep -v non > ../../processed_data/markers_only_genes.csv
