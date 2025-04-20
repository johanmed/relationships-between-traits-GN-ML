#!/usr/bin/env bash

#Script 25

#This script takes shared markers for selection of traits of interest

#It saves markers in file if LOD >= 3 for traits


< ../../output/result_markers_shared_hits.csv awk -F, 'NR==1 || ($2 >= 3)' | cut -d, -f 1,3 | tail -n +2 > ../../output/markers_interest.csv

