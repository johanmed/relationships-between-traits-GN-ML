#!/usr/bin/env bash

# Script 1

# Extract only the lines of the datasets files containing comma separated values

cd ../../raw_data/

for i in *
do
	tail -n +6 $i | cut -d, -f 5,6 > trimmed_$i
done
