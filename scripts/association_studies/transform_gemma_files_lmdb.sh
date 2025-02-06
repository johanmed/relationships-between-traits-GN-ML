#!/usr/bin/env bash

# Script 9

# Make gemma association files compatible with gemma2lmdb script because of version of gemma used, after gemma run

cd output/ # change to directory for simplicity

for i in bxd_association_trait*.assoc.txt; do output="relevant_${i}"; csvcut -t -C 10 $i | sed 's/,/\t/g' > ${output}; done

for i in bxd_association_*.log.txt; do output="relevant_${i}"; cp $i ${output}; done
