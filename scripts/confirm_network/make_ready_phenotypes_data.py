#!/usr/bin/env python

"""
Script 34

Impute missing values in final_phenotype_file.csv

Save in final_bnw_phenotypes.csv
"""

import pandas as pd
import numpy as np


# 1. Read phenotype data

trimmed_BXD_data=pd.read_csv('final_phenotype_file.csv', index_col=0)
trimmed_data=trimmed_BXD_data.copy()

print('trimmed data looks like \n', trimmed_data.head())


# 2. Remove traits with only 1 value before imputation

for column in trimmed_data.columns:
    count = trimmed_data[column].value_counts()
    uniq = list(count.index)
    if  len(uniq) < 2:
        trimmed_data.drop(labels=[column], axis = 1, inplace = True)
      
#print('trimmed data looks like \n', trimmed_data.head())



from sklearn.impute import KNNImputer

def impute_missing_values(dataset):
    """
    Use all features of dataset to impute missing values in columns with at least one non-missing values
    """
    imputer=KNNImputer(missing_values=np.nan) 
    new_dataset=imputer.fit_transform(dataset)
    features_out=imputer.get_feature_names_out()
    return new_dataset, features_out

    
# 3. Proceed to imputation

imputed_data, new_order_traits=impute_missing_values(trimmed_data)
imputed_BXD_data=pd.DataFrame(imputed_data, columns=trimmed_data.columns, index=trimmed_data.index)
print('Imputed data is \n', imputed_BXD_data.head())


# 4. Save new dataset

imputed_BXD_data.to_csv("final_bnw_phenotypes.csv", index=False, header=True)

