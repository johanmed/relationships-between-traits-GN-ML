#!/usr/bin/env python

# Script 7

# Impute missing values in project_prepared_phenotype_data.bimbam


import pandas as pd
import numpy as np

# 1. Read trait names in order

f1=open('../../processed_data/order_trait_names_phenotype_file.csv', 'r')
order_traits=f1.readline().split(',') # get an array of the trait names in the same order as the phenotype file used
#print('List of trait names is: ', order_trait_names)
#print(f'List of trait names has {len(order_trait_names)} elements')
f1.close()

# 2. Read phenotype data in right order

trimmed_BXD_data=pd.read_csv('../../processed_data/diabetes_fully_trimmed_phenotype_file.bimbam', header=None, names=order_traits)
trimmed_data=trimmed_BXD_data.copy()

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
imputed_BXD_data=pd.DataFrame(imputed_data)
#print('Imputed data is \n', imputed_BXD_data.head())
#print('New order trait names is: ', new_order_traits)
#print('Number of traits kept after modification', len(new_order_traits))

# 4. Save new dataset

imputed_BXD_data.to_csv("../../processed_data/diabetes_imputed_phenotype_file.bimbam", index=False, header=False)


# 5. Save the new order of the trait names

f2=open('../../processed_data/modified_order_trait_names_phenotype_file.csv', 'w')
f2.write(','.join(order_traits))
f2.close()
