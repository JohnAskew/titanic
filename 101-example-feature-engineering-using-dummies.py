#!/usr/bin/env python
import os, sys
from askew_utils import DF_Magic as dfm

import pandas as pd 
train = dfm.get_df('http://bit.ly/kaggletrain')
test = dfm.get_df('http://bit.ly/kaggletest')
train_test_dataset = [train, test]


#######################################
# WHAT DF_Magic discloses in the output:
#-------------------------------------#
# Focus on "embarked"
#-------------------------#
#------ Missing Data ------
#-------------------------#
# number missing for column age: 177
# number missing for column cabin: 687
# number missing for column embarked: 2
#######################################

#-------------------------------------#
def get_df_name(df):
#-------------------------------------#
    name =[x for x in globals() if globals()[x] is df][0]
    return name  
   

#-------------------------------------#
# This paragraph tries to find all columns
# which are not numeric and are candidates
# for us feature engineering the data
# within each column. You can ignore
# this code, as it only identifies
# potential columns, it does not 
# manipulate any data
# --->Example:
#     embarked contains alphabetic data
#-------------------------------------#
for column in train.columns:
    if (train[column].dtype  != 'int64') and (train[column].dtype != 'float64'):
        print(train[column].sample(n=1))


#######################################
#S T A R T  M A I N  L O G I C  H E R E
#######################################
# Start feature engineering
# before using dummies example code.
#-------------------------------------#        
# Feature Engineer embarked.
#    We review the value_counts and find
#       the most common value for embarked
#       is 'S', so we fill in any missing
#       values with 'S'.
for dataset in train_test_dataset:
    print("#------------------------------------#")
    print("# embarked value_counts for:", get_df_name(dataset))
    print("#------------------------------------#")
    # We review the value_counts and find
    # the most common value for embarked is 
    # the value 'S'.
    print(dataset['embarked'].value_counts())
    # Fill in any missing values with 'S'.
    dataset['embarked'].fillna('S', inplace = True)



#-------------------------------------#
# Here we specify which columns are
# going to be feature engineered and 
# split out from 1 column to multiple
# columns
#-------------------------------------#
category_columns = ['sex','embarked']

for dataset in train_test_dataset:
    print("#------------------------------------#")
    print("# Feature Engineering:", get_df_name(dataset))
    print("#------------------------------------#")
    for category_column in category_columns:
        df= pd.get_dummies(dataset[category_column], prefix = category_column)
        dataset = pd.concat([dataset, df], axis = 1,)
        dataset.drop(columns = [category_column], axis = 1, inplace = True)

        print(dataset.info())


