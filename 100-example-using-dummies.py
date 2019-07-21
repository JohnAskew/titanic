#!/usr/bin/env python
import os, sys
from askew_utils import DF_Magic as dfm

import pandas as pd 
train = dfm.get_df('http://bit.ly/kaggletrain')
test = dfm.get_df('http://bit.ly/kaggletest')

train_test_dataset = [train, test]

for column in train.columns:
    if (train[column].dtype  != 'int64') and (train[column].dtype != 'float64'):
        print(train[column].sample(n=1))

category_columns = ['sex','embarked']

for dataset in train_test_dataset:
    for category_column in category_columns:
        df= pd.get_dummies(dataset[category_column], prefix = category_column)
        dataset = pd.concat([dataset, df], axis = 1,)
        dataset.drop(columns = [category_column], axis = 1, inplace = True)
        print("#------------------------------------#")
        print("# ", category_column, )
        print("#------------------------------------#")
        print(dataset.info())


