import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 
from askew_utils import DF_Magic as dfm 
'''Review just the data that is complete and not missing any column data'''
#-------------------------------------#
class refresh_df:
#-------------------------------------#
    def __init__(self):
        pass

    def get_df(name):
        if name in ('train', 'test'):
            pass
        else:
             print("#--------------------------------------#")
             print("# function refresh_df does not recognize\nthe name of the dataframe you are asking for\nvalid values are: train and test")
             print("---------------------------------------#")
             return "X"
        if name == 'train':
            try:
                train = dfm.get_df('http://bit.ly/kaggletrain')
                return train
            except:
                train = dfm.get_df('train.csv')
                return train
        if name == 'test':
            try:
                test = dfm.get_df('http://bit.ly/kaggletest')
                return test
            except:
                test = dfm.get_df('test.csv')
                return test

#######################################
# Code Snippets start here
#######################################
#
## Start with fresh data
#
dataset = refresh_df.get_df('train')
#-------------------------------------#
# Find which columns are complete & which contain nan
#-------------------------------------#
print("#-------------------------------------#")
print("# Nan report by column")
print("#-------------------------------------#")
col = dataset.columns
[ print(columns, np.unique(dataset[columns].isnull())) for columns in col ]
#-------------------------------------#
# Find unique values in each column
#-------------------------------------#
print("#-------------------------------------#")
print("#Unique values report by column")
print("#-------------------------------------#")
[(print(columns, np.unique(dataset[columns].astype(str)).size)) for columns in col]
#-------------------------------------#
# Gender report
#-------------------------------------#
print("#-------------------------------------#")
print("# Gender report")
print("#-------------------------------------#")
print("Total_females = ", len(dataset[(dataset['sex'] == "female")].index))
print("Total_males = ", len(dataset[(dataset['sex'] == "male")].index))
#-------------------------------------#
# Who survived by gender
#-------------------------------------#
print("#-------------------------------------#")
print("# Survival report by gender")
print("#-------------------------------------#")
print("Total_females = ", len(dataset[(dataset['sex'] == "female") & (dataset['survived'] == 1)].index))
print("Total_males = ", len(dataset[(dataset['sex'] == "male") & (dataset['survived'] == 1)].index))
#-------------------------------------#
# Percentage report for Survived vs. Died
#-------------------------------------#
print(dataset.groupby("sex")["survived"].value_counts(normalize = True))
#-------------------------------------#
# Survival report by Passenger Class
#-------------------------------------#
print("#-------------------------------------#")
print("# Survival report by Class")
print("#-------------------------------------#")
print(dataset["pclass"].value_counts().sort_index())
print([dataset.groupby("pclass")["survived"].value_counts(normalize = True)])
#-------------------------------------#
# Correlation Matrix
#-------------------------------------#
print("#-------------------------------------#")
print("# Correlation Matrix")
print("#-------------------------------------#")
print(dataset.corr(method = 'pearson').round(3))
#-------------------------------------#
# Create AGE dataframe and Scale age down between -1 and 1
#-------------------------------------#
X_age = pd.DataFrame(dataset, columns = ['age']).copy(deep = True)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, scale
X_age['age_scaled'] = scale(X_age['age'])
X_age.drop(columns = ['age'], inplace = True)
print("#-------------------------------------#")
print("# AGE scaled between 0 and 1")
print("#-------------------------------------#")
print(X_age.sample(n=5))
#-------------------------------------#
# Display Catagorical Data
#-------------------------------------#
print("#-------------------------------------#")
print("# Catagorical features")
print("#-------------------------------------#")
print(dataset[dataset.columns[dataset.dtypes  == 'object']].describe())

# print(df_pure)
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#
#-------------------------------------#




    