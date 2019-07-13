import pandas as pd 
from askew_utils import DF_Magic

train = DF_Magic.get_df("train.csv") #pd.read_csv('train.csv')

train["Hyp"] = 0
train.loc[train.sex == "female", "Hyp"] = 1


train["Result"] = 0
train.loc[train.survived == train["Hyp"], "Result"] = 1


print(train["Result"].value_counts(normalize = True)) # 1=701; 0=190 --> 


print(train['Hyp'].value_counts())
print("The hypothesis count came to:\n", train['Hyp'].map({0:"Died",1:("Survived")}).value_counts())
print(train['sex'].value_counts())