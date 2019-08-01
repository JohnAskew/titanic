import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
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
                if os.path.exists("000-train_lowercase_cols.pickle"):
                  with open("000-train_lowercase_cols.pickle", 'rb') as in_file:
                      train = pickle.load(in_file)
                      print("Loading 000-train_lowercase_cols.pickle")
                else:
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
#Build Dataframes if fully populated records - no nans
#-------------------------------------#
print("#-------------------------------------#")
print("# Shape of dataset with no nulls (nans)")
print("#-------------------------------------#")
dataset_pure = dataset.dropna(axis = 0)
print(dataset_pure.shape)
print("#-------------------------------------#")
print("# Shape of Survived dataset with no nulls (nans)")
print("#-------------------------------------#")
dataset_pure_survived = dataset_pure[dataset_pure['survived'] ==1]
print(dataset_pure_survived.shape)
print("#-------------------------------------#")
print("# Shape of Died dataset with no nulls (nans)")
print("#-------------------------------------#")
dataset_pure_died = dataset_pure[dataset_pure['survived'] == 0]
print(dataset_pure_died.shape)
#-------------------------------------#
# Prepare a bank of good data. We
# will use this bank to derive what
# values to fill in missing data
# in the remainder of the dataset dataframe.
#-------------------------------------#
bank = dataset_pure.set_index(['pclass','embarked'])
bank.sort_index(inplace = True)
print(bank.head())
#-------------------------------------#
# Generate hint for how to populate missing AGEs
#-------------------------------------#
print("#-------------------------------------#")
print("# Hint on how to populate missing Ages using MEAN")
print("#-------------------------------------#")
print(bank.groupby(['pclass', 'embarked'])['age'].mean())
print("#-------------------------------------#")
print("# Hint on how to populate missing Ages using MEDIAN")
print("#-------------------------------------#")
print(bank.groupby(['pclass', 'embarked'])['age'].median())
#-------------------------------------#
# Start to build matrix to identify which 
#     record to extract as model for 
#     filling in missing 'age'. The output
#     is a set of index and true or false
# Example - print(dataset_pure.head(1))
#      will show the first record 
#      which corresponds to p1ecst first
#      entry is set to True. That is
#      the criteria for p1ecst is met 
#      for the first record of dataset_pure
#-------------------------------------#
p1ecst = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'C') & (dataset_pure['survived'] == 1))
p1ecsf = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'C') & (dataset_pure['survived'] == 0))
p1esst = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 1))
p1essf = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 0))
p1eqst = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'Q') & (dataset_pure['survived'] == 1))
p1eqsf = ((dataset_pure['pclass'] == 1) & (dataset_pure['embarked'] == 'Q') & (dataset_pure['survived'] == 0))
p2ecst = ((dataset_pure['pclass'] == 2) & (dataset_pure['embarked'] == 'C') & (dataset_pure['survived'] == 1))
p2ecsf = ((dataset_pure['pclass'] == 2) & (dataset_pure['embarked'] == 'C') & (dataset_pure['survived'] == 0))
p2esst = ((dataset_pure['pclass'] == 2) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 1))
p2essf = ((dataset_pure['pclass'] == 2) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 0))
p3esst = ((dataset_pure['pclass'] == 3) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 1))
p3essf = ((dataset_pure['pclass'] == 3) & (dataset_pure['embarked'] == 'S') & (dataset_pure['survived'] == 0))
#-------------------------------------#
# Extract the median age for each group.
# Example: the median age for 1st class
#     in 'S' cabin who survived is 37.
#     The median age for 1st class in 
#     'S' class who died is 47.
#-------------------------------------#
p1ecstMA = dataset_pure.loc[p1ecst, 'age'].median()
print("p1ecstMA:", p1ecstMA)
p1ecsfMA = dataset_pure.loc[p1ecsf, 'age'].median()
print("p1ecsfMA:", p1ecsfMA)
p1esstMA = dataset_pure.loc[p1esst, 'age'].median()
print("p1esstMA:", p1esstMA)
p1essfMA = dataset_pure.loc[p1essf, 'age'].median()
print("p1essfMA:", p1essfMA)
p1eqstMA = dataset_pure.loc[p1eqst, 'age'].median()
print("p1eqstMA:", p1eqstMA)
p1eqsfMA = dataset_pure.loc[p1eqsf, 'age'].median()
print("p1eqsfMA:", p1eqsfMA)
p2ecstMA = dataset_pure.loc[p2ecst, 'age'].median()
print("p2ecstMA:", p2ecstMA)
p2ecsfMA = dataset_pure.loc[p2ecsf, 'age'].median()
print("p2ecsfMA:", p2ecsfMA)
p2esstMA = dataset_pure.loc[p2esst, 'age'].median()
print("p2esstMA:", p2esstMA)
p2essfMA = dataset_pure.loc[p2essf, 'age'].median()
print("p2essfMA:", p2essfMA)
p3esstMA = dataset_pure.loc[p3esst, 'age'].median()
print("p3esstMA:", p3esstMA)
p3essfMA = dataset_pure.loc[p3essf, 'age'].median()
print("p3essfMA:", p3essfMA)
#-------------------------------------#
# Extract the cabin's using the same criteria as age.
# We are only going to extract the first alphabetic 
# cabin character and drop the room number.
#-------------------------------------#
p1ecstMC = dataset_pure.loc[p1ecst, 'cabin'].str[0].mode()
print("p1ecstMC:", p1ecstMC)
p1ecsfMC = dataset_pure.loc[p1ecsf, 'cabin'].astype(str).str[0].mode()
print("p1ecsfMC:", p1ecsfMC)
p1esstMC = dataset_pure.loc[p1esst, 'cabin'].astype(str).str[0].mode()
print("p1esstMC:", p1esstMC)
p1essfMC = dataset_pure.loc[p1essf, 'cabin'].astype(str).str[0].mode()
print("p1essfMC:", p1essfMC)
p1eqstMC = dataset_pure.loc[p1eqst, 'cabin'].astype(str).str[0].mode()
print("p1eqstMC:", p1eqstMC)
p1eqsfMC = dataset_pure.loc[p1eqsf, 'cabin'].astype(str).str[0].mode()
print("p1eqsfMC:", p1eqsfMC)
p2ecstMC = dataset_pure.loc[p2ecst, 'cabin'].astype(str).str[0].mode()
print("p2ecstMC:", p2ecstMC)
p2ecsfMC = dataset_pure.loc[p2ecsf, 'cabin'].astype(str).str[0].mode()
print("p2ecsfMC:", p2ecsfMC)
p2esstMC = dataset_pure.loc[p2esst, 'cabin'].astype(str).str[0].mode()
print("p2esstMC:", p2esstMC)
p2essfMC = dataset_pure.loc[p2essf, 'cabin'].astype(str).str[0].mode()
print("p2essfMC:", p2essfMC)
p3esstMC = dataset_pure.loc[p3esst, 'cabin'].astype(str).str[0].mode()
print("p3esstMC:", p3esstMC)
p3essfMC = dataset_pure.loc[p3essf, 'cabin'].astype(str).str[0].mode()
print("p3essfMC:", p3essfMC)
#-------------------------------------#
#Now that we have median AGE and mode Cabin,
#    let's refresh our mask using the entire
#    dataset, so we can search for missing age
#    and replace with the median age, based on
#    class, embarked and survived.
#-------------------------------------#
p1ecst = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'C') & (dataset['survived'] == 1))
p1ecsf = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'C') & (dataset['survived'] == 0))
p1esst = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'S') & (dataset['survived'] == 1))
p1essf = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'S') & (dataset['survived'] == 0))
p1eqst = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'Q') & (dataset['survived'] == 1))
p1eqsf = ((dataset['pclass'] == 1) & (dataset['embarked'] == 'Q') & (dataset['survived'] == 0))
p2ecst = ((dataset['pclass'] == 2) & (dataset['embarked'] == 'C') & (dataset['survived'] == 1))
p2ecsf = ((dataset['pclass'] == 2) & (dataset['embarked'] == 'C') & (dataset['survived'] == 0))
p2esst = ((dataset['pclass'] == 2) & (dataset['embarked'] == 'S') & (dataset['survived'] == 1))
p2essf = ((dataset['pclass'] == 2) & (dataset['embarked'] == 'S') & (dataset['survived'] == 0))
p3esst = ((dataset['pclass'] == 3) & (dataset['embarked'] == 'S') & (dataset['survived'] == 1))
p3essf = ((dataset['pclass'] == 3) & (dataset['embarked'] == 'S') & (dataset['survived'] == 0))
#-------------------------------------#
# Search dataset for missing age and replace with 
# median age from each group using the masks 
# we created
# #-------------------------------------#
# # First get counts of missing age
# #-------------------------------------#
# print("#-------------------------------------#")
# print("# First get counts of missing age")
# print("#-------------------------------------#")
# print(dataset['age'].isnull().sum())
# #-------------------------------------#
# dataset.loc[p1ecst, 'age'] = p1ecstMA
# dataset.loc[p1ecsf, 'age'] = p1ecsfMA
# dataset.loc[p1esst, 'age'] = p1esstMA
# dataset.loc[p1essf, 'age'] = p1essfMA
# dataset.loc[p1eqst, 'age'] = p1eqstMA
# dataset.loc[p1eqsf, 'age'] = p1eqsfMA
# dataset.loc[p2ecst, 'age'] = p2ecstMA
# dataset.loc[p2ecsf, 'age'] = p2ecsfMA
# dataset.loc[p2esst, 'age'] = p2esstMA
# dataset.loc[p2essf, 'age'] = p2essfMA
# dataset.loc[p3esst, 'age'] = p3esstMA
# dataset.loc[p3essf, 'age'] = p3essfMA

# #-------------------------------------#
# # After fillin age,  get counts of missing age
# #-------------------------------------#
# print("#-------------------------------------#")
# print("# After age.fillna, get counts of missing age")
# print("#-------------------------------------#")
# print(dataset['age'].isnull().sum())
#-------------------------------------#
#print(dataset.loc[p1ecst]['cabin'].astype(str).str[0].mode())
# Fill in cabin with singleton characters
#-------------------------------------#
# First get counts of missing age
#-------------------------------------#
# print("#-------------------------------------#")
# print("# First get counts of missing cabin")
# print("#-------------------------------------#")
# print(dataset['cabin'].isnull().sum())
# #-------------------------------------#
# dataset.loc[p1ecst, 'cabin'] = 'B'#p1ecstMC
# dataset.loc[p1ecsf, 'cabin'] = 'C'#p1ecsfMC
# dataset.loc[p1esst, 'cabin'] = 'C'#p1esstMC
# dataset.loc[p1essf, 'cabin'] = 'C'#p1essfMC
# dataset.loc[p1eqst, 'cabin'] = 'C'#p1eqstMC
# dataset.loc[p1eqsf, 'cabin'] = 'C'#p1eqsfMC
# dataset.loc[p2ecst, 'cabin'] = 'D'#p2ecstMC
# dataset.loc[p2ecsf, 'cabin'] = 'D'#'F'p2ecsfMC
# dataset.loc[p2esst, 'cabin'] = 'F'#p2esstMC
# dataset.loc[p2essf, 'cabin'] = 'E'#p2essfMC
# dataset.loc[p3esst, 'cabin'] = 'E'#p3esstMC
# dataset.loc[p3essf, 'cabin'] = 'F'#p3essfMC
# #-------------------------------------#
# #-------------------------------------#
# # After fillin cabin,  get counts of missing age
# #-------------------------------------#
# print("#-------------------------------------#")
# print("# After age.fillna, get counts of missing age")
# print("#-------------------------------------#")
# print(dataset['cabin'].isnull().sum())
#-------------------------------------#
# with open("000-train_prefill_age_cabin.pickle", "wb") as in_file:    #Pickle saves results as reuable object
#         pickle.dump(dataset, in_file)                     #Save results from above to Pickle.






    