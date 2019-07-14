import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


sns.set()
#-------------------------------------#
def bar_chart(feature):
#-------------------------------------#
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['Survived', 'Dead']
        df.plot(kind = 'bar', stacked = True, figsize = (10,15))
        plt.title("Survived vs. Died wrt " + feature)
        plt.show()

#######################################
# M A I N   L O G I C   S T A R T S  
#######################################
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#
## Title (from Name column) - clean up and standardize both training and test datasets.
#-------------------------------------#
train_test_data = [train, test]
#-------------------------------------#
#
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

print(f"Title value counts for train:\n",train['Title'].value_counts())

print(f"Title value counts for test:\n",test['Title'].value_counts())

#
## Build Title Map and set titles to either 0, 1, 2 or 3
#
title_mapping  = {"Mr":0
                , "Miss":1
                , "Mrs":2
                , "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mlle": 3, "Countess":3
                , "Ms":3, "Lady":3, "Jonkheer":3, "Don":3, "Dona":3, "Mme":3, "Capt":3, "Sir":3
                  ,
                  }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

print(train.sample(n= 5))

bar_chart('Title')


#
## Mapping Sex
#

sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
print(train.sample(n=5))
bar_chart("Sex")




