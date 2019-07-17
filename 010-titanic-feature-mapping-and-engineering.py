import os
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pickle
import titanic_utils as tu 
from askew_utils import DF_Magic as dfm 


sns.set()
#-------------------------------------#
def bar_chart(feature):
#-------------------------------------#
        survived = train[train['survived'] == 1][feature].value_counts()
        dead = train[train['survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['survived', 'died']
        df.plot(kind = 'bar', stacked = True, figsize = (10,15))
        plt.title("Mapped Features: Survived vs. Died wrt \"" + feature + "\" column")
        plt.show()

#######################################
# M A I N   L O G I C   S T A R T S  
#######################################
if os.path.exists("000-train_lowercase_cols.pickle"):
    with open("000-train_lowercase_cols.pickle", 'rb') as in_file:
        train = pickle.load(in_file)
        print("Loading 000-train_lowercase_cols.pickle")
else:
    train = dfm.get_df('train.csv')

#test = pd.read_csv('test.csv')
test = dfm.get_df('test.csv')
#######################################
## Title (from Name column) - clean up and standardize both training and test datasets.
#######################################
train_test_data = [train, test]
#-------------------------------------#


#######################################
## Build Title Map and set titles to either 0, 1, 2 or 3
#######################################

for dataset in train_test_data:
    tu.map_title(dataset)
    print("#------------------------------#")
    print("# Sample output of newly massaged title column")
    print("#------------------------------#")
    print(dataset[['title']].sample(n= 5))

bar_chart('title')
#######################################
## Mapping Sex
#######################################
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    #dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    tu.map_gender(dataset)
    print("#------------------------------#")
    print("# Sample output of newly massaged sex (gender) column")
    print("#------------------------------#")
    print(f'Sampling\n',dataset[['sex']].sample(n=5))
bar_chart("sex")

#
## Age Group by title and age, use median age to fill in any gaps
#

for dataset in train_test_data:
    dataset['age'].fillna(dataset.groupby("title")["age"].transform("median"), inplace = True)

#######################################
## Map out Age wrt to Survived
#######################################

for tup in [(0, train['age'].max()), (0, 20), (20,40), (40,60)]:
    facet = sns.FacetGrid(train, hue = "survived", aspect = 4, despine = True, legend_out = False)
    facet.map(sns.kdeplot, 'age', shade = True).add_legend()
    facet.set(xlim =  tup) #(0, train['age'].max()))
    plt.xlabel("Age range of: " + str(tup))
    plt.ylabel("Percentage")
    plt.title("Presenting Age wrt Survival")
    plt.show()
#
#######################################
## Manipulate Age using Bins - convert Age to Catagorial range
#######################################

for dataset in train_test_data:
    dataset.loc[dataset['age'] <= 16, 'age'] = 0, #16 and under are scored as 0.
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 26), 'age'] = 1, #17-26 scored as 1
    dataset.loc[(dataset['age'] > 26) & (dataset['age'] <= 36), 'age'] = 2, #
    dataset.loc[(dataset['age'] > 36) & (dataset['age'] <= 62), 'age'] = 3, #
    dataset.loc[dataset['age']  > 62, 'age'] =4
print("#------------------------------#")
print("# Sample output of newly massaged age column")
print("#------------------------------#")
print(train[['age']].sample(n=5))
bar_chart('age')



#######################################
## Manipulate Embarked: Present visual first then change catagories to 0, 1, or 2
#######################################
pclass1 = train[train['pclass'] == 1]['embarked'].value_counts()
pclass2 = train[train['pclass'] == 2]['embarked'].value_counts()
pclass3 = train[train['pclass'] == 3]['embarked'].value_counts()
df = pd.DataFrame([pclass1, pclass2, pclass3])
df.index = ["1st class", "2nd class", "3rd class"]
df.plot(kind = 'bar', stacked = True, figsize = (5,10))
plt.xlabel("Class")
plt.ylabel("Embarked")
plt.title("Class wrt Embarked")
plt.show()

for dataset in train_test_data:
    tu.map_embarked(dataset)

print("#------------------------------#")
print("# Sample output of newly massaged embarked column")
print("#------------------------------#")
print(train[['embarked']].sample(n=5))

#######################################
## Manipulate Fare: Fill in missing values with median of that record's pclass group, then visualize
#######################################
for dataset in train_test_data:
    tu.map_fare(dataset)
print("#------------------------------#")
print("# Sample output of newly massaged fare column")
print("#------------------------------#")
print(train[['fare']].sample(n=5))

xlimx = [(0, train['fare'].max()), (0,20), (0,50), (0,100),]
for xl in xlimx:
    facet = sns.FacetGrid(train, hue = "survived", aspect = 4).add_legend()
    facet.map(sns.kdeplot, 'fare', shade = True)
    facet.set(xlim = (xl)) #(0, train['fare'].max()))
    plt.title("Survived wrt Fare")
    plt.show()
#######################################
# skip remapping fare as it lowers our model's accuracy.
#######################################
# for dataset in train_test_data:
#     tu.map_fare2(dataset)
# print("#------------------------------#")
# print("# Sample output of second massaged fare column")
# print("#------------------------------#")
# print(train[['fare']].sample(n=5))
# bar_chart('fare')

#######################################
## Manipulate Cabin
#######################################
for dataset in train_test_data:
    tu.map_cabin(dataset)

pclass1 = train[train['pclass'] ==1]['cabin'].value_counts()
pclass2 = train[train['pclass'] ==2]['cabin'].value_counts()
pclass3 = train[train['pclass'] ==3]['cabin'].value_counts()
print("#------------------------------#")
print("# Sample output of newly massaged cabin column")
print("#------------------------------#")
print(train[['cabin']].sample(n=5))
df = pd.DataFrame([pclass1, pclass2, pclass3])
df.index = ["1st class", "2nd class", "3rd class"]
df.plot(kind = 'bar', stacked = True, figsize = (5,10))
plt.title("Class wrt Cabin")
plt.show()


###
### You may notice we have NAN in some cabin fields. This needs to change
###     in order to process data in machine learning. M/L does NOT process
###         nulls (nan).
for dataset in train_test_data:
    tu.map_cabin2(dataset)
print("#------------------------------#")
print("# Sample output of second massaged cabin column")
print("#------------------------------#")
print(train[['cabin']].sample(n=5))
bar_chart('cabin')
print("#------------------------------#")
print("# VALUE COUNTS for second massaged cabin column")
print("#------------------------------#")
print(train['cabin'].value_counts())

#######################################
## ADD new field family_size - combo of sibsp and parch
#######################################
for dataset in train_test_data:
    tu.map_familysize(dataset)
facet = sns.FacetGrid(train, hue = 'survived', aspect = 4)
facet.map(sns.kdeplot, 'familysize', shade = True)
facet.set(xlim = (0, train['familysize'].max()))
facet.add_legend()
plt.xlim(0)
plt.show()

#######################################
## MAP new field family_size 
#######################################
for dataset in train_test_data:
    tu.map_familysize2(dataset)
print("#------------------------------#")
print("# Sample output of second massaged familysize column")
print("#------------------------------#")
print(train[['familysize']].sample(n=5))
bar_chart('familysize')


#######################################
## DROP Features - this might cause other models to fail!
#######################################
## Leaving sibsp for 021-model accuracy...features_drop = ['ticket', 'sibsp', 'parch']
features_drop = ['ticket', 'name', 'parch']
for dataset in train_test_data:
    dataset = dataset.drop(features_drop, axis = 1, inplace = True)
train = train.drop(['passengerid'], axis = 1)
### Need this for 020-model....train = train.drop(['survived'], axis = 1)
print("#------------------------------#")
print("# current display of dataframe")
print("#------------------------------#")
print(train.info())


target = train['survived']
training_data = train.drop('survived', axis = 1)


with open("010-training_data_lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(training_data, in_file)                     #Save results from above to Pickle.

with open("010-target_lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(target, in_file)                     #Save results from above to Pickle.

with open("010-train_lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(train, in_file)                     #Save results from above to Pickle.

with open("010-test_lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(test, in_file)                     #Save results from above to Pickle.




