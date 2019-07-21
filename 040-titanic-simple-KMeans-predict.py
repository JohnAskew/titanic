import matplotlib.pyplot as plt 
import pandas as pd 
import csv
from matplotlib import style
style.use('ggplot')
import numpy as np 
import titanic_utils as tu 
from askew_utils import DF_Magic as dfm
from sklearn.cluster import KMeans
from sklearn import preprocessing

#-------------------------------------#
def handle_non_numerical_data(dataset):
#-------------------------------------#
    columns = dataset.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x +=1

            dataset[column] = list(map(convert_to_int, dataset[column]))

    return dataset
#######################################
# Start HERE with raw data
#######################################
train = dfm.get_df('train.csv')
test = dfm.get_df('test.csv')
train_test_data = [train, test]

for dataset in train_test_data:
    dataset.drop(columns = ['name'], axis =1, inplace = True)
    dataset.convert_objects(convert_numeric = True)
    print(dataset.columns)
#-------------------------------------#
# Map cabin to first letter of cabin
#-------------------------------------#
for dataset in train_test_data:
     tu.map_cabin(dataset)
#-------------------------------------#
# Age: Clean up Nan by looking at pclass for median age
#-------------------------------------#
for dataset in train_test_data:
    dataset['age'].fillna(dataset.groupby("pclass")["age"].transform("median"), inplace = True)

#-------------------------------------#
# Convert catagorical columns to numeric columns
#-------------------------------------#
for dataset in train_test_data:
    dataset = handle_non_numerical_data(dataset)
    #dataset.drop(columns = ['ticket'], axis = 1, inplace = True) # Needed for 1 pt accuracy

train.to_csv('040-train-out.csv')

X = np.array(train.drop(['survived'], axis = 1).astype(float)) # inplace = True))
X = preprocessing.scale(X) #Adds 20% accuracy
y = np.array(train['survived'])

clf = KMeans(n_clusters = 2) #Seems to be tuned at 2
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print("#-------------------------------------#")
print("# KMeans prediction accuracy self reports as: ", correct/len(X))
print("#-------------------------------------#")

    #labels = clf.labels_



