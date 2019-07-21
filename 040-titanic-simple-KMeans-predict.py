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
                    print("Function: handle_non_numerical_data is mapping " ,unique, "to number:", x)
                    x +=1

            dataset[column] = list(map(convert_to_int, dataset[column]))

    return dataset
#######################################
# Start HERE with raw data
#######################################
train = dfm.get_df('train.csv')
test = dfm.get_df('test.csv')
train_test_data = [train, test]
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# HINT: test data contains nulls in age, fare and cabin columns.
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
print("#-------------------------------------#")
print("# WARNING - test dataset contains nulls")
print("#-------------------------------------#")
print(test.isna().any())


for dataset in train_test_data:
    dataset.drop(columns = ['name', 'ticket', 'parch', ], axis =1, inplace = True)
    dataset.convert_objects(convert_numeric = True)
    print(dataset.columns)
#-------------------------------------#
# Map cabin to first letter of cabin
#-------------------------------------#
for dataset in train_test_data:
     tu.map_cabin(dataset)
#-------------------------------------#
# Fill in missing values for embarked and fare
#-------------------------------------#
     tu.map_embarked(dataset)
     tu.map_fare(dataset)
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

train.to_csv('040-train-out.csv') #Just output for content review midway thru module. For checkpoint only.

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# HINT: Best Practice uses capital X for matrix data (2 dimensions) and lowercase y for the scalar array (1 dimension).
#       This would correspond to our train (training data) as "X". Lowercase 'y' will be used to keep track of
#       the surviced column we removed from 'X'. Notice we are not using cross validation, but rather comparing 
#       the train data predictions to the actual "survived" column we saved off as 'y'.
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
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

prediction = clf.predict(test)
#
## Submission
#
import pandas as pd 
submission = pd.DataFrame({
    "PassengerID":test['passengerid'],
    "Survived":prediction
    })

submission.to_csv('submission_KMeans.csv')
pd.read_csv('submission_KMeans.csv')
print(submission.sample(n=5))


