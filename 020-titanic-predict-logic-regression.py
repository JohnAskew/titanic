import os, pickle
import pandas as pd 
import titanic_utils as tu
from sklearn import linear_model, preprocessing
from askew_utils import DF_Magic as dfm 


if os.path.exists("010-train_lowercase_cols.pickle"):
    with open("010-train_lowercase_cols.pickle", 'rb') as in_file:
        train = pickle.load(in_file)
        print("Loading 010-train_lowercase_cols.pickle")
else:
    train = dfm.get_df('train.csv')


tu.clean_data(train)

target = train["survived"].values # Desired output, usually named target
feature_names = ["pclass", "age", "sex", "fare", "title", "sibsp", "embarked"]# adding "title" (a generated column) enhanced accuracy
features = train[feature_names].values #Hints


classifier = linear_model.LogisticRegression() # Classify which bucket we need to assign. LogisticRegressioin yields twice as linearRegression
classifier_ = classifier.fit(features, target) # Find hidden relationships in data. Goes thru every row in data for sniffing relationships

print("#------------------------------#")
print("# Accuracy BASELINE with LogisticRegression: ", end = '')
print(classifier_.score(features, target)) # Yeilds 80.2%. Adding more values to features, lowers score
print("#------------------------------#")

##################################
# Does okay job, but data lies more at at arc and not a line, so we do polynomials
##################################

poly = preprocessing.PolynomialFeatures(degree =2) # setting degree=1, returns same value as above.
poly_features = poly.fit_transform(features)


classifier_ = classifier.fit(poly_features, target)

print("\n#------------------------------#")
print("# Accuracy BASELINE with PolynomialFeatures: ",  end = '')
print(classifier_.score(poly_features, target))
print("#------------------------------#")

with open("020-train_cleaned_data-lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(train, in_file)                     #Save results from above to Pickle.


