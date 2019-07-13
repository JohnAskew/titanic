import pandas as pd 
import titanic_utils
from sklearn import linear_model, preprocessing

train = pd.read_csv("train.csv")
titanic_utils.clean_data(train)

target = train["Survived"].values # Desired output, usually named target
feature_names = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Embarked"]#, "Parch",  "Embarked"]
features = train[feature_names].values #Hints


classifier = linear_model.LogisticRegression() # Classify which bucket we need to assign. LogisticRegressioin yields twice as linearRegression
classifier_ = classifier.fit(features, target) # Find hidden relationships in data. Goes thru every row in data for sniffing relationships


print(classifier_.score(features, target)) # Yeilds 80.2%. Adding more values to features, lowers score
##################################
# Does okay job, but data lies more at at arc and not a line, so we do polynomials
##################################

poly = preprocessing.PolynomialFeatures(degree =2) # setting degree=1, returns same value as above.
poly_features = poly.fit_transform(features)


classifier_ = classifier.fit(poly_features, target)


print(classifier_.score(poly_features, target))
