import os, pickle
import pandas as pd 
import titanic_utils
from askew_utils import DF_Magic as dfm 
from sklearn import tree, model_selection

try:
    if os.path.exists("010-test_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
        with open("010-test_lowercase_cols.pickle", 'rb') as in_file:
            test = pickle.load(in_file)
            print("Loading 010-test_lowercase_cols.pickle")
except:
    raise Exception ("Unable to open 010-test_lowercase_cols.pickle")


try:
    if os.path.exists("010-training_data_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
        with open("010-training_data_lowercase_cols.pickle", 'rb') as in_file:
            training_data = pickle.load(in_file)
            print("Loading 010-training_data_lowercase_cols.pickle")
except:
    raise Exception ("Unable to open 010-training_data_lowercase_cols.pickle")

try:
    if os.path.exists("010-target_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
        with open("010-target_lowercase_cols.pickle", 'rb') as in_file:
            target = pickle.load(in_file)
            print("Loading 010-target_lowercase_cols.pickle")
except:
    raise Exception ("Unable to open 010-targetlowercase_cols.pickle")

if os.path.exists("010-train_lowercase_cols.pickle"):
    with open("010-train_lowercase_cols.pickle", 'rb') as in_file:
        train = pickle.load(in_file)
        print("Loading 010-train_lowercase_cols.pickle")
else:
    train = dfm.get_df('train.csv')
    titanic_utils.clean_data(train) # We wrote this, a separate member named utils.py

feature_names = ["pclass", "age", "sex", "fare", "title", "embarked", "sibsp", "cabin", ]# parch"] # Other fields to analyze hidden patterns

features = train[feature_names].values #Hints

##
### Decision Tree with additional parameters
##

generalized_tree = tree.DecisionTreeClassifier(
    random_state = 42 #If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random
   ,max_depth =10 #7,   # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
   ,min_samples_split = 5 #  2 # The minimum number of samples required to split an internal node. Default=2
   ,max_features = 0.8
   ,min_samples_leaf = 1
   ,n_jobs = -1
    )

generalized_tree_ = generalized_tree.fit(features, target)

print("generalized_tree_1st_run", generalized_tree_.score(features, target))
scores = model_selection.cross_val_score(generalized_tree, features, target, scoring = 'accuracy', cv = 50)
print( "cross_val_scoring for generalized_tree" , scores)
print("cross_val_score MEAN for generalized_tree" ,scores.mean())



tree.export_graphviz(generalized_tree_, feature_names=feature_names, out_file="tree.dot")

clf = tree.DecisionTreeClassifier(
    random_state = 42 #If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random
   ,max_depth =10 #7,   # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
   ,min_samples_split = 20 #  2 # The minimum number of samples required to split an internal node. Default=2
   ,max_features = "auto"
   ,min_samples_leaf = 10
   ,
    )
clf.fit(training_data, target)
test_data = test.drop('passengerid', axis = 1).copy()
prediction = clf.predict(test_data)
print("#-------------------------------------#")
print("# Testing prediction")
print("#-------------------------------------#")
#print(prediction)

print("")

#
## Submission
#
import pandas as pd 
submission = pd.DataFrame({
    "PassengerID":test['passengerid'],
    "Survived":prediction
    })
submission.set_index('PassengerID', inplace = True)
submission.to_csv('submission.csv')
# pd.read_csv('submission.csv')

