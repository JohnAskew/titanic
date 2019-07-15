import sys, os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np 
from askew_utils import DF_Magic as dfm

#-------------------------------------#
def get_df_name(df):
#-------------------------------------#
    name =[x for x in globals() if globals()[x] is df][0]
    return name  
   

#######################################
# M A I N   L O G I C   S T A R T
#######################################
dir(sys.platform)
assert ('win32' in sys.platform), "This code runs on Linux only."
try:
    if os.path.exists("010-train_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
        with open("010-train_lowercase_cols.pickle", 'rb') as in_file:
            train = pickle.load(in_file)
            print("Loading 010-train_lowercase_cols.pickle")
except:
    raise Exception ("Unable to open 010-train_lowercase_cols.pickle")
try:
    if os.path.exists("010-test_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
        with open("010-test_lowercase_cols.pickle", 'rb') as in_file:
            test = pickle.load(in_file)
            print("Loading 010-test_lowercase_cols.pickle")
except:
    raise Exception ("Unable to open 010-test_lowercase_cols.pickle")

train_test_data = [train, test]
target = train['survived']
#-------------------------------------#
# Review contents
#-------------------------------------#
for dataset in train_test_data:
    print("#-------------------------------------#")
    print("Review content for:", get_df_name(dataset))
    print("#-------------------------------------#")
    dataset.info()

#-------------------------------------#
## Cros Validation (K-fold)
#-------------------------------------#
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state = 0)
#-------------------------------------#
# KNN
#-------------------------------------#
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train, target, cv = k_fold, n_jobs = 1, scoring = scoring)
print("#-------------------------------------#")
print(score)
print(score.mean())














