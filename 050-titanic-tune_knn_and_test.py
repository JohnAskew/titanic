import sys, os
import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic as dfm

#-------------------------------------#
# V A R I A B L E S  T O  C O N T R O L
#-------------------------------------#
nneighbors = 5      #n_neighbors is the number of nearest neighbors to consider
njobs = -1          #n_jobs is the number of parallel jobs
p = 1
algorithm = 'auto'
weights = 'distance'

xor = 'accuracy'    #score is the parameter for scoring, such as 'precision'
nsplits = 10        #n_splits is number of splits for decisions
shffl = True        #shuffle - refer to python.org documentation for details
random = 42         #random_state - machine learning started seed.

nneighbors_options = [2, 5, 10, 13, 15, 20]
njobs_options = [-1, 1, 2, 5]
p_options = [1, 2]
algorithm_options = ['ball_tree','kd_tree','brute','auto']
weights_options = ['distance', 'uniform']



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

train_test_data = [training_data, test]

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


#--------------------------------------#
# DEFINE K_FOLD
#--------------------------------------#
k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)

#######################################
# K   K   N     N   N     N
# K  K    NN    N   NN    N
# K K     N N   N   N N   N
# K       N  N  N   N  N  N
# K K     N   N N   N   N N
# K  k    N    NN   N    NN
# k   k   N     N   N     N
#######################################
#
## KNN tuning n_neighbors
#
results = []
for nneighbors_option in nneighbors_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors_option,
                               n_jobs = njobs,
                               p = p,
                               algorithm = algorithm,
                               weights = weights,
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN of" ,nneighbors_option, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN of",nneighbors_option, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")

pd.Series(results, nneighbors_options).plot()
plt.title('KNN Varying n_neighbors results')
plt.xlabel('Number of n_neighbors')
plt.ylabel('Accuracy')
plt.show()
#
## KNN tuning n_jobs
#
results = []
for njobs_option in njobs_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors,
                               n_jobs = njobs_option,
                               p=p,
                               algorithm = algorithm,
                               weights = weights,
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN of" ,njobs_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN of",njobs_option, "MEAN score:" ,mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('KNN Varying n_jobs results')
plt.xlabel('Number of n_jobs')
plt.ylabel('Accuracy')
plt.show()
#
## KNN tuning p
#
results = []
for p_option in p_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors,
                               n_jobs = njobs,
                               p=p_option,
                               algorithm = algorithm,
                               weights = weights,
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN P variable of" ,p, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN P variable of",p ,"MEAN score:" ,mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, p_options).plot()
plt.title('KNN P variable results')
plt.xlabel('P')
plt.ylabel('Accuracy')
plt.show()
#
## KNN tuning algorithm
#
results = []
for algorithm_option in algorithm_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors,
                               n_jobs = njobs,
                               p=p,
                               algorithm = algorithm_option,
                               weights = weights,
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN of" ,algorithm_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN of",algorithm_option ,"MEAN score:" ,mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, algorithm_options).plot()
plt.title('KNN algorithm_options results')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.show()
#
## KNN tuning weights
#
results = []
for weights_option in weights_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors,
                               n_jobs = njobs,
                               p=p,
                               algorithm = algorithm,
                               weights = weights_option,
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN of" ,weights_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN of",weights_option ,"MEAN score:" ,mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, weights_options).plot()
plt.title('KNN weights_options results')
plt.xlabel('Weights')
plt.ylabel('Accuracy')
plt.show()

## Testing

clf = KNeighborsClassifier(n_neighbors = nneighbors,
                               n_jobs = njobs,
                               p = p,
                               algorithm = algorithm,
                               weights = weights,
                               )
# score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor)
Z = clf.fit(training_data, target)
print("Z Fit:", Z)
test_data = test.drop('passengerid', axis = 1).copy()
prediction = clf.predict(test_data)
print("#-------------------------------------#")
print("# Testing prediction")
print("#-------------------------------------#")
print(prediction)

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
submission.to_csv('KNN_060_submission.csv')
pd.read_csv('KNN_060_submission.csv')
print(submission.sample(n=5))

