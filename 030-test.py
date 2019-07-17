import sys, os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic as dfm

#-------------------------------------#
# V A R I A B L E S  T O  C O N T R O L
#-------------------------------------#
njobs = -1          #n_jobs is the number of parallel jobs
njobs_dtree = 1     #n_jobs for Decision Tree
njobs_rfc =1        #n_jobs for Random Forest Classifier

xor = 'accuracy'    #score is the parameter for scoring, such as 'precision'

nneighbors = 5      #n_neighbors is the number of nearest neighbors to consider

nsplits = 10        #n_splits is number of splits for decisions
nsplits_dtree = 25  #n_splits for Decision Tree
nsplits_rfc = 50    #n_splits with Random Forect Classifier

nestimators = 400   #n_estimators for Random Forest Classifier = states how many estimators to allocate

shffl = True        #shuffle - refer to python.org documentation for details

random = 42         #random_state - machine learning started seed.
random_rfc = 1      #random_state for Remote Forest Classifier

gama = "auto"       #deprecated variable added for SVM sectionto remove warnings.

njobs_options = [1, -1]
nneighbors_options = [2, 5, 10, 13, 15, 20]
nsplits_options = [2, 5, 10, 20, 30, 40, 50, 60]
nsplits_rfc_options = [2, 5, 10, 15]
nestimators_options = [10, 50, 100, 200, 400, 1000]
shffl_options = [True, False]
random_options = [0, 1, 2, 42]
xor_options = ['accuracy', 'precision'] ## To-DO - Add me for each model


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
#-------------------------------------#
# KNN - vary shuffle
#-------------------------------------#

results = []
for shffl_option in shffl_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl_option, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl_option, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN shuffle using" ,shffl_option, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN shuffle using",shffl_option, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('KNN Varying shuffle results')
plt.xlabel('Shuffle')
plt.ylabel('Accuracy')
plt.show()
#-------------------------------------#
# KNN - vary random_state
#-------------------------------------#

results = []
for random_option in random_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random_option)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random_option), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN random using" ,random_option, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN random using",random_option, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, random_options).plot()
plt.title('KNN Varying random results')
plt.xlabel('Random')
plt.ylabel('Accuracy')
plt.show()
#-------------------------------------#
# KNN - vary n_jobs
#-------------------------------------#

results = []
for njob_option in njobs_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njob_option, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN n_jobs using" ,njob_option, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN n_jobs using",njob_option, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('KNN Varying n_jobs results')
plt.xlabel('N_job')
plt.ylabel('Accuracy')
plt.show()
#-------------------------------------#
# KNN - vary n_splits
#-------------------------------------#

results = []
for nsplit_option in nsplits_options:
    k_fold = KFold(n_splits = nsplit_option, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplit_option, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN n_splits of" ,nsplit_option, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN n_splits of",nsplit_option, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nsplits_options).plot()
plt.title('KNN Varying n_splits results')
plt.xlabel('Number of n_splits')
plt.ylabel('Accuracy')
plt.show()
#-------------------------------------#
# KNN - vary n_neighbors
#-------------------------------------#

results = []
for nneighbors_option in nneighbors_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors_option)
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

#######################################
# D D         TTTTTTT RRRRR   EEEEEEE   EEEEEEE
# D   D          T    R    R  E         E
# D     D   ---  T    R   R   E         E
# D      D  ---  T    R R     EEEE      EEE
# D      D       T    R   R   E         E
# D    D         T    R    R  E         E
# D  D           T    R    R  EEEEEEE   EEEEEEE
#######################################
#
## Decision Tree using n_splits
#
results = []
for nsplits_option in nsplits_options:
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_option, shuffle = shffl, random_state = random), n_jobs = njobs_dtree, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using n_splits of", nsplits_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using n_splits of", nsplits_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, nsplits_options).plot()
plt.title('Decision Tree n_splits results')
plt.xlabel('Number of n_splits')
plt.ylabel('Accuracy')
plt.show()
#
#
## Decision Tree using n_splits
#
results = []
for shffl_option in shffl_options:
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree, shuffle = shffl_option, random_state = random), n_jobs = njobs_dtree, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using shuffle of", shffl_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using n_splits of", shffl_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('Decision Tree shuffle results')
plt.xlabel('Shuffle')
plt.ylabel('Accuracy')
plt.show()

#
## Decision Tree using random_state
#
results = []
for random_option in random_options:
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree, shuffle = shffl, random_state = random_option), n_jobs = njobs_dtree, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using random_state of", random_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using random_state of", random_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, random_options).plot()
plt.title('Decision Tree random_state results')
plt.xlabel('Random_State')
plt.ylabel('Accuracy')
plt.show()
#
## Decision Tree using n_jobs
#
results = []
for njobs_option in njobs_options:
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree, shuffle = shffl, random_state = random_option), n_jobs = njobs_option, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using n_jobs of", njobs_option, "score:", score , end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using n_jobs of", njobs_option, "MEAN score:", mean_score , end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('Decision Tree n_jobs results')
plt.xlabel('Number of Jobs')
plt.ylabel('Accuracy')
plt.show()

#
## Decision Tree using scoring
#
results = []
for xor_option in xor_options:
    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree, shuffle = shffl, random_state = random_option), n_jobs = njobs_dtree, scoring = xor_option)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using scoring with", xor_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using scoring with", xor_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, xor_options).plot()
plt.title('Decision Tree scoring results')
plt.xlabel('Scoring')
plt.ylabel('Accuracy')
plt.show()

##########################################
# RRRR     FFFFFFF    CCCCCC
# R    R   F         C
# R   R    F        C
# RRR      FFFFF    C
# R  R     F        C
# R    R   F         C
# R     R  F          CCCCCC  
#####################################
#
## Random Forest Classifier using n_estimators
#
results = []
for nestimators_option in nestimators_options:
    clf = RandomForestClassifier(n_estimators = nestimators_option)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_estimators of", nestimators_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_estimators of", nestimators_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nestimators_options).plot()
plt.title('Random Forest Classifer n_estimators results')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()
#
#
## Random Forest Classifier using n_splits
#
results = []
for nsplits_option in nsplits_options:
    clf = RandomForestClassifier(n_estimators = nestimators)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_option, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_split of", nsplit_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_split of", nsplit_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nsplits_options).plot()
plt.title('Random Forest Classifer n_splits results')
plt.xlabel('Number of n_splits')
plt.ylabel('Accuracy')
plt.show()
#
## Random Forest Classifier using n_splits
#
results = []
for shffl_option in shffl_options:
    clf = RandomForestClassifier(n_estimators = nestimators)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl_option, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with shuffle of", shffl_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with shuffle of", shffl_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('Random Forest Classifer shuffle results')
plt.xlabel('Shuffle')
plt.ylabel('Accuracy')
plt.show()
#
## Random Forest Classifier using random_state
#
results = []
for random_option in random_options:
    clf = RandomForestClassifier(n_estimators = nestimators)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_option), n_jobs = njobs_rfc, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with random_state of", random_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with random_state of", random_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, random_options).plot()
plt.title('Random Forest Classifer random_state results')
plt.xlabel('Random_State')
plt.ylabel('Accuracy')
plt.show()
#
## Random Forest Classifier using n_jobs
#
results = []
for njobs_option in njobs_options:
    clf = RandomForestClassifier(n_estimators = nestimators)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_option, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_jobs of", njobs_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with n_jobs of", njobs_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('Random Forest Classifer n_jobs results')
plt.xlabel('Number of Jobs')
plt.ylabel('Accuracy')
plt.show()
#
## Random Forest Classifier using scoring
#
results = []
for xor_option in xor_options:
    clf = RandomForestClassifier(n_estimators = nestimators)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor_option)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Random Forest Classifier with scoring of", xor_option, "score:", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Random Forest Classifier with scoring of", xor_option, "MEAN score:", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, xor_options).plot()
plt.title('Random Forest Classifer scoring results')
plt.xlabel('Scoring')
plt.ylabel('Accuracy')
plt.show()
#######################################
# N       N     A     IIIIIII  V         V  EEEEEEE
# N N     N    A A       I      V       V   E
# N  N    N   A   A      I       V     V    EEEE
# N    N  N  AAAAAAA     I        V   V     E
# N     N N  A     A     I         V V      E
# N       N  A     A  IIIIIII       V       EEEEEEE
#######################################
#
## Naive Bayes using n_splits
#
results = []
for nsplits_option in nsplits_options:
    clf = GaussianNB()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_option, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Naive Bayes n_splits of", nsplits_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Naive Bayes n_splits of", nsplits_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nsplits_options).plot()
plt.title('Naive Bayes n_splits results')
plt.xlabel('Number of n_splits')
plt.ylabel('Accuracy')
plt.show()
#
## Naive Bayes using shuffle
#
results = []
for shffl_option in shffl_options:
    clf = GaussianNB()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl_option, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Naive Bayes shuffle of", shffl_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Naive Bayes shuffle of", shffl_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('Naive Bayes shuffles results')
plt.xlabel('Number of shuffles')
plt.ylabel('Accuracy')
plt.show()
#
## Naive Bayes using random_state
#
results = []
for random_option in random_options:
    clf = GaussianNB()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random_option), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Naive Bayes random_state of", random_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Naive Bayes random_state of", random_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, random_options).plot()
plt.title('Naive Bayes random_state results')
plt.xlabel('Random_State')
plt.ylabel('Accuracy')
plt.show()
#
## Naive Bayes using n_jobs
#
results = []
for njobs_option in njobs_options:
    clf = GaussianNB()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs_option, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Naive Bayes n_jobs of", njobs_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Naive Bayes n_jobs of", njobs_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('Naive Bayes n_jobs results')
plt.xlabel('Number of Jobs')
plt.ylabel('Accuracy')
plt.show()
#
## Naive Bayes using scoring
#
results = []
for xor_option in xor_options:
    clf = GaussianNB()
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor_option)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("Naive Bayes scoring of", xor_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("Naive Bayes scoring of", xor_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, xor_options).plot()
plt.title('Naive Bayes scoring results')
plt.xlabel('Score')
plt.ylabel('Accuracy')
plt.show()
#######################################
#     SSSSSS   V            V  M             M
#    SS         V          V   MM           MM
#       SS       V        V    M M         M M
#          SS     V      V     M   M      M  M
#           SS     V    V      M     M  M    M
#          SS       V V        M       M     M
#     SSSSS          V         M             M
#######################################
#
## Support Vector Machine using n_splits
#
results = []
for nsplits_option in nsplits_options:
    clf = SVC(gamma = gama)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_option, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("SVM using nsplit", nsplits_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("SVM using nsplit", nsplits_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nsplits_options).plot()
plt.title('Support Vector Machine n_splits results')
plt.xlabel('Number of Splits')
plt.ylabel('Accuracy')
plt.show()
#
## Support Vector Machine using shuffle
#
results = []
for shffl_option in shffl_options:
    clf = SVC(gamma = gama)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl_option, random_state = random), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("SVM using shuffle", shffl_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("SVM using shuffle", shffl_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('Support Vector Machine shuffle results')
plt.xlabel('Number of Shuffles')
plt.ylabel('Accuracy')
plt.show()
#
## Support Vector Machine using random_state
#
results = []
for random_option in random_options:
    clf = SVC(gamma = gama)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random_option), n_jobs = njobs, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("SVM using random_state", random_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("SVM using random_state", random_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, random_options).plot()
plt.title('Support Vector Machine random_state results')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.show()
#
## Support Vector Machine using n_jobs
#
results = []
for njobs_option in njobs_options:
    clf = SVC(gamma = gama)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs_option, scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("SVM using n_jobs", njobs_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("SVM using n_jobs", njobs_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, njobs_options).plot()
plt.title('Support Vector Machine n_jobs results')
plt.xlabel('Number of Jobs')
plt.ylabel('Accuracy')
plt.show()
#
## Support Vector Machine using scoring
#
results = []
for xor_option in xor_options:
    clf = SVC(gamma = gama)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor_option)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("SVM using scoring", xor_option, "score: ", score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("SVM using scoring", xor_option, "MEAN score: ", mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, xor_options).plot()
plt.title('Support Vector Machine scoring results')
plt.xlabel('Scoring')
plt.ylabel('Accuracy')
plt.show()



## Testing
#
#clf = SVC(gamma = gama)
#clf = KNeighborsClassifier(n_neighbors = 20)
#clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators = nestimators)
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_rfc, shuffle = shffl, random_state = random_rfc), n_jobs = njobs_rfc, scoring = xor)
clf.fit(training_data, target)
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

submission.to_csv('submission.csv')
pd.read_csv('submission.csv')
print(submission.sample(n=5))











