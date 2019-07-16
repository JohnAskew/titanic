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
def global_variables_function():
#-------------------------------------#
    global refresh_variables
    global return_njobs
    global return_xor
    global return_nneighbors
    global return_nsplits
    global return_nestimators
    global return_shffl
    global return_random
#-------------------------------------#
# V A R I A B L E S  T O  C O N T R O L
#-------------------------------------#
class return_njobs(object):
#-------------------------------------#
    def init():
        return -1          #n_jobs is the number of parallel jobs
#-------------------------------------#
class return_xor():
#-------------------------------------#
    def init():
        return 'accuracy'    #score is the parameter for scoring, such as 'precision'
#-------------------------------------#
def return_nneighbors():
#-------------------------------------#
    nneighbors = 5      #n_neighbors is the number of nearest neighbors to consider
#-------------------------------------#
def return_nsplits():
#-------------------------------------#
    nsplits = 50        #n_splits is number of splits for decisions
#-------------------------------------#
def return_nestimators():
#-------------------------------------#
    nestimators = 400   #n_estimators states how many estimators to allocate
#-------------------------------------#
class return_shffl(object):
#-------------------------------------#
    def init():
        return True         #shuffle - refer to python.org documentation for details
#-------------------------------------#
def return_random():
#-------------------------------------#
    random = 42         #random_state - machine learning started seed.

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
# W O R K I N G  A R E A 
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
gama = "auto"       #deprecated variable added for SVM sectionto remove warnings.
def refresh_variables():
        njobs = return_njobs()         #n_jobs is the number of parallel jobs
        xor = 'accuracy'    #score is the parameter for scoring, such as 'precision'
        nneighbors = 5      #n_neighbors is the number of nearest neighbors to consider
        nsplits = 50        #n_splits is number of splits for decisions
        nestimators = 400   #n_estimators states how many estimators to allocate
        shffl = True        #shuffle - refer to python.org documentation for details
        random = 42          #random_state - machine learning started seed.
        gama = "auto"       #deprecated variable added for SVM sectionto remove warnings.


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


njobs_options = [1, -1]
nneighbors_options = [1, 5, 10, 13, 15, 20]
nsplits_options = [10, 20, 30, 40, 50, 100]
nestimators_options = [10, 50, 100, 200, 400, 1000]
shffl_options = [True, False]
random_options = [0, 1, 2, 42]


#--------------------------------------#
# DEFINE K_FOLD
#--------------------------------------#
#k_fold = KFold(n_splits = return_nsplits(), shuffle = return_shffl(), random_state = return_random())


#-------------------------------------#
# KNN - vary random_state
#-------------------------------------#
global_variables_function()
refresh_variables()
results = []
for random in random_options:
    clf = KNeighborsClassifier(n_neighbors = return_nneighbors())
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = 50, shuffle = return_shffl(), random_state = random, n_jobs = return_njobs(), scoring = return_xor()))
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN random using" ,random, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN random using",random, "MEAN score:", end = '')
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
global_variables_function()
refresh_variables()
results = []
for njob in njobs_options:
    k_fold = KFold(n_splits = nsplits, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njob, scoring = return_xor())
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN n_jobs using" ,njob, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN n_jobs using",njob, "MEAN score:", end = '')
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
global_variables_function()
refresh_variables()
results = []
for nsplit in nsplits_options:
    k_fold = KFold(n_splits = nsplit, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN n_splits of" ,nsplit, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN n_splits of",nsplit, "MEAN score:", end = '')
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
global_variables_function()
refresh_variables()
results = []
for nneighbors in nneighbors_options:
    k_fold = KFold(n_splits = nsplit, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN of" ,nneighbors, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN of",nneighbors, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, nneighbors_options).plot()
plt.title('KNN Varying n_neighbors results')
plt.xlabel('Number of n_neighbors')
plt.ylabel('Accuracy')
plt.show()

#-------------------------------------#
# KNN - vary shuffle
#-------------------------------------#
global_variables_function()
refresh_variables()
results = []
for shffl in shffl_options:
    k_fold = KFold(n_splits = nsplit, shuffle = shffl, random_state = random)
    clf = KNeighborsClassifier(n_neighbors = nneighbors)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("KNN shuffle using" ,shffl, "score:", end = '')
    print(score)
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("# KNN shuffle using",shffl, "MEAN score:", end = '')
    print(mean_score)
    print("#-------------------------------------#")
    print("")
pd.Series(results, shffl_options).plot()
plt.title('KNN Varying shuffle results')
plt.xlabel('Shuffle')
plt.ylabel('Accuracy')
plt.show()




#
## Decision Tree
#
clf = DecisionTreeClassifier()
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
print("#-------------------------------------#")
print("DecisionTree score: ", end = '')
print(score)
print("#-------------------------------------#")
print("")

#
## Decision Tree mean score
#
print("#-------------------------------------#")
print("# Decision Tree mean score:", end = '')
print(round(np.mean(score) * 100, 2))
print("#-------------------------------------#")
print("")



#
## Random Forest Classifier
#

clf = RandomForestClassifier(n_estimators = nestimators)
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
print("#-------------------------------------#")
print("Random Forest Classifier score: ", end = '')
print(score)
print("#-------------------------------------#")
print("")

## Random Forest Classifier mean score
#
print("#-------------------------------------#")
print("# Random Forest Classifier mean score:", end = '')
print(round(np.mean(score) * 100, 2))
print("#-------------------------------------#")
print("")

#
## Naive Bayes
#

clf = GaussianNB()
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
print("#-------------------------------------#")
print("Naive Bayes score: ", end = '')
print(score)
print("#-------------------------------------#")
print("")

## Naive Bayes mean score
#
print("#-------------------------------------#")
print("# Naive Bayes mean score:", end = '')
print(round(np.mean(score) * 100, 2))
print("#-------------------------------------#")
print("")
#
## Support Vector Machine
#
clf = SVC(gamma = gama)
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
print("#-------------------------------------#")
print("SVM score: ", end = '')
print(score)
print("#-------------------------------------#")
print("")


## Support Vector Machine mean score
#
print("#-------------------------------------#")
print("# SVM mean score:", end = '')
print(round(np.mean(score) * 100, 2))
print("#-------------------------------------#")
print("")

#
## Testing
#
#clf = SVC(gamma = gama)
#clf = KNeighborsClassifier(n_neighbors = 20)
#clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators = nestimators)
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = return_njobs(), scoring = return_xor())
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











