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
criterion = 'entropy' #'gini'
max_depth  = 10
max_features = 9#'auto'
random = 42
min_samples_split = 5

criterion_options = ['gini', 'entropy']
max_depth_options = [1, 5, 10, 20, 50]
max_features_options = [2, 9, 'sqrt', 'log2', 'auto']
random_options = [ 1, 2, 42]
min_samples_split_options = [2,5,10, 25, 20]


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


random_rfc = 1      #random_state for Remote Forest Classifier

gama = "auto"       #deprecated variable added for SVM sectionto remove warnings.

njobs_options = [1, -1]
nneighbors_options = [2, 5, 10, 13, 15, 20]
nsplits_options = [2, 5, 10, 20, 30, 40, 50, 60]
nsplits_rfc_options = [2, 5, 10, 15]
nestimators_options = [10, 50, 100, 200, 400, 1000]
shffl_options = [True, False]

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
# D D         TTTTTTT RRRRR   EEEEEEE   EEEEEEE
# D   D          T    R    R  E         E
# D     D   ---  T    R   R   E         E
# D      D  ---  T    R R     EEEE      EEE
# D      D       T    R   R   E         E
# D    D         T    R    R  E         E
# D  D           T    R    R  EEEEEEE   EEEEEEE
#######################################

#
## Decision Tree using criterion
#
results = []
for criterion_option in criterion_options:
    clf = DecisionTreeClassifier(criterion = criterion_option, 
                                 max_depth = max_depth, 
                                 max_features = max_features, 
                                 random_state = random,
                                 min_samples_split = min_samples_split)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree,
                                                                   shuffle = shffl, 
                                                                   random_state = random
                                                                   ), 
                                                                   n_jobs = njobs_dtree, 
                                                                   scoring = xor
                            )
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using criterion of", criterion_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using criterion of", criterion_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, criterion_options).plot()
plt.title('Decision Tree criterion results')
plt.xlabel('Criterion')
plt.ylabel('Accuracy')
plt.show()
#
#
## Decision Tree using max_depth
#
results = []
for max_depth_option in max_depth_options:
    clf = DecisionTreeClassifier(criterion = criterion, 
                                 max_depth = max_depth_option, 
                                 max_features = max_features, 
                                 random_state = random,
                                 min_samples_split = min_samples_split
                                 )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree,
                                                                   shuffle = shffl, 
                                                                   random_state = random
                                                                   ), 
                                                                   n_jobs = njobs_dtree, 
                                                                   scoring = xor
                            )
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using max_depth of", max_depth_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using max_depth of", max_depth_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, max_depth_options).plot()
plt.title('Decision Tree max_depth results')
plt.xlabel('Max_Depth')
plt.ylabel('Accuracy')
plt.show()
#
#
## Decision Tree using max_features
#
results = []
for max_features_option in max_features_options:
    clf = DecisionTreeClassifier(criterion = criterion, 
                                 max_depth = max_depth, 
                                 max_features = max_features_option, 
                                 random_state = random,
                                 min_samples_split = min_samples_split
                                 )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree,
                                                                   shuffle = shffl, 
                                                                   random_state = random
                                                                   ), 
                                                                   n_jobs = njobs_dtree, 
                                                                   scoring = xor
                            )
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using max_feature of", max_features_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using max_feature of", max_features_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, max_features_options).plot()
plt.title('Decision Tree max_features results')
plt.xlabel('Max_Features')
plt.ylabel('Accuracy')
plt.show()
#
## Decision Tree using random_state
#
results = []
for random_option in random_options:
    clf = DecisionTreeClassifier(criterion = criterion, 
                                 max_depth = max_depth, 
                                 max_features = max_features, 
                                 random_state = random,
                                 min_samples_split = min_samples_split)
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
#
## Decision Tree using min_samples_splits
#
results = []
for min_samples_split_option in min_samples_split_options:
    clf = DecisionTreeClassifier(criterion = criterion, 
                                 max_depth = max_depth, 
                                 max_features = max_features, 
                                 random_state = random,
                                 min_samples_split = min_samples_split_option)
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_dtree, 
                                                                   shuffle = shffl, 
                                                                   random_state = random_option), 
                                                                   n_jobs = njobs_dtree, 
                                                                   scoring = xor)
    mean_score = round(np.mean(score) * 100, 2)
    results.append(mean_score)
    print("#-------------------------------------#")
    print("DecisionTree using min_samples_split of", min_samples_split_option, "score:", score )
    print("#-------------------------------------#")
    print("")
    print("#-------------------------------------#")
    print("DecisionTree using min_samples_split of", min_samples_split_option, "MEAN score:", mean_score )
    print("#-------------------------------------#")
    print("")
pd.Series(results, min_samples_split_options).plot()
plt.title('Decision Tree min_samples_split results')
plt.xlabel('Min_Samples_Split')
plt.ylabel('Accuracy')
plt.show()
#
## Testing
#
#clf = SVC(gamma = gama)
#clf = KNeighborsClassifier(n_neighbors = 20)
#clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion = criterion, 
                                 max_depth = max_depth, 
                                 max_features = max_features, 
                                 random_state = random,
                                 min_samples_split = min_samples_split)
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
submission.set_index("PassengerID", inplace = True)
submission.to_csv('DecisionTreeClassifier_submission.csv')
pd.read_csv('DecisionTreeClassifier_submission.csv')
print(submission.sample(n=5))

