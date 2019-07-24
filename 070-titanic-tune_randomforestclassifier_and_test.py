import sys, os
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic as dfm

#-------------------------------------#
# Model Tuning
#-------------------------------------#
nestimators = 100   #n_estimators for Random Forest Classifier = states how many estimators to allocate
max_depth = 5
criterion = 'gini'
min_samples_split =2
max_features = "auto"
nsplits = 10    #n_splits with Random Forect Classifier
random = 42      #random_state for Remote Forest Classifier
#-------------------------------------#
# Scoring Tuning
#-------------------------------------#
shffl = True        #shuffle - refer to python.org documentation for details
njobs = -1          #n_jobs is the number of parallel jobs
xor = 'accuracy'    #score is the parameter for scoring, such as 'precision'

#-------------------------------------#
# Turn on or off each tuning test (the test and graphical output)
#-------------------------------------#
nestimators_tuned = False
max_depth_tuned = False
criterion_tuned = False
min_samples_split_tuned = False
max_features_tuned = False
nsplits_tuned = False
random_state_tuned = False
njobs_tuned = False

#-------------------------------------#
# Ranges for each tuning option
#-------------------------------------#
nestimators_options = [10, 50, 100, 200, 400, 1000]
max_depth_options = [1,2,3,4,5,6,7,8,8,10]
criterion_options = ['gini',]
min_samples_split_options = [2,3,4,5,10]
max_features_options = ['log2', 'sqrt', 'auto']
random_options = [0, 1, 2, 42]
#
## Scoring ranges for tuning scoring
#
nsplits_options = [2, 5, 10, 15]
njobs_options = [1, -1]
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
if nestimators_tuned:
    pass
else:
    results = []
    for nestimators_option in nestimators_options:
        clf = RandomForestClassifier(n_estimators = nestimators_option
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features
                                   )
        score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
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
## Random Forest Classifier using max_depth
#
if max_depth_tuned:
    pass
else:
    results = []
    for max_depth_option in max_depth_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth_option
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features
                                   )
        score = cross_val_score(clf, training_data, target, cv = KFold( n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
        mean_score = round(np.mean(score) * 100, 2)
        results.append(mean_score)
        print("#-------------------------------------#")
        print("Random Forest Classifier with max_depth of", max_depth_option, "score:", score)
        print("#-------------------------------------#")
        print("")
        print("#-------------------------------------#")
        print("Random Forest Classifier with max_depth_option of", max_depth_option, "MEAN score:", mean_score)
        print("#-------------------------------------#")
        print("")
    pd.Series(results, max_depth_options).plot()
    plt.title('Random Forest Classifer max_depth results')
    plt.xlabel('Max_Depth')
    plt.ylabel('Accuracy')
    plt.show()
#
## Random Forest Classifier using criterion
#
if criterion_tuned:
    pass
else:
    results = []
    for criterion_option in criterion_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion_option
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features
                                   )
        score = cross_val_score(clf, training_data, target, cv = KFold( n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
        mean_score = round(np.mean(score) * 100, 2)
        results.append(mean_score)
        print("#-------------------------------------#")
        print("Random Forest Classifier with criterion of", criterion_option, "score:", score)
        print("#-------------------------------------#")
        print("")
        print("#-------------------------------------#")
        print("Random Forest Classifier with criterion_option of", criterion_option, "MEAN score:", mean_score)
        print("#-------------------------------------#")
        print("")
    pd.Series(results, criterion_options).plot()
    plt.title('Random Forest Classifer criterion results')
    plt.xlabel('Criterion')
    plt.ylabel('Accuracy')
    plt.show()
#
## Random Forest Classifier using min_samples_split
#
if min_samples_split_tuned:
    pass
else:
    results = []
    for min_samples_split_option in min_samples_split_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split_option
                                   , max_features = max_features
                                   )
        score = cross_val_score(clf, training_data, target, cv = KFold( n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
        mean_score = round(np.mean(score) * 100, 2)
        results.append(mean_score)
        print("#-------------------------------------#")
        print("Random Forest Classifier with min_samples_split_option of", min_samples_split_option, "score:", score)
        print("#-------------------------------------#")
        print("")
        print("#-------------------------------------#")
        print("Random Forest Classifier with min_samples_split_option of", min_samples_split_option, "MEAN score:", mean_score)
        print("#-------------------------------------#")
        print("")
    pd.Series(results, min_samples_split_options).plot()
    plt.title('Random Forest Classifer min_samples_split_options results')
    plt.xlabel('Min Sample Splits')
    plt.ylabel('Accuracy')
    plt.show()
#
## Random Forest Classifier using max_features
#
if max_features_tuned:
    pass
else:
    results = []
    for max_features_option  in max_features_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features_option
                                   )
        score = cross_val_score(clf, training_data, target, cv = KFold( n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
        mean_score = round(np.mean(score) * 100, 2)
        results.append(mean_score)
        print("#-------------------------------------#")
        print("Random Forest Classifier with max_features_option of", max_features_option, "score:", score)
        print("#-------------------------------------#")
        print("")
        print("#-------------------------------------#")
        print("Random Forest Classifier with max_features_option of", max_features_option, "MEAN score:", mean_score)
        print("#-------------------------------------#")
        print("")
    pd.Series(results, max_features_options).plot()
    plt.title('Random Forest Classifer max_features_options results')
    plt.xlabel('Max Features')
    plt.ylabel('Accuracy')
    plt.show()
#
#
## Random Forest Classifier using n_splits
#
if nsplits_tuned:
    pass
else:
    results = []
    for nsplits_option in nsplits_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                    , random_state = random
                                    , n_jobs = njobs
                                    , max_depth = max_depth
                                    , criterion = criterion
                                    , min_samples_split = min_samples_split
                                    , max_features = max_features
                                    )
        score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits_option, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
        mean_score = round(np.mean(score) * 100, 2)
        results.append(mean_score)
        print("#-------------------------------------#")
        print("Random Forest Classifier with n_split of", nsplits_option, "score:", score)
        print("#-------------------------------------#")
        print("")
        print("#-------------------------------------#")
        print("Random Forest Classifier with n_split of", nsplits_option, "MEAN score:", mean_score)
        print("#-------------------------------------#")
        print("")
    pd.Series(results, nsplits_options).plot()
    plt.title('Random Forest Classifer n_splits results')
    plt.xlabel('Number of n_splits')
    plt.ylabel('Accuracy')
    plt.show()

#
## Random Forest Classifier using random_state
#
if random_state_tuned:
    pass
else:

    results = []
    for random_option in random_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features
                            )
        score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random_option), n_jobs = njobs, scoring = xor)
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
if njobs_tuned:
    pass
else:
    results = []
    for njobs_option in njobs_options:
        clf = RandomForestClassifier(n_estimators = nestimators
                                   , random_state = random
                                   , n_jobs = njobs
                                   , max_depth = max_depth
                                   , criterion = criterion
                                   , min_samples_split = min_samples_split
                                   , max_features = max_features
                               )
        score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs_option, scoring = xor)
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
    clf = RandomForestClassifier(n_estimators = nestimators
                               , random_state = random
                               , n_jobs = njobs
                               , max_depth = max_depth
                               , criterion = criterion
                               , min_samples_split = min_samples_split
                               , max_features = max_features
                               )
    score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor_option)
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

## Testing
#
#clf = SVC(gamma = gama)
#clf = KNeighborsClassifier(n_neighbors = 20)
#clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators = nestimators
                           , random_state = random
                           , n_jobs = njobs
                           , max_depth = max_depth
                           , criterion = criterion
                           , min_samples_split = min_samples_split
                           , max_features = max_features
                           )
score = cross_val_score(clf, training_data, target, cv = KFold(n_splits = nsplits, shuffle = shffl, random_state = random), n_jobs = njobs, scoring = xor)
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

submission.to_csv('submission_RandomForestClassifier.csv')
pd.read_csv('submission_RandomForestClassifier.csv')
print(submission.sample(n=5))

