<<<<<<< HEAD
import os, pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd 
import timeit
import matplotlib.pyplot as plt
from askew_utils import DF_Magic as dfm
import titanic_utils as tu 

#-------------------------------------#
def clean_cabin(x):
#-------------------------------------#
    try:
        return x[0]
    except TypeError:
        return "None"
#-------------------------------------#
def clean_catagoricals(x):
#-------------------------------------#
    x.fillna("Missing", inplace = True) # Fill in the missing data with the word, "Missing"

#######################################
# M A I N   L O G I C   S T A R T
#######################################
# V A R I A B L E S to adjust processing accuracy
#--------------------------------------#
var_bootstrap = True
var_n_estimators = 1000# 1000 (400 = .873400)
oob_score = False   # True of False does not seem to impact accuracy
var_n_jobs = -1
var_random_state = 1
var_max_features = "auto"
var_min_samples_leaf = 5 #(8 - .873757)
var_criterion = 'mse'
var_max_depth = 30 #10
var_warm = False

if os.path.exists("010-train_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
    with open("010-train_lowercase_cols.pickle", 'rb') as in_file:
        X = pickle.load(in_file)
        print("Loading 010-train_lowercase_cols.pickle")
else:
    X = dfm.get_df('train.csv')
    tu.clean_data(train) 

y = X.pop("survived")

#
## Fill in missing values as machine learning does not fare well with nulls.
#
### Moved to titanic_utils...X['age'].fillna(X.age.mean(), inplace = True)  # Fill in missing Age with average age

#
## Extract the columns that only contain numbers
#
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
#categorical_variables = ['sex', 'cabin', 'embarked']
categorical_variables = ['cabin'] #, 'embarked'] embarked made numeric
#numeric_variables = DF_Magic.get_num_list(df)
print(numeric_variables)
#
## Set up and run first training session to establish a base line.
#
model = RandomForestRegressor(n_estimators = 100, oob_score = True, random_state = var_random_state)
model.fit(X[numeric_variables], y)

model.oob_score_
print("BASELINE model.oob_score_:",model.oob_score_)

y_oob = model.oob_prediction_
print("BASELINE c-stat:", roc_auc_score(y, y_oob))

#
## At this point, we have a benchmark, but it needs improving.
#
#--------------------------------------#
# Tweak one - drop irrelevant data
#--------------------------------------#
### Removed with drop logic moved to 010-feature....X.drop(["name", "ticket", "passengerid"], axis =1 , inplace = True)
#Removed with drop logic moved to 010-feature...X.drop(["name"], axis =1 , inplace = True)

#--------------------------------------#
# Tweak two - Clean data
#--------------------------------------#
## Clean cabin and massage data to only include the alphabetic prefix.
##     If data contains C24, the result is either:
##         a. C
##         b. None 

X["cabin"] = X.cabin.apply(clean_cabin)
for variable in categorical_variables:
    clean_catagoricals(X[variable]) #X[variable].fillna("Missing", inplace = True) # Fill in the missing data with the word, "Missing"
#--------------------------------------#
# Tweak 3:
#--------------------------------------#
## Convert catagorical columsn to numeric
##     columns containing either a 0 or 1
##     Example: Sex with values male/female
##              becomes sex_male with values
##              0 or 1 and sex_female, 0/1.

for variable in categorical_variables:
    dummies = pd.get_dummies(X[variable], prefix = variable) # Create an array of dummies
    X = pd.concat([X, dummies], axis = 1) # Update X to include dummies and drop the main variable
    X.drop([variable], axis = 1, inplace = True)

print('#------------------------------#')
print('## After tweaking, we have new columns:')
print('##     Review output and ensure all columns')
print('##     have the same value under "count" column')
print('#------------------------------#')
print(X.describe().T.round(2))

#-------------------------------------#
# Refine parameters for next training run
#-------------------------------------#
model = RandomForestRegressor(100, oob_score = True, n_jobs = var_n_jobs, random_state = var_random_state)
model.fit(X, y)
print('#------------------------------#')
print("TRAIN session 1 c-stat:", roc_auc_score(y, model.oob_prediction_))
print('#------------------------------#')

#-------------------------------------#
## Variable importance measures
#-------------------------------------#
##     We will create visual to validate
##     findings.

feature_importances = pd.Series(model.feature_importances_, index = X.columns)
feature_importances.sort_values(inplace = True)
feature_importances.plot(kind = 'barh', figsize = (12,4))
plt.title('Columns by feature_importances')
plt.show()

#-------------------------------------#
## Parameter Tests
#-------------------------------------#
#
## n_jobs
#
model = RandomForestRegressor(1000, oob_score = True, n_jobs = 1, random_state = var_random_state)
model.fit(X, y)
print('#------------------------------#')
print("TRAIN session 2 (n_jobs=1) c-stat:", roc_auc_score(y, model.oob_prediction_))
print('#------------------------------#')


#
## n_estimators - 1000 seems maximized
#
results = []
n_estimator_options = [ 10, 20, 30, 40, 50 , 100, 200, 500, 1000, 1100 ]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees
                                , bootstrap = var_bootstrap
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state)
    model.fit(X, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("trees", trees )
    print('#------------------------------#')
    print("TRAIN session 3 (vary n_estimator): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, n_estimator_options).plot()
plt.title('Varying n_estimator results')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()


# #
# ## max_features - maximizes with "auto"
# #
results = []
max_features_options = [ "auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = 1000
                                , oob_score = True
                                , n_jobs = var_n_jobs 
                                , random_state = var_random_state
                                , max_features = max_features)
    model.fit(X, y)
    print("max_features option", max_features)
    roc = roc_auc_score(y, model.oob_prediction_)
    print('#------------------------------#')
    print("TRAIN session 4 (vary max_features): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, max_features_options).plot(kind = 'barh', xlim=(.85, .88))
plt.title("Varying max_features")
plt.xlabel('Accuracy')
plt.ylabel('Max_feature')
plt.show()

# #
# ## min_samples_leaf - maximizes at 5
# #
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = var_n_estimators
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state
                                , max_features = var_max_features
                                , min_samples_leaf = min_samples)
    model.fit(X, y)
    print("min_samples", min_samples)
    roc = roc_auc_score(y, model.oob_prediction_)
    print('#------------------------------#')
    print("TRAIN session 5 (vary min_samples_leaf): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, min_samples_leaf_options).plot()
plt.xlabel('min_samples_leaf_options')
plt.ylabel('accuracy')
plt.title('Varying min_samples_leaf')
plt.show()

# #
# ## criterion  - mse shuts out mae
# #
results = []
criterion_options = ['mse', 'mae']

for criterion_option in criterion_options:
    model = RandomForestRegressor(n_estimators = var_n_estimators
                                  , oob_score = True
                                  , n_jobs = var_n_jobs
                                  , random_state = var_random_state
                                  , max_features = var_max_features
                                  , min_samples_leaf = var_min_samples_leaf
                                  , criterion = criterion_option)
    model.fit(X,y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("criterion_option:", criterion_option)
    print('#------------------------------#')
    print("TRAIN session 6 (vary criterion_options: c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, criterion_options).plot()
plt.show()

#
## Max_depth - peaks between 10 and 20. 20 at .87425 accurate
##

results = []
max_depth_options = [1, 10, 20, 30, 40, 50,]

for max_depth in max_depth_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = var_n_estimators
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state
                                , max_features = var_max_features
                                , min_samples_leaf = var_min_samples_leaf
                                , criterion = var_criterion
                                , max_depth = max_depth)
    model.fit(X, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("max_depth option", max_depth)
    print('#------------------------------#')
    print("TRAIN session vary max_depth: c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)
pd.Series(results, max_depth_options).plot()
plt.title("Varying max_depth")
plt.xlabel('max_depth')
plt.ylabel("accuracy")
plt.show()

#
## warmstart
#
# results = []
# warm_start_options = [True, False]

# for warm in warm_start_options:
#     model = RandomForestRegressor(bootstrap = var_bootstrap
#                                 , n_estimators = var_n_estimators
#                                 , oob_score = True
#                                 , n_jobs = var_n_jobs
#                                 , random_state = var_random_state
#                                 , max_features = var_max_features
#                                 , min_samples_leaf = var_min_samples_leaf
#                                 , criterion = var_criterion
#                                 , max_depth = var_max_depth
#                                 , warm_start = warm
#         )
#     model.fit(X, y)
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print("warm_start option", str(warm))
#     print('#------------------------------#')
#     print("TRAIN session vary warm_start: c-stat:", roc)
#     print('#------------------------------#')
#     results.append(roc)
# pd.Series(results, warm_start_options).plot()
# plt.title("Varying warm_start")
# plt.xlabel('warm_start')
# plt.ylabel("accuracy")
# plt.show()


#
## min_impurity_decrease: 0 yeilds no change in accuracy. As min_impurity_decrease increases, accurracy does down
## A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
#
# results = []
# min_impurity_decrease_options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# for min_impurity_decrease in min_impurity_decrease_options:
#     model = RandomForestRegressor(bootstrap = var_bootstrap
#                                   , n_estimators = var_n_estimators
#                                   , oob_score = True
#                                   , n_jobs = var_n_jobs
#                                   , random_state = var_random_state
#                                   , max_features = var_max_features
#                                   , min_samples_leaf = var_min_samples_leaf
#                                   , criterion = var_criterion
#                                   , max_depth = 20
#                                   , min_impurity_decrease = min_impurity_decrease)
#     model.fit(X, y)
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print("min_impurity_decrease:", min_impurity_decrease)
#     print('#------------------------------#')
#     print("TRAIN session ? (min_impurity_decrease): c-stat:", roc)
#     print('#------------------------------#')
#     results.append(roc)
# pd.Series(results, min_impurity_decrease_options).plot()
# plt.xlabel('min_impurity_decrease')
# plt.ylabel('accuracy')
# plt.title('Varying min_impurity_decrease')
# plt.legend()
# plt.show()









=======
import os, pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd 
import timeit
import matplotlib.pyplot as plt
from askew_utils import DF_Magic as dfm
import titanic_utils as tu 

#-------------------------------------#
def clean_cabin(x):
#-------------------------------------#
    try:
        return x[0]
    except TypeError:
        return "None"
#-------------------------------------#
def clean_catagoricals(x):
#-------------------------------------#
    x.fillna("Missing", inplace = True) # Fill in the missing data with the word, "Missing"

#######################################
# M A I N   L O G I C   S T A R T
#######################################
# V A R I A B L E S to adjust processing accuracy
#--------------------------------------#
var_bootstrap = True
var_n_estimators = 1000# 1000 (400 = .873400)
oob_score = False   # True of False does not seem to impact accuracy
var_n_jobs = -1
var_random_state = 1
var_max_features = "auto"
var_min_samples_leaf = 5 #(8 - .873757)
var_criterion = 'mse'
var_max_depth = 30 #10
var_warm = False

if os.path.exists("010-train_lowercase_cols.pickle"): #020-train_cleaned_data-lowercase_cols.pickle"):
    with open("010-train_lowercase_cols.pickle", 'rb') as in_file:
        X = pickle.load(in_file)
        print("Loading 010-train_lowercase_cols.pickle")
else:
    X = dfm.get_df('train.csv')
    tu.clean_data(train) 

y = X.pop("survived")

#
## Fill in missing values as machine learning does not fare well with nulls.
#
### Moved to titanic_utils...X['age'].fillna(X.age.mean(), inplace = True)  # Fill in missing Age with average age

#
## Extract the columns that only contain numbers
#
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
#categorical_variables = ['sex', 'cabin', 'embarked']
categorical_variables = ['cabin'] #, 'embarked'] embarked made numeric
#numeric_variables = DF_Magic.get_num_list(df)
print(numeric_variables)
#
## Set up and run first training session to establish a base line.
#
model = RandomForestRegressor(n_estimators = 100, oob_score = True, random_state = var_random_state)
model.fit(X[numeric_variables], y)

model.oob_score_
print("BASELINE model.oob_score_:",model.oob_score_)

y_oob = model.oob_prediction_
print("BASELINE c-stat:", roc_auc_score(y, y_oob))

#
## At this point, we have a benchmark, but it needs improving.
#
#--------------------------------------#
# Tweak one - drop irrelevant data
#--------------------------------------#
### Removed with drop logic moved to 010-feature....X.drop(["name", "ticket", "passengerid"], axis =1 , inplace = True)
#Removed with drop logic moved to 010-feature...X.drop(["name"], axis =1 , inplace = True)

#--------------------------------------#
# Tweak two - Clean data
#--------------------------------------#
## Clean cabin and massage data to only include the alphabetic prefix.
##     If data contains C24, the result is either:
##         a. C
##         b. None 

X["cabin"] = X.cabin.apply(clean_cabin)
for variable in categorical_variables:
    clean_catagoricals(X[variable]) #X[variable].fillna("Missing", inplace = True) # Fill in the missing data with the word, "Missing"
#--------------------------------------#
# Tweak 3:
#--------------------------------------#
## Convert catagorical columsn to numeric
##     columns containing either a 0 or 1
##     Example: Sex with values male/female
##              becomes sex_male with values
##              0 or 1 and sex_female, 0/1.

for variable in categorical_variables:
    dummies = pd.get_dummies(X[variable], prefix = variable) # Create an array of dummies
    X = pd.concat([X, dummies], axis = 1) # Update X to include dummies and drop the main variable
    X.drop([variable], axis = 1, inplace = True)

print('#------------------------------#')
print('## After tweaking, we have new columns:')
print('##     Review output and ensure all columns')
print('##     have the same value under "count" column')
print('#------------------------------#')
print(X.describe().T.round(2))

#-------------------------------------#
# Refine parameters for next training run
#-------------------------------------#
model = RandomForestRegressor(100, oob_score = True, n_jobs = var_n_jobs, random_state = var_random_state)
model.fit(X, y)
print('#------------------------------#')
print("TRAIN session 1 c-stat:", roc_auc_score(y, model.oob_prediction_))
print('#------------------------------#')

#-------------------------------------#
## Variable importance measures
#-------------------------------------#
##     We will create visual to validate
##     findings.

feature_importances = pd.Series(model.feature_importances_, index = X.columns)
feature_importances.sort_values(inplace = True)
feature_importances.plot(kind = 'barh', figsize = (12,4))
plt.title('Columns by feature_importances')
plt.show()

#-------------------------------------#
## Parameter Tests
#-------------------------------------#
#
## n_jobs
#
model = RandomForestRegressor(1000, oob_score = True, n_jobs = 1, random_state = var_random_state)
model.fit(X, y)
print('#------------------------------#')
print("TRAIN session 2 (n_jobs=1) c-stat:", roc_auc_score(y, model.oob_prediction_))
print('#------------------------------#')


#
## n_estimators - 1000 seems maximized
#
results = []
n_estimator_options = [ 10, 20, 30, 40, 50 , 100, 200, 500, 1000, 1100 ]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees
                                , bootstrap = var_bootstrap
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state)
    model.fit(X, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("trees", trees )
    print('#------------------------------#')
    print("TRAIN session 3 (vary n_estimator): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, n_estimator_options).plot()
plt.title('Varying n_estimator results')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()


# #
# ## max_features - maximizes with "auto"
# #
results = []
max_features_options = [ "auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = 1000
                                , oob_score = True
                                , n_jobs = var_n_jobs 
                                , random_state = var_random_state
                                , max_features = max_features)
    model.fit(X, y)
    print("max_features option", max_features)
    roc = roc_auc_score(y, model.oob_prediction_)
    print('#------------------------------#')
    print("TRAIN session 4 (vary max_features): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, max_features_options).plot(kind = 'barh', xlim=(.85, .88))
plt.title("Varying max_features")
plt.xlabel('Accuracy')
plt.ylabel('Max_feature')
plt.show()

# #
# ## min_samples_leaf - maximizes at 5
# #
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = var_n_estimators
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state
                                , max_features = var_max_features
                                , min_samples_leaf = min_samples)
    model.fit(X, y)
    print("min_samples", min_samples)
    roc = roc_auc_score(y, model.oob_prediction_)
    print('#------------------------------#')
    print("TRAIN session 5 (vary min_samples_leaf): c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, min_samples_leaf_options).plot()
plt.xlabel('min_samples_leaf_options')
plt.ylabel('accuracy')
plt.title('Varying min_samples_leaf')
plt.show()

# #
# ## criterion  - mse shuts out mae
# #
results = []
criterion_options = ['mse', 'mae']

for criterion_option in criterion_options:
    model = RandomForestRegressor(n_estimators = var_n_estimators
                                  , oob_score = True
                                  , n_jobs = var_n_jobs
                                  , random_state = var_random_state
                                  , max_features = var_max_features
                                  , min_samples_leaf = var_min_samples_leaf
                                  , criterion = criterion_option)
    model.fit(X,y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("criterion_option:", criterion_option)
    print('#------------------------------#')
    print("TRAIN session 6 (vary criterion_options: c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)

pd.Series(results, criterion_options).plot()
plt.show()

#
## Max_depth - peaks between 10 and 20. 20 at .87425 accurate
##

results = []
max_depth_options = [1, 10, 20, 30, 40, 50,]

for max_depth in max_depth_options:
    model = RandomForestRegressor(bootstrap = var_bootstrap
                                , n_estimators = var_n_estimators
                                , oob_score = True
                                , n_jobs = var_n_jobs
                                , random_state = var_random_state
                                , max_features = var_max_features
                                , min_samples_leaf = var_min_samples_leaf
                                , criterion = var_criterion
                                , max_depth = max_depth)
    model.fit(X, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    print("max_depth option", max_depth)
    print('#------------------------------#')
    print("TRAIN session vary max_depth: c-stat:", roc)
    print('#------------------------------#')
    results.append(roc)
pd.Series(results, max_depth_options).plot()
plt.title("Varying max_depth")
plt.xlabel('max_depth')
plt.ylabel("accuracy")
plt.show()

#
## warmstart
#
# results = []
# warm_start_options = [True, False]

# for warm in warm_start_options:
#     model = RandomForestRegressor(bootstrap = var_bootstrap
#                                 , n_estimators = var_n_estimators
#                                 , oob_score = True
#                                 , n_jobs = var_n_jobs
#                                 , random_state = var_random_state
#                                 , max_features = var_max_features
#                                 , min_samples_leaf = var_min_samples_leaf
#                                 , criterion = var_criterion
#                                 , max_depth = var_max_depth
#                                 , warm_start = warm
#         )
#     model.fit(X, y)
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print("warm_start option", str(warm))
#     print('#------------------------------#')
#     print("TRAIN session vary warm_start: c-stat:", roc)
#     print('#------------------------------#')
#     results.append(roc)
# pd.Series(results, warm_start_options).plot()
# plt.title("Varying warm_start")
# plt.xlabel('warm_start')
# plt.ylabel("accuracy")
# plt.show()


#
## min_impurity_decrease: 0 yeilds no change in accuracy. As min_impurity_decrease increases, accurracy does down
## A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
#
# results = []
# min_impurity_decrease_options = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# for min_impurity_decrease in min_impurity_decrease_options:
#     model = RandomForestRegressor(bootstrap = var_bootstrap
#                                   , n_estimators = var_n_estimators
#                                   , oob_score = True
#                                   , n_jobs = var_n_jobs
#                                   , random_state = var_random_state
#                                   , max_features = var_max_features
#                                   , min_samples_leaf = var_min_samples_leaf
#                                   , criterion = var_criterion
#                                   , max_depth = 20
#                                   , min_impurity_decrease = min_impurity_decrease)
#     model.fit(X, y)
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print("min_impurity_decrease:", min_impurity_decrease)
#     print('#------------------------------#')
#     print("TRAIN session ? (min_impurity_decrease): c-stat:", roc)
#     print('#------------------------------#')
#     results.append(roc)
# pd.Series(results, min_impurity_decrease_options).plot()
# plt.xlabel('min_impurity_decrease')
# plt.ylabel('accuracy')
# plt.title('Varying min_impurity_decrease')
# plt.legend()
# plt.show()









>>>>>>> 1dc1bffe271460b0c73c357ce8aaedfffa0f0aee
