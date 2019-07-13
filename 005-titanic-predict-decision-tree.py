import pandas as pd 
import titanic_utils, askew_utils
from sklearn import tree, model_selection

train = pd.read_csv("train.csv")
titanic_utils.clean_data(train) # We wrote this, a separate member named utils.py

target = train["Survived"].values # Desired output, usually named target. Separate the column answer from the rest of the columns.
feature_names = ["Pclass", "Age", "Sex", "Fare", "Embarked", "SibSp"]#, "Parch"] # Other fields to analyze hidden patterns
features = train[feature_names].values #Hints


decision_tree = tree.DecisionTreeClassifier(random_state = 1) #Random_state is a hint for machine learning. Setting to 0 means "duh...which way did he go?"

decision_tree_ = decision_tree.fit(features, target) # Fit is doing the actual training.

print("decision_tree_1st_run:", decision_tree_.score(features, target))

##
## Our score is biased and tried to hard, resulting in fitting the data with a sinusoid and not a curve (2nd form polynom.)
## Model_selection.cross_val_score - cross validates the machine learning curve. It hides part of the data and resamples comparing results.
##
scores = model_selection.cross_val_score(decision_tree, features, target, scoring = 'accuracy', cv = 50)

print("model_selection_1st_run",scores)
print("model_selection_1st_run mean", scores.mean())

##
### Rerun the Decision Tree with additional parameters
##

generalized_tree = tree.DecisionTreeClassifier(
    random_state = 2, #If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random
    max_depth =7,   # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split = 2 # The minimum number of samples required to split an internal node. Default=2
    )

generalized_tree_ = generalized_tree.fit(features, target)

print("generalized_tree_1st_run", generalized_tree_.score(features, target))
scores = model_selection.cross_val_score(generalized_tree, features, target, scoring = 'accuracy', cv = 50)
print( "model_selection_2nd_run" , scores)
print("model_selection_2nd_run mean" ,scores.mean())



tree.export_graphviz(generalized_tree_, feature_names=feature_names, out_file="tree.dot")

### This is a dos command line: c:\app\Graphviz\bin>dot.exe -Tpng  tree.dot > tree.png