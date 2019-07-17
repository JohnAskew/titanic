import os, pickle
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from askew_utils import DF_Magic as dfm

#######################################
## Intention was to publish scripts 
##   as part of tutorial or beginner's
##   set of instructions for jumping 
##   into Python.
#######################################


sns.set()

if os.path.exists("000-train_lowercase_cols.pickle"):
    with open("000-train_lowercase_cols.pickle", 'rb') as in_file:
        train = pickle.load(in_file)
        print("loading 000-train_lowercase_cols.pickle")
else:
     train = dfm.get_df('train.csv')#pd.read_csv("train.csv")

#
## Barchart
#
def bar_chart(feature):
        survived = train[train['survived'] == 1][feature].value_counts()
        dead = train[train['survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['Survived', 'Died']
        df.plot(kind = 'bar', stacked = True, figsize = (10,15))
        plt.title("Pristine Data: Survived vs. Died wrt \"" + feature + "\" column")
        plt.show()

bar_chart('sex')
bar_chart('pclass')
bar_chart('sibsp')
bar_chart('parch')
bar_chart('embarked')

