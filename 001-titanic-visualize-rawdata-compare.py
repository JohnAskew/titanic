import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

#######################################
## Does not utilize askew_utils.DF_Magic
##   to provide contrast against File i/o
##   from 000-titanic-visualize_raw-histograms.py
## Intention was to publish scripts 
##   as part of tutorial or beginner's
##   set of instructions for jumping 
##   into Python.
#######################################


sns.set()

train = pd.read_csv('train.csv')
#
## Barchart
#
def bar_chart(feature):
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['Survived', 'Dead']
        df.plot(kind = 'bar', stacked = True, figsize = (10,15))
        plt.title("Survived vs. Died wrt " + feature)
        plt.show()

bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
