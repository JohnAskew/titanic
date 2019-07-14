import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


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
