import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from askew_utils import DF_Magic as dfm 


train = dfm.get_df('http://bit.ly/kaggletrain')
test = dfm.get_df('http://bit.ly/kaggletest')

train_pure = train.dropna(axis = 0)
train_pure_survived = train_pure[train_pure['survived'] ==1]
train_pure_died = train_pure[train_pure['survived'] == 0]

print("#-------------------------------------#")
print("# train pure survived")
print("#-------------------------------------#")
print(train_pure_survived.describe().T.round(3))
ax_survived = train_pure_survived.hist(grid = True,   xlabelsize = 7, xrot = 45, ylabelsize = 7, figsize = (15,20), layout = (2,4), bins = 20, color='green',)
plt.hist([train_pure_survived['age'], train_pure_died['age']])
plt.suptitle("Titanic training data: COMPLETE data of SURVIVED (no nans)")
plt.show()

print("#-------------------------------------#")
print("# train pure died")
print("#-------------------------------------#")
print(train_pure_died.describe().T.round(3))
ax_died = train_pure_died.hist(grid = True,   xlabelsize = 7, xrot = 45, ylabelsize = 7, figsize = (15,20), layout = (2,4), bins = 20, color='red',)

plt.suptitle("Titanic training data: COMPLETE data of DIED (no nans)")
plt.show()

for plot in ['fare', 'sex', 'age', 'pclass', 'sibsp', 'parch']:
    if train_pure[plot].dtype == 'int64' or train_pure[plot].dtype == 'float64':
        plt.hist([train_pure_survived[plot], train_pure_died[plot]], label = 'survived', color = ['green', 'red'])
        plt.title("COMPLETE DATA column \"" + plot.title() + "\": Compare survived vs. died")
        plt.legend()
        plt.show()


