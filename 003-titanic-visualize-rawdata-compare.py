#!/usr/bin/env python

import sys, os
import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic as dfm
import pickle
#-------------------------------------------------------#
def rotate_xaxis(owner):
#-------------------------------------------------------#
    for label in owner.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(6)


if os.path.exists("000-train_lowercase_cols.pickle"):
    with open("000-train_lowercase_cols.pickle", 'rb') as in_file:
        df = pickle.load(in_file)
        print("Loading 000-train_lowercase_cols.pickle")
else:
    df = dfm.get_df('train.csv')


df['survived']  = df['survived'].map({0:"Died", 1:"Survived"})
fig = plt.figure(figsize = (18,12)).suptitle('Pristine Data: Titanic Training Data Analysis: Page 3')


#-----------------------------------#
ax = plt.subplot2grid((2,3), (0,0))
#-----------------------------------#
ax = df['survived'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax)
plt.title("Survived")

#-----------------------------------#
ax1 = plt.subplot2grid((2,3), (0,1))
#-----------------------------------#
ax1 = plt.scatter(df['survived'], df['age'], alpha=0.1) #Linear line, so alpha is light to see outlier, clusters are darker
plt.title("Age wrt Survived")

#-----------------------------------#
ax2 = plt.subplot2grid((2,3), (0,2))
#-----------------------------------#
ax2 = df['pclass'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b')
rotate_xaxis(ax2)
plt.title("Class")

#-----------------------------------#
ax3 = plt.subplot2grid((2,3), (1,0), colspan = 2)
#-----------------------------------#
a = set(df['pclass'])
a = sorted(a)
for x in a:
    df['age'][df.pclass == x].plot(kind = "kde")
plt.title("Class wrt Age")
plt.legend(a)

#-----------------------------------#
ax4 = plt.subplot2grid((2,3), (1,2))
#-----------------------------------#
ax4 = df['embarked'].value_counts(normalize = True).plot(kind = "bar")
plt.title("Embarked")

plt.show()

