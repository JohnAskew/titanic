#!/usr/bin/env python

import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic
#-------------------------------------------------------#
def rotate_xaxis(owner):
#-------------------------------------------------------#
    for label in owner.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(6)


df = DF_Magic.get_df('train.csv') #pd.read_csv("train.csv")


df['survived']  = df['survived'].map({0:"Died", 1:"Survived"})
fig = plt.figure(figsize = (18,12))


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







fig.title = 'Titanic Analysis Page 1'
plt.show()

