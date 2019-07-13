#
#002-titanic-visualize-data2.py
import sys, os
import pandas as pd 
import matplotlib.pyplot as plt 
from askew_utils import DF_Magic
import enforce
try:
    import modules
except:
    os.system('pip install module')
    import modules

female_color = "#FA0000"
#-------------------------------------------------------#
def rotate_xaxis(owner):
#-------------------------------------------------------#
    for label in owner.xaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(6)


df = DF_Magic.get_df(filename='train.csv')#pd.read_csv("train.csv")
DF_Magic.usage()
#df.info()

df['survived']  = df['survived'].map({0:"Died", 1:"Survived"})
fig = plt.figure(figsize = (18,12))

#-----------------------------------#
ax = plt.subplot2grid((3,4), (0,0))
#-----------------------------------#
ax = df['survived'].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax)
plt.title("Survived")

#-----------------------------------#
ax1 = plt.subplot2grid((3,4), (0,1))
#-----------------------------------#
ax1 = df['survived'][df.sex == "male"].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b', sharey = '') #Normalize turns into percentages
ax1.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax1.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax1)
plt.title("Men Survived")

#-----------------------------------#
ax2 = plt.subplot2grid((3,4), (0,2))
#-----------------------------------#
ax2 = df['survived'][df.sex == "female"].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = female_color) #Normalize turns into percentages
ax2.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax2.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax2)
plt.title("Women Survived")

#-----------------------------------#
ax3 = plt.subplot2grid((3,4), (0,3))
#-----------------------------------#
ax3 = df['sex'][df.survived == "Survived"].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = [female_color, 'b']) #Normalize turns into percentages
ax3.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax3.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax3)
plt.title("Sex wrt Survived")


df['survived']  = df['survived'].map({"Died":0, "Survived":1})


#-----------------------------------#
ax4 = plt.subplot2grid((3,4), (1,0), colspan = 4)
#-----------------------------------#
a = set(df['pclass'])
a = sorted(a)
for x in a:
    df['survived'][df.pclass == x].plot(kind = "kde")
plt.title("Class wrt Survived")
ax4.legend(a)

#-----------------------------------#
ax5 = plt.subplot2grid((3,4), (2,0))
#-----------------------------------#
ax5 = df['survived'][(df.sex == "male") & (df.pclass == 1)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax5.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax5.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax5)
plt.title("Rich Men Survived")

#-----------------------------------#
ax6 = plt.subplot2grid((3,4), (2,1))
#-----------------------------------#
ax6 = df['survived'][(df.sex == "male") & (df.pclass == 3)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax6.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax6.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax6)
plt.title("Poor Men Survived")


#-----------------------------------#
ax7 = plt.subplot2grid((3,4), (2,2))
#-----------------------------------#
ax7 = df['survived'][(df.sex == "female") & (df.pclass == 1)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax7.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax7.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax7)
plt.title("Rich Women Survived")

#-----------------------------------#
ax8 = plt.subplot2grid((3,4), (2,3))
#-----------------------------------#
ax8 = df['survived'][(df.sex == "female") & (df.pclass == 3)].value_counts(normalize = True).plot(kind = 'bar', alpha = 0.5, color = 'b') #Normalize turns into percentages
ax8.set_xlabel('Died or Survived', fontsize=8, fontweight =5, color = 'g')
ax8.set_ylabel('Percentage rate', fontsize=8, fontweight =5, color = 'g')
rotate_xaxis(ax8)
plt.title("Poor Women Survived")



plt.show()