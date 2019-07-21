import matplotlib.pyplot as plt
from pylab import *
import numpy as np 
print( "numpy version:",np.__version__)
'''This module gives information about the titanic dataset
   as well as demonstrates using plots with axis (axes),
   dictionaries and slicing the dictionary.
   '''

dict_feature = {0:"survived"
, 1:"pclass"
, 2:"sex"
, 3:"age"
, 4:"sibsp"
, 5:"embarked"
, 6:"parch"
, 7:"fare"
, 8:"cabin"}
dict_explain = {0:"Explanation: passenger fate of survived or died"
,1:"Explanation: Ticket class"
,2:"Explanation: Passenger Gender"
,3:"Explanation: Passenger Age in Years"
,4:"Explanation: Number of siblings and spouse"
,5:"Explanation: Port where passenger boarded ship"
,6:"Explanation: Number of parent and child"
,7:"Explanation: Price of ticket"
,8:"Explanation: Passenger cabin number"

}
dict_values = {0:"Values: 0 = died; 1 = survived"
,1:"Values: 1 = 1st class, 2 = 2nd class, 3 = 3rd class"
,2:"Values: male or female"
,3:"Values: Range between 0 and 100"
,4:"Values: Range between 0 and 10"
,5:"Values: C, Q and S"
,6:"Values: Range between 0 and 10"
,7:"Values: US Dollars and cents"
,8:"Values: Combination of letter and numbers"
}


#fig = plt.figure()
fig, axes = plt.subplots(3,3,sharex='col', sharey='row', figsize=(18,12))

counter = 0
for j in range(9):
    if j in [0,3,6]:
        axes.flat[j].set_ylabel('Row '+str(counter), rotation=0, size='large',labelpad=40)
        axes.flat[j].set_title('plot '+str(j))
        counter = counter + 1
        
    if j in [0,1,3,4,6,7]:
        axes.flat[j].set_title('Numeric Data ')#+str(j)+'\n\nplot '+str(j))
    if j in [2,5]:
        axes.flat[j].set_title('Catagorical Data ')# +str(j))
    if j in [8]:
        axes.flat[j].set_title('Mixed characters')

    axes.flat[j].text(0.4, 0.8, dict_feature[j], style = 'italic', color = 'white', bbox = {'facecolor':'red', 'alpha':0.5, "pad":10 })
    axes.flat[j].text(0.1, 0.4, dict_explain[j], color = 'purple')
    axes.flat[j].text(0.1, 0.2, dict_values[j], horizontalalignment = 'left', color = 'darkgreen')

fig = gcf()
fig.suptitle("Titanic Data Dictionary", fontsize=14)
plt.show()
