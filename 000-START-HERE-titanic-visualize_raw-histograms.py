#!/usr/bin/env python
#name: 000-titanic-visulize.py
'''
#######################################
Show raw data histograms and demonstrate DF_Magic for dataframe report.
#--------------------------------------#
## Intention was to publish scripts 
##   as part of tutorial or beginner's
##   set of instructions for jumping 
##   into Python.
#######################################
The series of scripts starts with this one,
which will load most of the packages needed
if not already installed.

The usual file input (pd.read_csv) is replaced with 
dfm.get_df - which is part of the module
askew_utils. The intent is to demonstrate
file i/o using a class, as well as providing
user with extensive details about the dataframe
content. The details are written to the console
output. The content includes the typical data 
exploration commands, such as head and info. 
In addition, we summarize the content with nulls,
remove spaces from the column names and set
the column names to all lowercase.
'''

import sys, os
try:
    import pandas as pd 
except:
    os.system('pip install pandas')
    import pandas as pd 
try:
    import matplotlib.pyplot as plt 
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot
from askew_utils import DF_Magic as dfm
try:
    from pylab import *
except:
    os.system('pip install pylab')
    from pylab import *
try:
    import pickle
except:
    os.system('pip install pickle')
    import pickle

try:
    train = dfm.get_df('http://bit.ly/kaggletrain')
except:
    train = dfm.get_df('train.csv')
try:
    test = dfm.get_df('http://bit.ly/kaggletest')
except:
    test = dfm.get_df('test.csv')

dfm.usage() # just so you know it's available. Only time we run this.


fig = train.hist(grid = True,   xlabelsize = 7, xrot = 45, ylabelsize = 7, figsize = (15,20), layout = (2,4), bins = 10)
plt.suptitle("Titanic training data: raw NUMERIC data histograms --> Identify which columns have only few values and which will need to be feature engineered")
plt.show()

with open("000-train_lowercase_cols.pickle", "wb") as in_file:    #Pickle saves results as reuable object
        pickle.dump(train, in_file)                     #Save results from above to Pickle.





