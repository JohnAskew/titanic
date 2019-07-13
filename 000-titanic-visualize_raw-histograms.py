#!/usr/bin/env python
#name: 000-titanic-visulize.py
'''Show raw data histograms and demonstrate DF_Magic for dataframe report'''

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
from askew_utils import DF_Magic
try:
    from pylab import *
except:
    os.system('pip install pylab')
    from pylab import *

df = DF_Magic.get_df(filename='train.csv')#pd.read_csv("train.csv")


fig = df.hist(grid = True, xlabelsize = 7, xrot = 45, ylabelsize = 7, figsize = (20,20), layout = (2,4), sharey = True)
plt.show()


