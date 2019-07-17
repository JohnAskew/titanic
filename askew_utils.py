#!/usr/bin/env python
#name: askew_utils.py

import pandas as pd

'''General usage utilities and shortcuts. Weird name so as not to confuse with genuine utility scripts.'''

##
### Make a dataframe
###
class DF_Magic(object):
    def __init__(self):
        pass
    #-------------------------------------#
    def get_df(filename):
    #-------------------------------------#
        df = pd.read_csv(filename)
        missing_values = ["n/a", "na", "--", "-", "N/A"] #Add missing value semiphores here
        blank_line = ''
        print("#-------------------------#")
        print("# Returning a DataFrame-->Rows:",df.shape[0], "Cols:", df.shape[1])
        print("#-------------------------#")
        print('#')
        print("###### Report Starts Here #######")
        print('#')
        print("#-------------------------#")
        print("---- Review Raw Data -----")
        print("#-------------------------#")
        print(blank_line)
        print(df.sample(n=5))
        print(blank_line)
        print("#-------------------------#")
        print("Fix cols: rm space/lwrcase")
        print("#-------------------------#")
        df.columns = df.columns.str.replace( ' ', '_').str.strip().str.lower()
        my_columns = list(df.columns)
        print(my_columns)
        print(blank_line)
        print("#-------------------------#")
        print("-- shape (Rows and Cols.) -", end ='')
        print(df.shape)
        print("#-------------------------#")
        print(blank_line)
        print("#-------------------------#")
        print("--------- info -----------")
        print("number of non-missing records for each column plus data types")
        print("#-------------------------#")
        df.info()
        print(blank_line)
        print("#-------------------------#")
        print("- stats on numeric fields ")
        print("       (describe())")
        print("#-------------------------#")
        print(df.describe().T.round(3))
        print(blank_line)
        print("#-------------------------#")
        print("------- Null count -------")
        print("#-------------------------#")
       #--------------------------------------------#
        for col in df.columns:
            print('column {} has MEAN null_count: {}'.format(col, df[col].isnull().mean()))
        #--------------------------------------------#
        print(blank_line)
        print("#-------------------------#")
        print("------ Null Shape --------")
        print("#-------------------------#") 
        print(df.dropna().shape)
        print(blank_line)
        print("#-------------------------#")
        print("------ Missing Data ------")
        print("#-------------------------#") 
        missing_info = list(df.columns[df.isnull().any()])
        for col in missing_info:
            num_missing = df[df[col].isnull() == True].shape[0]
            print('number missing for column {}: {}'.format(col, num_missing))
        #--------------------------------------------#
        print(blank_line)
        print("#-------------------------#") 
        print("------ Review Sample -----")
        print("#-------------------------#") 
        print(df.sample(n=5))
        print(blank_line)
        return df
    #-------------------------------------#
    def get_num_list(dataframe):
    #-------------------------------------#
        num_list  = list(dataframe.dtypes[dataframe.dtypes != object].index)
        print("#-------------------------#")
        print("# Making a numeric list of numeric columns: \n",num_list)
        print("#-------------------------#")
        return num_list

    #-------------------------------------#
    def usage():
    #-------------------------------------#
        print("#-------------------------#")
        print("# DF_Magic usage")
        print("#")
        print("# List of methods (try help(DF_Magic))")
        print("# ...get_df = pass csv, get back DFrame")
        print("# ...get_num_list = pass DFrame, get")
        print("#                   back LIST of numeric")
        print("#                   columns (for M/L")
        print("# ...get_df_name(df) = pass it dataframe")
        print("#                      it returns df name")
        print("#")
        print("#-------------------------#")

 


