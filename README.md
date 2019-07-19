This project aims to demonstrate Python using simple machine learning models. 
Demonstrated topics include using classes for file i/o, functions to organize code, 
importing your own custom modules, using pickle to write and then read data
and starter examples for data analytics and machine learning. The term Data Science
is deliberately avoided; this project is developer driven and not authored by a Data Scientist.

## USAGE:

Start by running 000-START-HERE-titanic-visualize_raw-histograms.py.

It will generate a visualized page showing histograms of columns containing numeric data.
The intent is to introduce the user to data exploration. With the visualized page,
pay attention to the x-axis of each histogram, noting which columns contain only a 
few values vs. a broad range of values, which will require feature engineering - which 
will be covered in later modules.

#### 001-titanic-visualize-rawdata-compare.py
Raw data visuals comparing data (features, or each relevant
column in spreadsheet) to who survived.
It demonstrates simply code for extracting data
and using simple bar charts.

#### 002-titanic-visualize-rawdata-compare.py
Raw data presenting more stringent features (column data)
than 001-titanic-visualize-rawdata-compare.py. Here we 
move from simple data exploration and start moving towards 
data mining our data content. When reviewing, pay attention
to what data is being presented and ask yourself, 
does this give me a better idea of what the data looks like? 
Does a particular chart give me additional information 
about data contents that deems further exploration?

#### 003-titanic-visualize-rawdata-compare.py
Returning to more generalized presentation of our data content,
we introduce using a KDE or Kernal Density Estimator to our 
bar chart - see Class wrt Age chart. Suggested you 
"Google" KDE chart and understand what value it adds.

#### 010-titanic-feature-mapping-and-engineering.py
Before we run models, we aim to reduce data cardinality (diversity of data values)
to column formats optimized for machine learning. Here 
assumptions are made and data is manipulated to fit our 
assumptions. We intend the user to review this module and
determine what assumptions to make, in order to optimize
the data content. For example, if a particular data row
contains NULLS or "nothing" for a column, what data do 
we fill in to replace the missing data, as machine learning
does not fare well with missing data or "nan" (not a number,
or NULLS).

# References:
1. Minsuk Heo - "Kaggle - Titanic Data Analysis".
2. Ju Liu - "Predicting Titanic survivors with machine learning".
3. Mike Bernico - "Introduction to Pandas with Titanic Dataset"
```
