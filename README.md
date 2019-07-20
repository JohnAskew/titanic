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

#### 020-titanic-predict-logic-regression.py
Learning can be as much observing what NOT to do as learning what to do. 
Here we use a simple LogisticRegression example. We observer the prediction displayed in the command line output, 
is NOT that accurate, ergo propter hoc, we should look for a more accurate model 
or at least tune our existing model. Here we are introduced to new functionality from sklearn,
sklearn.preprocessing. We tune the model and go from a very simple logistic regression model,
to using a little algebra - polynomials. By comparing the outputs between a simple logistical regression
and the polynomial model, we see the polynomial model reveals a higher prediction accuracy. 
It serves to point out a hint, from the output derived from running the raw data visualization programs
starting with 00x- (000-xxx, 001-xxx, 002-xxx). The data does not fit or produce the training answers,
which can be mapped to a simple line. We should consider using a more accurate model. 

#### 021-titanic-predict-decision-tree.py
The point of this module is to prove you can't just take predictions as being true and 
as accurate as they report. Sad, I know. We start with simple decision tree models,
where we now are exposed to tuning a model using options (parameters and their arguments).
The first decision tree model is just a simple model, with only one option being tuned. 
An output file of predictions is generated and can be uploaded to the 
kaggle.com's "Titanic competition" website for judging the actual results. 

There is a second model, which is a generalized decision tree. Here we specify more options to further tune the model. The model paragraph being tuned appears as:
##### generalized_tree = tree.DecisionTreeClassifier(
     random_state = 42
    ,max_depth =10
    ,min_samples_split =5 ...
Feel free to play around with these options and observe how changing each parameter's argument
can impact the reported prediction results displayed in the command line output. The conclusion 
is the prediction result listed in the command line output, as well as a prediction results csv file, 
which too, can be uploaded to kaggle.com for judging the actual results of our model. Be prepared 
for a let-down, but chins high and chests out, as we run the next models, our accuracy does truly improve.
    



# References:
1. Minsuk Heo - "Kaggle - Titanic Data Analysis".
2. Ju Liu - "Predicting Titanic survivors with machine learning".
3. Mike Bernico - "Introduction to Pandas with Titanic Dataset"
```
