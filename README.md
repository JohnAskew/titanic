
This project aims to demonstrate Python using simple machine learning models. 
Demonstrated topics include using: *classes for file i/o, functions to organize code, 
plotting, including visualizing the data dictionary via plotting (not always a graph),
importing your own custom modules, using pickle to write and then read data
and starter examples for data analytics and machine learning*. The term Data Science
is deliberately avoided; this project is developer driven and not authored by a Data Scientist.

# USAGE:

_Start by running_: 
### 000-START-HERE-titanic-visualize_raw-histograms.py.

It will generate a visualized page showing histograms of columns containing numeric data.
The intent is to introduce the user to data exploration. With the visualized page,
pay attention to the x-axis of each histogram, noting which columns contain only a 
few values vs. a broad range of values, which may require feature engineering -  covered
in later modules. Much can be written on feature engineering, so keeping it simple, 
machine learning processes numbers more efficiently than other data, and too many 
columns can bias the model and impede the model's effectiveness.

### 000-titanic-code-snippets-and-tricks
Demonstrates new ways to extract and organize data. There are no visualizations, only
command line output displaying the information requested about selected data. MIGHT
be helpful in feature selection, at least from an ideas perspective.

### 000-titanic-visualize_complete_records
Extracts and visualizes only complete records, with no missing data. The visualizations
include comparing column values split out by survived vs. died.

### 000-titanic-visualize-data-dictionary.py
Demonstrates plotting does not have to include points on a graph.
This module gives information about the titanic dataset as well as demonstrates using 
plots with axis (axes), dictionaries and slicing the dictionary.

### 001-titanic-visualize-rawdata-compare.py
Raw data visuals comparing data (features, or each relevant column in spreadsheet)
to who survived. It demonstrates simple code for extracting data
and using simple bar charts.

### 002-titanic-visualize-rawdata-compare.py
Raw data presenting more stringent features (column data) than 
001-titanic-visualize-rawdata-compare.py. Here we move from simple 
data exploration and start moving towards data mining our data content. 
When reviewing, pay attention to what data is being presented and ask yourself, 
*does this give me a better idea of what the data looks like? Does a 
particular chart give me additional information about data contents that 
deems further exploration?*

### 003-titanic-visualize-rawdata-compare.py
Returning to more generalized presentation of our data content,
we introduce using a KDE or Kernal Density Estimator to our bar chart,
see *Class wrt Age chart*. Suggested that one "Google" KDE chart 
and understand what value it adds.

### 010-titanic-feature-mapping-and-engineering.py
Before we run models, we aim to reduce data cardinality (diversity of data values)
to column formats optimized for machine learning. Here assumptions are made 
and data is manipulated to fit our assumptions. We intend the user to review this module 
and determine what assumptions to make, in order to optimize the data content.
For example, if a particular data row contains NULLS or "nothing" for a column, 
what data do we fill in to replace the missing data, as machine learning does not 
fare well with missing data or "nan" (not a number, or NULLS).

### 020-titanic-predict-logic-regression.py
Learning can be as much observing what NOT to do as learning what to do. 
Here we use a simple LogisticRegression example. We observe the prediction displayed 
in the command line output, is NOT that accurate, ergo propter hoc, we should 
strive for greater accuracy or at least tune our existing model. Here we are 
introduced to new functionality from sklearn, sklearn.preprocessing. We tune the model 
and go from a very simple logistic regression model, to using algebra - polynomials. 
By comparing the outputs between a simple logistical regressionand the polynomial model,
we see the polynomial model reveals a higher prediction accuracy. It serves to point out 
a hint, from the output derived from running the raw data visualization programs
starting with 00x- (000-xxx, 001-xxx, 002-xxx). The data does not fit or produce the 
training answers, which can be mapped to a simple line. We should consider using 
a more appropriate model. Did you catch the previous point? We are looking for 
a more appropriate model, not a more accurate model. The Zen lies in the addage:
"*Good is the enemy of what is best.*"

### 021-titanic-predict-decision-tree.py
The point of this module is to prove you can't just take predictions as being true and 
as accurate as they report. Sad, I know. We start with simple decision tree models,
where we now are exposed to tuning a model using options (parameters and their arguments).
The first decision tree model is just a simple model, with only one option being tuned. 
An output file of predictions is generated and can be uploaded to the 
kaggle.com's "Titanic competition" website for judging the actual results. 

There is a second model, which is a generalized decision tree. Here we specify 
more options to further tune the model. The model paragraph being tuned appears as:
#### generalized_tree = tree.DecisionTreeClassifier(
     random_state = 42
    ,max_depth =10
    ,min_samples_split =5 ...
Feel free to play around with these options and observe how changing each parameter's argument
can impact the reported prediction results displayed in the command line output. The conclusion 
is the prediction result listed in the command line output, as well as a prediction results csv file, 
which too, can be uploaded to kaggle.com for judging the actual results of our model. Be prepared 
for a let-down, but chins high and chests out, as we run the next models, our accuracy does truly improve.

### 022-titanic-tune-random-forest.py (more feature engineering, too)
This modules moves into more intermediate aspects of machine learning and negligibly more advanced
Python functionality: cross validating our results, generating interactive charts revealing 
the model's accuracy and additional feature engineering. This module further relies on model tuning 
and engages user to tune the model by changing the option's arguments. Module objective is to engage user
in changing arguments (settings) for reviewing the impact to prediction accuracy. You may skip reading the 
remainder of this section and simply run the module.

We introduce cross validation functionality to review our generated predictions
and provide a second opinion of how accurate our predictions actually were. This module also exposes the user
to Panda's "get_dummy" functionality, which takes a feature (column), reviews the contents and creates 
new columns containing only a zero or a one. This 0 or 1 state, if referred to as a "binary" state, 
implies there are only 2 answers: 0 or 1, True or False, etc. Example: our "sex" (gender) column 
contains either "male" or "female". After sending the column "sex" through the Panda's get_dummy functionality
results in 2 new columns - "sex_male" and "sex_female", each column containing either 0 or 1. If the "sex_male"
column contains a 1, then that record is for a male. You may be wondering why we need 2 separate columns 
to specify sex or gender. You don't. This was only an example, simplified for explanation.

Example of feature engineering in the code:

for variable in categorical_variables:
    dummies = pd.get_dummies(X[variable], prefix = variable) # Create an array of dummies
    X = pd.concat([X, dummies], axis = 1) # Update X to include dummies and drop the main variable
    X.drop([variable], axis = 1, inplace = True)

for variable in categorical_variables:
    dummies = pd.get_dummies(Y[variable], prefix = variable) # Create an array of dummies
    Y = pd.concat([Y, dummies], axis = 1) # Update X to include dummies and drop the main variable
    Y.drop([variable], axis = 1, inplace = True)

Moving on, we feature engineer the column / feature, "cabin". Past observations of the raw data reveals a wide range
of data for this feature, generally in the format of a letter and then some numbers, such as C28. In a few instances,
multiple cabins are listed in the column (feature) separated by a comma. This will not play well with simple
machine learning models. This module provides ONE WAY of feature engineering the "cabin" feature, the
objective is to point out to the user, that feature engineering is an art (imho) and open for the user
to change and manipulate the data contained in the cabin column, to drive what ever results you are after. 
Our example simply uses the very first character as the cabin.

### 031-test-general-tree.py
Breather, in terms of complexity and time. Simple repeat of Decision Tree using the DecisionTreeClassifer model
to produce prediction results and a results csv file for uploading to kaggle.com. NO visals generated, just 
command line output of prediction accuracy and a second opinion using cross_validation functionality previously
introduced. 

### 040-titanic-simple-KMeans-predict.py 
*In-Progress: Still validating results. Use with caution.*

Advanced Python functionality to feature engineer data to fit the KMeans model. This module introduces an
alternative way to convert catagorical data and non-numeric data to numeric format. The other examples 
rely on mapping ranges to represent ranges of non-numeric data. This module attempts to manage non-numeric
data using lists and dictionaries to track each non-numeric data value and mapping it to a number. 
The first non-numeric data value for any given feature is mapped to the value 0. Any other row containing
the same column data value receives a 0 in place of the non-numeric data. The next non-numeric data value
different from the first value we mapped to 0, is assigned or mapped to 1 and so on. Example, if the first
5 rows contain embarked data of S, C or Q, then any column containing 'S' in the embarked column would be
changed to a 0. Any row containing 'C' in the embarked column would be mapped to a 1. Subsequently, the 
value of 'Q' would be mapped to a 2, and so on.

This module also exposes user to more intermediate numpy functionality, reshape, as well as giving 
another example of sklearn's preprocessing functionality. End result is a self reported accuracy rate 
and a predicted output csv file for uploading to kaggle.

### 050-titanic-tune_knn_and_test.py
Simple K Nearest Neighbors model. Tune, test and build prediction model.

### 060-titanic-tune_dtreeclassifier_and_test.py
Decision Tree Classifier tune, test and build prediction model.

### 070-titanic-tune_randomforestclassifier_and_test.py
Random Forest Classifier model. No new functionality introduced. Provides framework to tune the model and the cross validation scoring functionality.

As the tests can be time consuming, a new option was added at the top of the module, a set of
model_tuned switches which if a switch is set to True, then that corresponding test and graphical 
output will be skipped. If you wish to tune a particular model parameter, ensure the corresponding
model_tuned switch is set to false. 

Final result is predicted survivors output csv fit to upload to kaggle.com.

### 100-example-using-dummies.py
Practice module demonstrating Panda's get_dummy functionality.

### 101-example-feature-engineering-using-dummies.py
Practice module which feature engineer's "sex" and "embarked" columns and then feeds results to Panda's get_dummy.
End result is new dataframe with new columns matching the engineered "sex" and "embarked" values. Upon review
you may be considering having 2 columns to represent "sex" as being redundant. That would be a secondary point of 
this module. Feel free to drop one of the columns. Another exercise on feature engineering can be seen in the
module "titanic_utils". The function starting with the line: "def clean_data(data):" has 2 "sex" column mapping
lines: "data.loc[data["sex"] == "male", "sex"] = 0", "data.loc[data["sex"] == "female", "sex"] =1". We engage
the user to contemplate the optimal way to change "male and "female" into 0 or 1, where  0 = male and 1 = female.
This is easily accomplished with an advanced Python feature known as a "list comprehension", but is deemed to be
too advanced a topic to explain on an introductory project. Ideally, there would be one column named, "sex" 
containing 0 and 1. Feel free to change the code and implement your own solution. The overall objective of this 
project is to get the user engaged and start thinking outside the narrow box of the example's provided.

### 1000-titanic-code-snippets-and-tricks.py
Runnable code giving examples of how to extract and display various bits of information about the titanic data.


# References:
1. **Minsuk Heo** - "_Kaggle - Titanic Data Analysis_".
2. **Ju Liu** - "_Predicting Titanic survivors with machine learning_".
3. **Mike Bernico** - "_Introduction to Pandas with Titanic Dataset_".
4. **John Harrison** - __*Sentdex*__ - "_KMeans with Titanic Dataset - Practical Machine Learning_".
```
