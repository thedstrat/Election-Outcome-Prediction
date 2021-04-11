#!/usr/bin/env python
# coding: utf-8

# # Second Attempt at Different Models
# Summary:
# - Create a base logistic regression
# - Look at feature importance
# - Address data imbalance by attempting a few methodologies (oversampling, undersampling, SMOTE, SMOTE + Undersampling)
# - After addressing imbalance create a new base model
# - Look at feature importance by a few different methodologies (f_classif and RandomForest)
# - Random Forests results in a better base model 
# - Use 20 features to run through various other algorithms (with balanced data)
# - Random Forests is the best overall algorithm

# In[1]:


import pandas as pd
import numpy as np
import getpass
import psycopg2
import numpy as np
import pandas as pd
from psycopg2.extensions import adapt, register_adapter, AsIs
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from matplotlib import pyplot
import seaborn as sns

pd.set_option('display.max_columns', 300)  # allows to display all columns up to 300 in this case

# Magic adapters for Numpy
register_adapter(np.int64,AsIs)
register_adapter(np.float64,AsIs)

# initialize SQL engine
get_ipython().run_line_magic('load_ext', 'sql')
get_ipython().run_line_magic('sql', 'postgres://capstone21_group4:nm_change.me@pgsql.dsa.lan/capstone21_group4')

engine = create_engine('postgres://capstone21_group4:nm_change.me@pgsql.dsa.lan/capstone21_group4')
print (engine.table_names())  # list existing DB tables


# In[2]:


# Look at "normalized" features

#query = """SELECT * FROM cleaned_prediction_data_v2_fips"""    
#df = pd.read_sql(query, engine)

query = """SELECT * FROM normalized_prediction_data_fips"""    
df = pd.read_sql(query, engine)
#display(df.head())
display(df.shape)


# In[3]:


# Drop features that are uncorrelated with our target variable or that could cause overfitting.

# 'protest_votes_2004', 'protest_votes_2000' - these columns contain some NaN values so drop them

df.drop([     
    #   '2016_dem_votes','2016_gop_votes', 
    #  
    '2000_tot_votes','2004_tot_votes','2008_tot_votes','2012_tot_votes','2016_tot_votes','total_votes_2020',
    '2000_gop_votes','2004_gop_votes','2008_gop_votes','votes_gop_2020',
    '2000_dem_votes','2004_dem_votes','2008_dem_votes','votes_dem_2020',
    'protest_votes_2000', 'protest_votes_2004','protest_votes_2008','protest_votes_2012','protest_votes_2016',
    'eday_covid_deaths','eday_covid_perc','Confirmed','Active','Deaths', 
    'county_fips','POPESTIMATE',
    '2012_dem_votes','2012_gop_votes'
    , '2016_dem_votes','2016_gop_votes'  # 98% accuracy if keep these in the prediction    
                       ], axis=1
                       , inplace=True)
display(df.head())
display(df.shape)

# confirm that there are no Null values in the data
print(sum([True for idx,row in df.iterrows() if any(row.isnull())]))   # count number rows with a NaN value


# In[4]:


get_ipython().system('pip install --upgrade scikit-learn')


# # Base Logistic Regression
# Creating a model with all features to see base performance

# In[5]:


# Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics

df_filter = df.copy()

print("list of features used in the prediction",df_filter.columns)

# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns != '2020_result']
y = df_filter['2020_result']

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Create regression
logistic_regression= LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_regression, X_train, y_train, cv=5)
print(scores)


# In[6]:


# Fit the model on the base model to see what results

logistic_regression.fit(X_train, y_train)


# #### The logistic model does a great job at predicting Republican outcomes but is worse at predicting Democratic results. This could be because there are more Republican counties/data points.
# 
# #### We know this because the model incorrectly predicted that 22% of Republican-outcome counties (1) would vote Democrat (0) in 2020. We can see this by the 22 percent in the top right of the confusion matrix below. 

# In[7]:


# Confusion Matrix visualization
plot_confusion_matrix(logistic_regression, X_test, y_test, normalize='true')


# In[8]:


# Predict test data
y_pred = logistic_regression.predict(X_test)


# In[9]:


# Classification report
print(classification_report(y_test, y_pred))


# ### Classification Report Explanation
# 
# Recall - 78% of Democrat counties are classified correctly. 97% of Republican counties are classified correctly.
# 
# Precision -  83% of *predicted* Democrat counties were really Democrat counties. 94% of predicted Republican counties were really Republican counties.
# 
# Since the dataset contains far more Republican counties than Democrat counties, we calculate a 'weighted average', which turns out to be 94%. The weighted average accuracy takes into account this fact, so it is a better gauge of accuracy than 'macro average' in this situation.

# # Feature Selection
# 
# We do not want to include all 170+ features in our model because this may cause overfitting, so we have decided to train each of our models on the top 10 and top 20 features. Below, we attempt to select the most important features based on the Analysis of Variance (ANOVA) statistical test and Decision Tree Classifier.
# 
# To see more information about why we chose our features based on the Decision Tree classifier, please review our supplementary Feature Selection notebook here:
# https://europa.dsa.missouri.edu/user/jdxqd/notebooks/capstone_Group4/GroupProducts/Milestone3/Feature_Selection.ipynb

# In[10]:


from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

# Using SelectKBest and f_classif to identify best features
# Try 10 features
fs = SelectKBest(score_func=f_classif, k=10)
X_selected = fs.fit_transform(X, y)


cols = fs.get_support(indices=True)
features_df_new = df_filter.iloc[:,cols]


# In[11]:


print(features_df_new.columns.values)


# In[12]:


# Try 20 features
fs = SelectKBest(score_func=f_classif, k=20)
X_selected = fs.fit_transform(X, y)


cols = fs.get_support(indices=True)
features_df_new = df_filter.iloc[:,cols]


# In[13]:


print(features_df_new.columns.values)


# In[14]:


# decision tree for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

# set the indep and dep variable
X = df.loc[:, df.columns != '2020_result']   
y = df.filter(['2020_result'])

# define the model
model = DecisionTreeClassifier()

# fit the model
model.fit(X, y)

# get importance
importance = model.feature_importances_

#Show top 20 important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

# Display feature importance as list
top_10_features = feat_importances.nlargest(10)
top_20_features = feat_importances.nlargest(20)
print(top_20_features)


# In[15]:


top_10_features = ['eduation_bachelors_higher_25_64','num_owner_occupied_houses'
,'num_mental_health_providers','num_some_college','pop_18_44'
,'female_18_44','percent_employed_natural_construct_main'
,'percent_employed_professional_scientific_manage_admin','hispanic_perc'
,'rent_1000_2000']
top_20_features = ['eduation_bachelors_higher_25_64', 'nonhisp_white_perc', 'num_mental_health_providers',
                   'percent_employed_natural_construct_main', 'num_uninsured', 'asian_perc', 'rent_lt_1000'
                   'percent_employed_eduation_health_soc', 'perc_physically_inactive', 'MEDHHINC_2019', 'rent_gt_2000', 
                   'percent_house_public_assistance', 'black_perc', 'percent_employed_self_emp_16gr', 'd_New England_', 
                   'num_owner_occupied_houses', 'percent_food_snap_past_12mo', 'female_17_under', 'female_45_64', 
                   'percent_employed_service_16gr']


# # Test Base Model with New Features

# ## 10 features

# In[16]:


# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns.isin(top_10_features)]
y = df_filter['2020_result']

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Create regression
logistic_regression_10features= LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_regression_10features, X_train, y_train, cv=5)
print(scores)


# In[17]:


logistic_regression_10features.fit(X_train, y_train)


# In[18]:


plot_confusion_matrix(logistic_regression_10features, X_test, y_test, normalize='true')


# In[19]:


y_pred = logistic_regression_10features.predict(X_test)
print(classification_report(y_test, y_pred))


# # 20 features
# Rerun base model but this time using top 20 features

# In[20]:


# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns != '2020_result']
y = df_filter['2020_result']

X = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y = df_filter['2020_result']

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# Create regression
logistic_regression_20features= LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_regression_20features, X_train, y_train, cv=5)
print(scores)


# Note: Above, we test the data on a cross validation dataset consisting of 5 randomly selected folds. It appears that the accuracy is relatively similar between each fold which indicates that the model is not overfitting on any particular part of the dataset.

# In[21]:


# Logistic Regression with 20 features
logistic_regression_20features.fit(X_train, y_train)


# In[22]:


plot_confusion_matrix(logistic_regression_20features, X_test, y_test, normalize='true')


# In[23]:


y_pred = logistic_regression_20features.predict(X_test)


# In[24]:


print(classification_report(y_test, y_pred))


# ## An Issue
# 
# The model has substantially more Republican data points. In some cases, this skew in class distribution can lead some algorithms to ignore the minority class (Democrat counties).
# 
# ## Solution
# We will try oversampling (creating duplicate Democratic counties in our train data so that we have an equal number of Republican and Democrat counties) and undersampling (decreasing the number of Republican counties in our train data so that we have an equal number of Republican and Democrat counties) to reduce the imbalance in our data.
# 

# # Oversampling

# In[25]:


get_ipython().system('pip install imbalanced-learn')


# In[26]:


import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# set the indep and dep variable
#X = df_filter.loc[:, df_filter.columns != '2020_result']

X = df_filter.loc[:, df_filter.columns.isin(top_20_features)]

y = df_filter['2020_result']


oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,test_size=0.3,random_state=0)

# Create regression
logistic_over = LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_over, X_train, y_train, cv=5)
print(scores)


# In[27]:


logistic_over.fit(X_train, y_train)


# In[28]:


plot_confusion_matrix(logistic_over, X_test, y_test, normalize='true')


# In[29]:


y_pred = logistic_over.predict(X_test)
print(classification_report(y_test, y_pred))


# # Undersampling

# In[30]:


# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y = df_filter['2020_result']

undersample = RandomUnderSampler(sampling_strategy='majority')
X_over, y_over = undersample.fit_resample(X, y)

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X_over,y_over,test_size=0.3,random_state=0)

# Create regression
logistic_under = LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_under, X_train, y_train, cv=5)
print(scores)


# In[31]:


logistic_under.fit(X_train, y_train)


# In[32]:


plot_confusion_matrix(logistic_under, X_test, y_test, normalize='true')


# In[33]:


y_pred = logistic_under.predict(X_test)
print(classification_report(y_test, y_pred))


# ### Explanation
# Both oversampled and undersampled datasets had far fewer Democrat counties that were labeled Republican. That being said, the oversampled dataset performed the best out of all previous logistic regressions. It received the highest precision, recall, and F1 scores.

# # SMOTE
# 
# Here, we use a novel technique called "Synthetic Minority Oversampling Technique" (SMOTE) to combine both undersampling and oversampling techniques. The technique undersamples the majority class (Republican counties) and oversamples the minority class (Democrat counties). A full explanation can be found in the paper here:
# https://arxiv.org/abs/1106.1813
# 
# According to the classification reports below, this technique produces an F1 scores of 93% for both classes. This is a slight (but possibly negligible) improvement over our oversampled dataset above.

# In[34]:


from imblearn.over_sampling import SMOTE
from collections import Counter

# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y = df_filter['2020_result']

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)
counter = Counter(y_smote)
print(counter)


# In[35]:


# train and test split
X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,test_size=0.3,random_state=0)

# Create regression
logistic_smote = LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_smote, X_train, y_train, cv=5)
print(scores)


# In[36]:


logistic_smote.fit(X_train, y_train)


# In[37]:


plot_confusion_matrix(logistic_smote, X_test, y_test, normalize='true')


# In[38]:


y_pred = logistic_smote.predict(X_test)
print(classification_report(y_test, y_pred))


# # SMOTE + Undersampling

# In[39]:


from imblearn.pipeline import Pipeline
# set the indep and dep variable
X = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y = df_filter['2020_result']

over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)


X_smoteunder, y_smoteunder = pipeline.fit_resample(X, y)

counter = Counter(y_smoteunder)
print(counter)


# In[40]:


# train and test split
X_train,X_test,y_train,y_test = train_test_split(X_smoteunder,y_smoteunder,test_size=0.3,random_state=0)

# Create regression
logistic_smoteunder = LogisticRegression(max_iter=5000)

# Cross validate test data
scores = cross_val_score(logistic_smoteunder, X_train, y_train, cv=5)
print(scores)


# In[41]:


logistic_smoteunder.fit(X_train, y_train)


# In[42]:


plot_confusion_matrix(logistic_smoteunder, X_test, y_test, normalize='true')


# In[43]:


y_pred = logistic_smoteunder.predict(X_test)
print(classification_report(y_test, y_pred))


# # Test All Algorithms with SMOTE dataset and top 20 features

# ## Logistic

# In[57]:


X_20 = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y_20 = df_filter['2020_result']


oversample = SMOTE()
X_smote_20, y_smote_20 = oversample.fit_resample(X_20, y_20)

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X_smote_20,y_smote_20,test_size=0.3,random_state=0)

# Create regression
logistic_smote_20= LogisticRegression(max_iter=5000)

# fit model
logistic_smote_20.fit(X_train, y_train)
lrtest = str(logistic_smote_20.score(X_test, y_test))
lrtrain = str(logistic_smote_20.score(X_train, y_train))

# Cross validate test data
scores = cross_val_score(logistic_smote_20, X_train, y_train, cv=5)
print(scores)


# calculate precision, recall, f1, support to add to comparison dataframe
lr=logistic_smote_20
precision, recall, fscore, support = score(y_test, y_pred)
lrp0 = round(precision[0],2)
lrp1 = round(precision[1],2)
lrr0 = round(recall[0],2)
lrr1 = round(recall[1],2)
lrf0 = round(fscore[0],2)
lrf1 = round(fscore[1],2)
lrs0 = round(support[0],2)
lrs1 = round(support[1],2)  

plot_confusion_matrix(logistic_smote_20, X_test, y_test, normalize='true')  
y_pred = logistic_smote_20.predict(X_test)
print(classification_report(y_test, y_pred))


# ## Naive Bayes

# In[44]:


# Naive Bayes method is a supervised learning algorithm based on applying Bayes’ theorem with the “naive” 
#   assumption of conditional independence between every pair of features given the value of the class variable.
#
#  The score on the training and test data are close to each other, which indicates that the we are not overfitting.
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

X_20 = df_filter.loc[:, df_filter.columns.isin(top_20_features)]
y_20 = df_filter['2020_result']

X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,test_size=0.3,random_state=0)

mnb = MultinomialNB().fit(X_train, y_train)

mnbtest = str(mnb.score(X_test, y_test))
mnbtrain = str(mnb.score(X_train, y_train))

print("score on test: " + mnbtest)
print("score on train: "+ mnbtrain)
print()

y_pred=mnb.predict(X_test)
plot_confusion_matrix(mnb, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
mnbp0 = round(precision[0],2)
mnbp1 = round(precision[1],2)
mnbr0 = round(recall[0],2)
mnbr1 = round(recall[1],2)
mnbf0 = round(fscore[0],2)
mnbf1 = round(fscore[1],2)
mnbs0 = round(support[0],2)
mnbs1 = round(support[1],2)


# ## KNN

# In[45]:


# The k-nearest neighbors (KNN) algorithm is a supervised machine learning algorithm that can be used to solve 
#  both classification and regression problems. For KNN, it is known that it does not work so well with large 
#  datasets (high sample size) and in with many features (high dimensions) in particular.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knntest = str(knn.score(X_test, y_test))
knntrain = str(knn.score(X_train, y_train))

print("train shape: " + str(X_train.shape))
print("score on test: " + knntest)
print("score on train: "+ knntrain)
print()

y_pred=knn.predict(X_test)
plot_confusion_matrix(knn, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
knnp0 = round(precision[0],2)
knnp1 = round(precision[1],2)
knnr0 = round(recall[0],2)
knnr1 = round(recall[1],2)
knnf0 = round(fscore[0],2)
knnf1 = round(fscore[1],2)
knns0 = round(support[0],2)
knns1 = round(support[1],2)


# ## SVM

# In[46]:


# The Support Vector Machine is a simple algorithm for classification and regression tasks. Setting the regularisation parameter C=0.0001 improves the quality of the prediction and 
#   reduces overfitting.

#  The objective of the support vector machine algorithm is to find a hyperplane (ie.. decision boundaries) in an N-dimensional 
#    space(N — the number of features) that distinctly classifies the data points.

#  it is highly recommended to scale your data

from sklearn.svm import LinearSVC

svm=LinearSVC(C=.01)  # C default 1, decrease it to remove noisy observations = more regularization
svm.fit(X_train, y_train)

svmtest = str(svm.score(X_test, y_test))
svmtrain = str(svm.score(X_train, y_train))

print("score on test: " + svmtest)
print("score on train: "+ svmtrain)
print()

y_pred=svm.predict(X_test)
plot_confusion_matrix(svm, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
svmp0 = round(precision[0],2)
svmp1 = round(precision[1],2)
svmr0 = round(recall[0],2)
svmr1 = round(recall[1],2)
svmf0 = round(fscore[0],2)
svmf1 = round(fscore[1],2)
svms0 = round(support[0],2)
svms1 = round(support[1],2)


# ## Decision Tree

# In[47]:


# Decision Tree is a non-parametric supervised learning method used for classification and regression. 
#   The goal is to create a model that predicts the value of a target variable by learning simple decision rules 
#   (if-else) inferred from the data features.

# Applying a single decision tree to this feature rich dataset leads to massive overfitting. Indeed, an accuracy of 
#   100% means it has remembered exactly the training dataset and is therefore generalising poorly on the test data.
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

clftest = str(clf.score(X_test, y_test))
clftrain = str(clf.score(X_train, y_train))

print("score on test: "  + clftest)
print("score on train: " + clftrain)
print()

y_pred=clf.predict(X_test)
plot_confusion_matrix(clf, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
clfp0 = round(precision[0],2)
clfp1 = round(precision[1],2)
clfr0 = round(recall[0],2)
clfr1 = round(recall[1],2)
clff0 = round(fscore[0],2)
clff1 = round(fscore[1],2)
clfs0 = round(support[0],2)
clfs1 = round(support[1],2)


# ## Bagging Classifier

# In[48]:


# If decision tree overfits, applying an ensemble learning algorithm like bagging might improve the quality of the prediction 
#   model. In bagging, the training data is increased by taking bootstraps from the training data. This means multiple samples 
#   are taken (with replacement) from the training data and the model is trained on these sub-datasets. The final prediction 
#   is the average over all predictions from each bootstrap sample.

# Bagging Classifier is much slower as it basically runs multiple decision trees but one can see a reduction of the overfitting 
#  versus a single Decision Tree leading to an increase in the test score.

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# max_samples: maximum size 0.5=50% of each sample taken from the full dataset
# max_features: maximum of features 1=100% taken here all 10K 
# n_estimators: number of decision trees 
bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
bg.fit(X_train, y_train)

bgtest = str(bg.score(X_test, y_test))
bgtrain = str(bg.score(X_train, y_train))

print("score on test: " + bgtest)
print("score on train: "+ bgtrain)
print()

y_pred=bg.predict(X_test)
plot_confusion_matrix(bg, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
bgp0 = round(precision[0],2)
bgp1 = round(precision[1],2)
bgr0 = round(recall[0],2)
bgr1 = round(recall[1],2)
bgf0 = round(fscore[0],2)
bgf1 = round(fscore[1],2)
bgs0 = round(support[0],2)
bgs1 = round(support[1],2)


# ## RandomForestClassifier

# In[49]:


# This is another frequently used ensemble learning classifier which uses multiple decision trees. The Random Forest classifier 
#   is basically a modified bagging algorithm of a Decision Tree that selects the subsets differently.

from sklearn.ensemble import RandomForestClassifier

# n_estimators = number of decision trees
rf = RandomForestClassifier(n_estimators=10, max_depth=10)
rf.fit(X_train, y_train)

rftest = str(rf.score(X_test, y_test))
rftrain = str(rf.score(X_train, y_train))

print("score on test: " + rftest)
print("score on train: "+ rftrain)
print()

y_pred=rf.predict(X_test)
plot_confusion_matrix(rf, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
rfp0 = round(precision[0],2)
rfp1 = round(precision[1],2)
rfr0 = round(recall[0],2)
rfr1 = round(recall[1],2)
rff0 = round(fscore[0],2)
rff1 = round(fscore[1],2)
rfs0 = round(support[0],2)
rfs1 = round(support[1],2)


# # VotingClassifier

# In[58]:


#  evaluates different classifiers and selects the best

# The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a 
#   majority vote or the average predicted probabilities to predict the class labels. Such a classifier can be useful for 
#   a set of equally well performing models in order to balance out their individual weaknesses.

from sklearn.ensemble import VotingClassifier
# 1) naive bias = mnb
# 2) logistic regression =lr
# 3) random forest =rf
# 4) support vector machine = svm

evc=VotingClassifier(estimators=[('mnb',mnb),('lr',lr),('rf',rf),('svm',svm)],voting='hard')
evc.fit(X_train, y_train)

evctest = str(evc.score(X_test, y_test))
evctrain = str(evc.score(X_train, y_train))

print("score on test: " + evctest)
print("score on train: "+ evctrain)
print()

y_pred=evc.predict(X_test)
plot_confusion_matrix(evc, X_test, y_test, normalize='true')

print()
print(classification_report(y_test,y_pred))

# calculate precision, recall, f1, support to add to comparison dataframe
precision, recall, fscore, support = score(y_test, y_pred)
evcp0 = round(precision[0],2)
evcp1 = round(precision[1],2)
evcr0 = round(recall[0],2)
evcr1 = round(recall[1],2)
evcf0 = round(fscore[0],2)
evcf1 = round(fscore[1],2)
evcs0 = round(support[0],2)
evcs1 = round(support[1],2)


# # Model Performance with 20 features and SMOTE sampling

# In[59]:


# Calculate overfitting for each prediction model (train-test)
evcof = round((float(evctrain)-float(evctest)),4)  # overfitting
rfof = round((float(rftrain)-float(rftest)),4)  # overfitting
bgof = round((float(bgtrain)-float(bgtest)),4)  # overfitting
clfof = round((float(clftrain)-float(clftest)),4)  # overfitting
svmof = round((float(svmtrain)-float(svmtest)),4)  # overfitting
knnof = round((float(knntrain)-float(knntest)),4)  # overfitting
mnbof = round((float(mnbtrain)-float(mnbtest)),4)  # overfitting
lrof = round((float(lrtrain)-float(lrtest)),4)  # overfitting

columns = ['Voting Classifer','Random Forest','Bagging Classifier','Decision Tree','SVM','K Nearest Neighbor','Naive Bayes','Logistic Regression']
data = [evcof,rfof,bgof,clfof,svmof,knnof,mnbof,lrof]
prec0 = [evcp0,rfp0,bgp0,clfp0,svmp0,knnp0,mnbp0,lrp0]
prec1 = [evcp1,rfp1,bgp1,clfp1,svmp1,knnp1,mnbp1,lrp1]
rec0 = [evcr0,rfr0,bgr0,clfr0,svmr0,knnr0,mnbr0,lrr0]
rec1 = [evcr1,rfr1,bgr1,clfr1,svmr1,knnr1,mnbr1,lrp1]
f0 = [evcf0,rff0,bgf0,clff0,svmf0,knnf0,mnbf0,lrf0]
f1 = [evcf1,rff1,bgf1,clff1,svmf1,knnf1,mnbf1,lrf1]
s0 = [evcs0,rfs0,bgs0,clfs0,svms0,knns0,mnbs0,lrs0]
s1 = [evcs1,rfs1,bgs1,clfs1,svms1,knns1,mnbs1,lrs1]
of = pd.DataFrame([columns,data,prec0,prec1,rec0,rec1,f0,f1,s0,s1]).T
of.columns = ['model', 'over_fitting','Precision-0','Precision-1','Recall-0','Recall-1','F1-0','F1-1','Support-0','Support-1']
of = of.sort_values(by=['Precision-0'],ascending=False)
#of = of.sort_values(by=['over_fitting'],ascending=True)

of.head(10)


# ### Random Forest model has the highest precision and recall for both classes.
