#!/usr/bin/env python
# coding: utf-8

# ### 1. Load in data

# In[10]:


import sqlalchemy
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
import seaborn as sns
import scipy.cluster.hierarchy as hca

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[11]:


# initialize SQL engine
get_ipython().run_line_magic('reload_ext', 'sql')
get_ipython().run_line_magic('sql', 'postgres://capstone21_group4:nm_change.me@pgsql.dsa.lan/capstone21_group4')

engine = create_engine('postgres://capstone21_group4:nm_change.me@pgsql.dsa.lan/capstone21_group4')
print (engine.table_names())


# In[19]:


# Read in data
query = """ SELECT * FROM cleaned_prediction_data_v2_fips """
df = pd.read_sql(query, engine)
df = df.rename({'county_fips': 'fips'}, axis=1)
df.head()


# In[20]:


# Get state data
query = """ SELECT * FROM county_state_electoral_votes """
df2 = pd.read_sql(query, engine)


# In[21]:


def fill_zeros(x):    # function to pad with 0
    if len(x) < 6:
        return x.zfill(5)

df['fips'] = df['fips'].apply(str)
df['fips'] = df['fips'].apply(fill_zeros)
df2['fips'] = df2['fips'].apply(str)
df2['fips'] = df2['fips'].apply(fill_zeros)


# In[15]:


# Select a few columns
df2=df2[['fips', 'name', 'State', 'Votes', 'elec_vote_proportion']]

#Fill NAN values with 0
df2['elec_vote_proportion'] = df2['elec_vote_proportion'].fillna(0)


# In[16]:


# Combine this with state data
df3 = pd.merge(df, df2, on='fips')
df3.tail()


# ### Create/borrow an algorithm that predicts county swing chance using state level data

# In[8]:


# Create a dataframe that top 20 features correlated with 2020_result
df = df3[['2020_result', 'fips','2004-2016_dem_wins', '2016_gop_votes', 'elec_vote_proportion', 'num_owner_occupied_houses', 
         'percent_employed_natural_construct_main', 'education_high_school_25_64', 'perc_physically_inactive', 'percent_house_ss',
         'percent_house_inc_50000_74999', 'percent_employed_production_trans_material', 'pop_65_over', 'percent_employed_construction_16gr',
         'perc_adults_obesity', 'percent_employed_manufacturing_16gr', 'education_some_college_25_64']]


# In[9]:


# Logistic Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# set the indep and dep variable
X = df.loc[:, df.columns != '2020_result']
y = df.filter(['2020_result'])

# train and test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Apply regression
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

# drop index from y_test
y_test = pd.Series(y_test['2020_result'].values)
y_test = y_test.values

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, fmt ='', cmap='Blues')

print()
print('Prediction on 2020_result Accuracy: ',metrics.accuracy_score(y_test, y_pred))
lrtest = str(logistic_regression.score(X_test, y_test))
lrtrain = str(logistic_regression.score(X_train, y_train))
print("score on test: " + lrtest)
print("score on train: "+ lrtrain)


# ### From the output, predict the voting outcome of a specific county

# In[68]:


#returns how the logistic regression predicts the county voted
county_prediction = logistic_regression.predict_proba(X.loc[X['fips'] == '41039']) # Lane county where I live

#returns the actual way that this county voted
county_actual = df.loc[df['fips'] == '56039', '2020_result'].iloc[0]


# In[69]:


print(county_prediction)
print(county_actual)

# Note, the first value in 'county_prediction' tells you the likelihood of going dem and the second the likelihood of 
# going repub


# #### This logistic regression predicted that the county with fips 01001 would vote republican (2020_result=1) with 99% likelihood and turns out it did vote this way in 2020

# ### Which counties were predicted to vote Republican in 2020 and didn't?

# In[50]:


df['county_repub_likelihood'] = logistic_regression.predict_proba(X)[:,1]
df['county_dem_likelihood'] = logistic_regression.predict_proba(X)[:,0]


# In[47]:


print(logistic_regression.predict_proba(X))


# In[51]:


print(df['county_repub_likelihood'])


# In[52]:


# Select the rows that were likely to go republican, but didn't
df4 = df.loc[(df['county_repub_likelihood'] >= .8) & (df['2020_result'] == 0)]
df4.head()


# In[54]:


# Select the rows that were likely to go democrat, but didn't
df5 = df.loc[(df['county_dem_likelihood'] >= .8) & (df['2020_result'] == 1)]
df5.head()


# ### Map

# In[55]:


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

import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 300)  # allows to display all columns up to 300 in this case

# Magic adapters for Numpy
register_adapter(np.int64,AsIs)
register_adapter(np.float64,AsIs)


# In[56]:


get_ipython().system('pip install plotly --upgrade ')
get_ipython().system('pip install geojson')


# In[60]:


import pandas as pd
import plotly.express as px
import geojson

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

fig = px.choropleth(df, geojson=counties, locations='fips', color='county_repub_likelihood',
                           color_continuous_scale="Viridis", range_color=(0, 1), scope="usa")

fig.update_layout(
    title={
        'text': "Likelihood of Counties Voting Republican",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_layout(legend_title="Likelihood of Voting Republican")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[67]:


republican_counties = df.loc[(df['2020_result'] == 1)]

fig = px.choropleth(republican_counties, geojson=counties, locations='fips', color='county_repub_likelihood',
                           color_continuous_scale="Viridis", range_color=(0, 1), scope="usa")

fig.update_layout(
    title={
        'text': "Republican Counties (2020)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[63]:


dem_counties = df.loc[(df['2020_result'] == 0)]

fig = px.choropleth(dem_counties, geojson=counties, locations='fips', color='county_dem_likelihood',
                           color_continuous_scale="Viridis", range_color=(0, 1), scope="usa")

fig.update_layout(
    title={
        'text': "Democrat Counties (2020)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

