#!/usr/bin/env python
# coding: utf-8

# In[617]:


#Import all libraries as req for this model designing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier #For Random Forest model
import seaborn as sns # For grapths
from sklearn.model_selection import train_test_split # For spliting the dataset
from sklearn.metrics import accuracy_score 
from sklearn import linear_model # For linear regression
from sklearn import tree #For decision tree model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[618]:


os.getcwd() # To get the current working directory


# In[619]:


bike = pd.read_csv('day.csv') # Data loading


# In[620]:


bike.head()


# In[621]:


bike.shape # To get the shape of Data


# In[622]:


bike.columns # Get columns of data set


# # Insert Distribution plot for all variables

# In[623]:


sns.distplot(bike.cnt) 


# In[659]:


# Karnel density estimation (KDE) and regression trial with each variables.
sns.pairplot(bike,vars = ["cnt",'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered'] ,diag_kind= "kde",kind = "reg")


# In[625]:


sns.distplot(bike.casual) # This 'Casual' variable has outliers.


# In[626]:


print("Skewness: %f" % bike['cnt'].skew()) #Check skewness of the target variable


# In[627]:


sns.boxplot(x="weekday",y="cnt",data = bike)


# In[628]:


sns.boxplot(x="holiday",y = "cnt",data = bike)


# In[629]:


bike.describe()


# In[630]:


sns.boxplot(x="casual",data =bike) # Lot of outliers [Before Outlier removal]


# In[631]:


sns.scatterplot(x="casual",y="cnt",data =bike_1)
print ("Correlation before removal of outlier: %f" % bike_1.casual.corr(bike_1.cnt))
bike_1['casual'].shape


# In[632]:



cnames = ['casual']
for i in cnames:
    q75, q25 = np.percentile(bike.loc[:,i], [75 ,25])
    iqr = q75 - q25
     
min = q25 - (iqr*1.5)
max = q75 + (iqr*1.5)

bike_out = bike.copy()

bike_out = bike_out.drop(bike_out[bike_out.loc[:,i] < min].index)
bike_out = bike_out.drop(bike_out[bike_out.loc[:,'casual'] > max].index)


# In[633]:


bike_out.shape


# In[634]:


bike_out.casual.corr(bike_out.cnt)


# In[635]:


sns.boxplot(x="casual",data = bike_out) # After outlier removal


# # Decision Tree Regressor

# In[636]:


# Design of first model for Decision Tree Regressor
bike.head()


# In[637]:


bike.columns


# In[638]:


data1 =['temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt']

data1 = bike[data1]


# In[639]:


corr =data1.corr() #Finding correlation between numeric variables

corr.style.background_gradient(cmap='coolwarm')


# In[640]:


# Feature and target selection

features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt']

X= bike[features]

y= bike.cnt


# In[641]:


#Data spliting

train_X,test_X,train_y,test_y =train_test_split(X,y,random_state =1)


# In[642]:


#Decision tree model

first_model = tree.DecisionTreeRegressor(random_state =1)
first_model.fit(train_X,train_y)


# In[643]:


first_pred = first_model.predict(test_X)


# In[644]:


#Custom function to calculate MAPE

def MAPE(y_test,y_predict):
    mape = np.mean(np.abs(test_y-y_predict)/test_y)*100
    print("Mean absolute percentage error is: ",round(mape,2),"%")
    return mape


# In[645]:


#Custom function to calculate RMSE

def RMSE(test_y,y_predict):
    mse = np.mean((test_y-y_predict)**2)
    print("Mean Square : ",mse)
    rmse=np.sqrt(mse)
    print("Root Mean Square : ",rmse)
    return rmse


# In[646]:


# MAPE and RMSE score is very less which implies it's good model

MAPE(test_y,first_pred)
RMSE(test_y,first_pred)


# # Random Forest Regressor

# In[651]:


from sklearn.ensemble import RandomForestRegressor


# In[652]:


third_model = RandomForestRegressor(n_estimators= 500,random_state =1)
third_model.fit(train_X,train_y)


# In[653]:


third_pred = third_model.predict(test_X)


# In[654]:


# MAPE ADN RMSE score are good as compare to previous model

MAPE(test_y,third_pred)
RMSE(test_y,third_pred)


# # Linear Model Regression

# In[655]:


forth_model = linear_model.LinearRegression()
forth_model.fit(train_X,train_y)


# In[656]:


forth_pred = forth_model.predict(test_X)


# In[657]:


# The score of MAPE AND RMSE are veru good and list as mopare to above two models.

MAPE(test_y,forth_pred)
RMSE(test_y,forth_pred)


# In[658]:


new_bike = pd.DataFrame({'Predict': forth_pred,'Test':test_y})
new_bike.head()


# # Linear Model Regression is the best model for bike count prediction.
