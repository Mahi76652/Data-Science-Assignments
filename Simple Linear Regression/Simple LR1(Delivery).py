#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import numpy as np


# In[164]:


df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Simple Linear Regression\\delivery_time.csv")


# In[165]:


df.head()


# In[166]:


df.info()


# In[167]:


df.describe()


# In[168]:


df.dtypes


# In[169]:


df.isnull().sum()  #there are no null values


# In[170]:


df["Delivery Time"]


# In[171]:


df["Sorting Time"]


# # Data Visualization

# In[172]:


import seaborn as sns


# In[173]:


sns.set_style(style='darkgrid')
sns.pairplot(df)


# In[174]:


#Distribution plot
sns.displot(df["Sorting Time"])


# # Histogram

# In[175]:


df["Delivery Time"].describe()


# In[176]:


df["Delivery Time"].hist()


# In[177]:


df["Delivery Time"].skew()  #0.3523900822831107


# In[178]:


df["Delivery Time"].kurt()  #0.31795982942685397


# In[179]:


df["Sorting Time"].describe()


# In[180]:


df["Sorting Time"].hist()


# In[181]:


df["Sorting Time"].skew()  #0.047115474210530174


# In[182]:


df["Sorting Time"].kurt()  #-1.14845514534878


# # Scatter Plot

# In[183]:


import matplotlib.pyplot as plt


# In[184]:


plt.scatter(x=df['Sorting Time'], y=df['Delivery Time'],color='red')
plt.show()


# In[185]:


df.corr()


# In[186]:


#Correlation
df[['Delivery Time','Sorting Time']].corr()


# In[187]:


df.corr()


# # Box plot 

# In[188]:


df.boxplot(column='Sorting Time',vert=False)


# In[189]:


Q1 = np.percentile(df["Sorting Time"],25)


# In[190]:


Q3 = np.percentile(df["Sorting Time"],75)


# In[191]:


IQR = Q3 - Q1


# In[192]:


LW = Q1 - (1.5*IQR)


# In[193]:


UW = Q3 + (1.5*IQR)


# In[194]:


df["Sorting Time"]>UW


# In[195]:


df[df["Sorting Time"]>UW]


# In[196]:


len(df[df["Sorting Time"]>UW]) #There are no outliers


# In[197]:


#Split the variables into X and Y
X = df[["Sorting Time"]]


# In[198]:


X


# In[199]:


#Transformation on X variables
#X[["Sorting Time"]] = X[["Sorting Time"]]**2
X[["Sorting Time"]] = np.sqrt(X[["Sorting Time"]])  #This X variable gives the most accurate model for the current dataset
#X[["Sorting Time"]] = np.log(X[["Sorting Time"]])
#X[["Sorting Time"]] = 1/np.sqrt(X[["Sorting Time"]])


# In[200]:


X


# In[201]:


df.dropna(subset=['Sorting Time'], inplace = True)


# In[202]:


Y = df["Delivery Time"]


# In[203]:


Y


# In[204]:


#Split the variable into training and testing set
from sklearn.model_selection import train_test_split


# In[205]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.75, random_state = 25)


# In[206]:


#Fitting the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[207]:


LR.fit(X,Y)  #Bo+B1x1


# In[208]:


LR.intercept_


# In[209]:


LR.coef_


# In[210]:


#Prediction
Y_pred = LR.predict(X)


# In[211]:


Y_pred


# In[212]:


#Actual
Y


# In[213]:


from sklearn.metrics import mean_squared_error


# In[214]:


mse = mean_squared_error(Y,Y_pred)


# In[215]:


print("Mean Square Error:", mse.round(3))


# In[216]:


print("Root Mean squared Error:", np.sqrt(mse).round(3))


# In[217]:


from sklearn.metrics import r2_score


# In[218]:


r2 = r2_score(Y,Y_pred)


# In[219]:


print("R-square:",r2.round(3))


# In[220]:


# to fix the accuracy scores of training and testing data we need to take validation set approach
training_error = []
testing_error = []


# In[221]:


for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, train_size = 0.75, random_state = i)
    X_train.shape
    X_test.shape
    LR.fit(X_train,Y_train)
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)
    training_error.append(np.sqrt(mean_squared_error(Y_train,Y_pred_train)))
    testing_error.append(np.sqrt(mean_squared_error(Y_test,Y_pred_test)))


# In[222]:


print(training_error)


# In[223]:


print(testing_error)


# In[224]:


print("average training error :",np.mean( training_error).round(3))


# In[225]:


print("average testing error :",np.mean( testing_error).round(3))


# In[226]:


# metrics
mse = mean_squared_error(Y,Y_pred)


# In[227]:


print("Mean squared Error:", mse.round(3))


# In[228]:


print("Root Mean squared Error:", np.sqrt(mse).round(3))


# In[229]:


r2 = r2_score(Y,Y_pred)
print("R square:", r2.round(3))


# In[230]:


#The current model is showing low accuracy for the current data we are having
#So this is a poor model for the current dataset


# In[ ]:




