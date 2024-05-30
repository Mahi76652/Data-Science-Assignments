#!/usr/bin/env python
# coding: utf-8

# # Importing the File

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Simple Linear Regression\\Salary_Data.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.dtypes


# In[6]:


df["Salary"]


# # Plotting the data in the graph

# In[7]:


import seaborn as sns


# In[8]:


sns.set_style(style = 'darkgrid')
sns.pairplot(df)


# In[9]:


sns.distplot(df["YearsExperience"])


# # Histogram

# In[10]:


df["YearsExperience"].describe()


# In[11]:


df["YearsExperience"].hist()


# In[12]:


df["YearsExperience"].skew()


# In[13]:


df["YearsExperience"].kurt()


# # Scatter Plot

# In[14]:


import matplotlib.pyplot as plt


# In[15]:


plt.scatter(x = df["YearsExperience"], y = df['Salary'])


# In[16]:


plt.scatter(x = df["YearsExperience"], y = df['Salary'], color = 'red')


# In[17]:


plt.show()


# # Correlation

# In[18]:


df[["Salary","YearsExperience"]].corr()


# In[19]:


df.corr()


# # BoxPlot

# In[20]:


df.boxplot(column = 'YearsExperience', vert = False)


# In[21]:


import numpy as np


# In[22]:


np


# In[23]:


Q1 = np.percentile(df["YearsExperience"],25)


# In[24]:


Q3 = np.percentile(df["YearsExperience"],75)


# In[25]:


IQR = Q3 - Q1


# In[26]:


LW = Q1 - (1.5*IQR)


# In[27]:


UW = Q3 + (1.5*IQR)


# In[30]:


df["YearsExperience"]>UW


# In[31]:


df[df["YearsExperience"]>UW]


# In[32]:


len(df[df["YearsExperience"]>UW])#No outliers present in the following data


# # Split the variable as X and Y

# In[33]:


X = df[['YearsExperience']]


# In[34]:


X[['YearsExperiencedSquared']] = X[['YearsExperience']]**2


# In[35]:


X[['SquareRootYearsExperience']] = np.sqrt(X[['YearsExperience']])


# In[36]:


X[['LogYearsExperience']] = np.log(X[['YearsExperience']])


# In[37]:


X[['InverseSquareRootYearsExperience']] = 1 / np.sqrt(X[['YearsExperience']])


# In[38]:


X


# In[39]:


Y = df["Salary"]


# In[41]:


Y


# # Data Partition

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, train_size = 0.75, random_state = 5)


# In[44]:


X_train


# In[45]:


X_test


# In[46]:


Y_train


# In[47]:


Y_test


# # Fitting the Model

# In[65]:


from sklearn.linear_model import LinearRegression


# In[66]:


LR = LinearRegression()


# In[67]:


LR.fit(X_train,Y_train) #Bo+B1x1


# In[68]:


LR.intercept_  # Bo


# In[69]:


LR.coef_   #B1


# In[70]:


#Calculating Y_pred
Y_pred = LR.predict(X)


# In[71]:


Y_pred


# In[72]:


Y


# # Contructing Metrices

# In[73]:


from sklearn.metrics import mean_squared_error


# In[74]:


mse = mean_squared_error(Y_pred,Y)


# In[75]:


print("Mean Sqaured Error:", mse.round(3))


# In[76]:


print("Root Mean squared Error:", np.sqrt(mse).round(3))  


# In[77]:


#R-squared(R2) value for the linear regression model
from sklearn.metrics import r2_score


# In[78]:


r2 = r2_score(Y,Y_pred)


# In[80]:


print("R Square:", r2.round(3)) 


# # Plotting the Scatter Plot

# In[81]:


import matplotlib.pyplot as plt


# In[82]:


plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =df["Salary"] )
plt.scatter(x = 1 / np.sqrt(X['YearsExperience']),y =Y_pred ,color = 'red' )
plt.plot(1 / np.sqrt(X['YearsExperience']),Y_pred,color='Black')   #x value is automatically taken by python###
plt.show()


# In[ ]:


# here in the above scenario, we applied a transformation on x variable and that transformation is 
#X['InverseSquareRootYearsExperience'] = 1 / np.sqrt(X['YearsExperience'])
#so after taking this transformed X value along with our target variable provides less mean squared error along with good R2 score.

