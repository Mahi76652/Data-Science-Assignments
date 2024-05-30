#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Multiple Linear Regression\\50_Startups.csv")


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


list(df)


# In[8]:


#EDA --> Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


data = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']


# In[10]:


for column in data:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(f" Horizontal Box Plot of {column}")
    plt.show()


# In[11]:


#There are no outliers shown in the box plot
df.hist()


# In[12]:


df.skew()


# In[13]:


df.kurt()


# In[14]:


#Data Transformation in the Continuous variables
df_cont = df[["R&D Spend","Administration","Marketing Spend","Profit"]]


# In[15]:


df_cont.shape


# In[16]:


list(df_cont)


# In[17]:


#Now we perform transformation on Continuous Variable using Standardization
from sklearn.preprocessing import StandardScaler


# In[18]:


SS = StandardScaler()


# In[19]:


SS_X = SS.fit_transform(df_cont)


# In[20]:


SS_X = pd.DataFrame(SS_X)


# In[21]:


SS_X.columns = list(df_cont)


# In[22]:


SS_X


# In[23]:


#Now we perform transformation on categorical value
df_cat = df.iloc[:,3:4]


# In[24]:


df_cat.shape


# In[25]:


#We use Label Encoding for transformation
from sklearn.preprocessing import LabelEncoder


# In[26]:


LE = LabelEncoder()


# In[27]:


Y = LE.fit_transform(df["State"])


# In[28]:


Y = pd.DataFrame(Y)


# In[29]:


Y.columns = list(df.iloc[:,3:4])


# In[30]:


Y


# In[31]:


df_final = pd.concat([SS_X,Y],axis = 1)


# In[32]:


df_final.info()  #Now we have all the columns in continuous format


# In[33]:


#Splitting the variables
X = df_final.drop(df_final.columns[[3]], axis = 1)


# In[34]:


X.info


# In[35]:


X


# In[36]:


Y = df_final[df_final.columns[[3]]]


# In[37]:


Y.info()


# In[38]:


Y


# In[39]:


#We need to find the out the corelation between all the X variables
df_final.corr()


# In[40]:


#Here we have highest relation between R&D Spend and Profit so we will fit that with X and Y variables respectively
#using Linear Regression
#Fitting the Model
#MODEL 1
X = df_final[["R&D Spend"]]
Y = df_final["Profit"]


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


LR = LinearRegression()


# In[43]:


LR.fit(X,Y)


# In[44]:


LR.intercept_


# In[45]:


LR.coef_


# In[46]:


Y_pred = LR.predict(X)


# In[47]:


Y_pred


# In[48]:


#Metrices
from  sklearn.metrics import mean_squared_error


# In[49]:


mse = mean_squared_error(Y,Y_pred)


# In[50]:


print("Mean Squared Error:",mse.round(3))             #0.053    
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))     #0.231


# In[51]:


from sklearn.metrics import r2_score


# In[52]:


r2 = r2_score(Y,Y_pred) 
print("R-square:",r2.round(3))  #0.947


# In[53]:


""" VIF (variance influence factor is one of the metric which is used to calculate the relationship between  the two independent variables in order to see
there is a presence of multi collinearity, if exists it will effect the accuracy score of the model. so the vif factor ranges follows as below mentioned
VIF = 1/1-r2 ,VIF < 5 no multi collinearity
VIF : 5- 10  some multi collinearity issues will be present but can be accepted
VIF > 10 not at all acceptable """

'''i will check collinearty between "X" features ,based on that i will build models to get best R square value''' 


# In[55]:


#Checking collinearity between RDS,Administration
Y = df_final["R&D Spend"]
X = df_final[["Administration"]]


# In[56]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[57]:


LR.fit(X,Y)


# In[58]:


Y_pred = LR.predict(X)


# In[59]:


r2 = r2_score(Y,Y_pred)


# In[60]:


VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)    #VIF =1.06218  
# as VIF here between the mentioned variables is < 5 these can be taken together


# In[61]:


#adding the administration column to the R&D Spend 
#MODEL 2
Y = df_final["Profit"]    
X = df_final[["R&D Spend","Administration"]]


# In[62]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[63]:


LR.intercept_


# In[64]:


LR.coef_


# In[65]:


Y_pred = LR.predict(X)
Y_pred


# In[66]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3)) #0.052


# In[69]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) #0.228


# In[70]:


from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2     


# In[71]:


print("r2:",r2.round(3))  #0.948


# In[72]:


#Checking collinearity between R&D Spend,Administration,State
Y = df_final["R&D Spend"]
X = df_final[["State"]]


# In[73]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[74]:


Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)


# In[75]:


VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #1.0110804010836976 


# In[77]:


Y = df_final["State"]
X = df_final[["Administration"]]


# In[78]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[79]:


Y_pred = LR.predict(X)
r2 = r2_score(Y,Y_pred)


# In[80]:


VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  # 1.0001403759001628


# In[81]:


#MODEL 4
Y = df_final["Profit"]     
X = df_final[["R&D Spend","Administration","State","Marketing Spend"]]


# In[82]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[83]:


LR.intercept_


# In[84]:


LR.coef_


# In[85]:


Y_pred = LR.predict(X)
Y_pred


# In[86]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))     #MSE = 0.049


# In[87]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3))  #RMSE = 0.222


# In[88]:


from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred) 
r2


# In[89]:


print("r2:",r2.round(3))  #0.951


# In[90]:


#From all the above models, model 4 can be considered as best model as it has the least RMSE value and high R square value.

