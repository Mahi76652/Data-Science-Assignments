#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Multiple Linear Regression\\ToyotaCorolla.csv",encoding = 'latin')
df.head()


# In[2]:


df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


# In[3]:


df.info()


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


# EDA #
#EDA----->EXPLORATORY DATA ANALYSIS
#BOXPLOT AND OUTLIERS CALCULATION #
import seaborn as sns
import matplotlib.pyplot as plt
df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


# In[7]:


for column in df:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
    #so basically we have seen the ouliers for each variable using seaborn#


# In[8]:


"""removing the ouliers"""
# List of column names with continuous variables
df = df[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
# Create a new DataFrame without outliers for each continuous column


# In[9]:


data_without_outliers = df.copy()


# In[10]:


for df.cloumns in df:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker_Length = Q1 - 1.5 * IQR
    upper_whisker_Length = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker_Length) & (data_without_outliers[column]<= upper_whisker_Length)]


# In[11]:


# Print the cleaned data without outliers
print(data_without_outliers)


# In[12]:


df = data_without_outliers
print(df)


# In[13]:


df.shape


# In[14]:


df.info() 


# In[16]:


#Histogram,skewness , KURTOSIS
df.hist()


# In[17]:


df.skew()


# In[18]:


df.kurt()


# In[19]:


df[df.duplicated()]


# In[20]:


df = df.drop_duplicates().reset_index(drop=True)


# In[21]:


df.describe()


# In[22]:


df


# In[23]:


#correlation analysis
df.corr()


# In[24]:


#Continous variables
df_cont = df.iloc[:,1:9]
df_cont.info()


# In[25]:


#Standardisation
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[26]:


SS_X = SS.fit_transform(df_cont)
SS_X


# In[27]:


X = pd.DataFrame(SS_X)
X.columns = list(df_cont)


# In[28]:


X


# In[29]:


# Y variable
Y_trans = df.iloc[:,0:1]
Y_trans


# In[30]:


list(Y_trans)


# In[31]:


from sklearn.preprocessing import StandardScaler
SS = StandardScaler()


# In[32]:


SS_Y = SS.fit_transform(Y_trans)
SS_Y


# In[33]:


Y = pd.DataFrame(SS_Y)
Y.columns = list(Y_trans)
Y


# In[34]:


#final transformed data
df_final = pd.concat([X,Y],axis = 1)
df_final


# In[35]:


#Data Visualisation
import seaborn as sns


# In[36]:


sns.set_style(style='darkgrid')
sns.pairplot(df)
sns


# In[37]:


# Pairplot to visualize the relationships between variables
import matplotlib.pyplot as plt


# In[38]:


sns.set_style(style='darkgrid')
sns.pairplot(df_final, vars=["Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"])
plt.show()


# In[39]:


# Correlation heatmap
import matplotlib.pyplot as plt


# In[40]:


plt.figure(figsize=(10, 8))
sns.heatmap(df_final.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[41]:


#correlation
pd.set_option('display.max_columns', None)	
df_final.corr()	


# In[42]:


# in multilinear regression we check every X variable's relation with the Y variable 
# --> here we keep on adding each x variable to our model one by one so then we can descide which model is best
# x variables = Age_08_04,KM,HP,cc,Doors,Gears,Quarterly_Tax,Weight
# Y variable = Price
# Model 1
Y = df_final["Price"]
X = df_final[["Age_08_04"]]


# In[43]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y) #b0+b1x1


# In[44]:


LR.intercept_ #b0


# In[45]:


LR.coef_ #b1


# In[46]:


#predicted values
Y_pred = LR.predict(X)
Y_pred


# In[47]:


#calculating sum of errors 
from sklearn.metrics import mean_squared_error


# In[48]:


error = mean_squared_error(Y, Y_pred)
print("MSE :",error.round(3))           #MSE : 0.226


# In[49]:


print("RMSE :",np.sqrt(error).round(3)) #RMSE : 0.476


# In[50]:


#r^2 error
from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)


# In[52]:


print("R square :",(r2*100).round(3))
# R square : 77.355


# In[53]:


""" checking for multi collineaity """
Y = df_final["Age_08_04"]
X = df_final[["KM"]]


# In[54]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[55]:


Y_pred = LR.predict(X)


# In[56]:


Y_pred


# In[57]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.3364447147688487


# In[58]:


Y = df_final["Age_08_04"]
X = df_final[["Weight"]]


# In[59]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[60]:


Y_pred = LR.predict(X)
Y_pred


# In[61]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.1793165628383648


# In[62]:


Y = df_final["KM"]
X = df_final[["Weight"]]


# In[63]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[64]:


Y_pred = LR.predict(X)
Y_pred


# In[65]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.0045541797596278


# In[66]:


# Model 2
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight"]]


# In[67]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[68]:


LR.intercept_


# In[69]:


LR.coef_


# In[70]:


Y_pred = LR.predict(X)
Y_pred


# In[71]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))


# In[72]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3))


# In[73]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[74]:


print("r2:",R2.round(3))   #0.835


# In[76]:


Y = df_final["HP"]
X = df_final[["Weight"]]


# In[77]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[79]:


Y_pred = LR.predict(X)
Y_pred


# In[80]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0000805312722458


# In[81]:


r2


# In[82]:


Y = df_final["KM"]
X = df_final[["HP"]]


# In[83]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[84]:


Y_pred = LR.predict(X)
Y_pred


# In[85]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.1246382490498912


# In[86]:


Y = df_final["Age_08_04"]
X = df_final[["HP"]]


# In[87]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[88]:


Y_pred = LR.predict(X)
Y_pred


# In[89]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0106168073460788


# In[90]:


# Model 3
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP"]]


# In[91]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[92]:


LR.intercept_


# In[93]:


LR.coef_


# In[94]:


Y_pred = LR.predict(X)
Y_pred


# In[95]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))


# In[96]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3))


# In[97]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[98]:


print("r2:",R2.round(3))  #r2: 0.841


# In[99]:


Y = df_final["Quarterly_Tax"]
X = df_final[["HP"]]


# In[100]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[101]:


Y_pred = LR.predict(X)
Y_pred


# In[102]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.100567159032836


# In[103]:


Y = df_final["Quarterly_Tax"]
X = df_final[["KM"]]


# In[104]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[105]:


Y_pred = LR.predict(X)
Y_pred


# In[106]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0872702145535558


# In[107]:


Y = df_final["Quarterly_Tax"]
X = df_final[["Weight"]]


# In[108]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[109]:


Y_pred = LR.predict(X)
Y_pred


# In[110]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.3846437869675032


# In[111]:


Y = df_final["Quarterly_Tax"]
X = df_final[["Age_08_04"]]


# In[112]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[113]:


Y_pred = LR.predict(X)
Y_pred


# In[114]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.000783881741269


# In[115]:


#MODEL 4
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax"]]


# In[116]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[117]:


LR.intercept_


# In[118]:


LR.coef_


# In[119]:


Y_pred = LR.predict(X)
Y_pred


# In[120]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))           #Mean Squared Error: 0.159


# In[121]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) # Root Mean Squared Error: 0.399


# In[122]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[123]:


print("r2:",R2.round(3))  #r2: 0.841


# In[124]:


Y = df_final["Doors"]
X = df_final[["KM"]]


# In[125]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[126]:


Y_pred = LR.predict(X)
Y_pred


# In[127]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)   #Variance Influence Factor:  1.0025465926246604Y = df_final["cc"]
X = df_final[["KM"]]


# In[128]:


Y = df_final["Doors"]
X = df_final[["Age_08_04"]]


# In[129]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[130]:


Y_pred = LR.predict(X)
Y_pred


# In[131]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0243321146000455


# In[132]:


Y = df_final["Doors"]
X = df_final[["Weight"]]


# In[133]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[134]:


Y_pred = LR.predict(X)
Y_pred


# In[135]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor: 1.2373203830220392


# In[136]:


Y = df_final["Doors"]
X = df_final[["HP"]]


# In[137]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[138]:


Y_pred = LR.predict(X)
Y_pred


# In[139]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:   1.0198511216020565


# In[140]:


Y = df_final["Doors"]
X = df_final[["Quarterly_Tax"]]


# In[141]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[142]:


Y_pred = LR.predict(X)
Y_pred


# In[143]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.0105380736840652


# In[145]:


Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors"]]


# In[146]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[147]:


LR.intercept_


# In[148]:


LR.coef_


# In[149]:


Y_pred = LR.predict(X)
Y_pred


# In[150]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)


# In[151]:


print("Mean Squared Error:",mse.round(3))
print(" Root Mean Squared Error:",np.sqrt(mse).round(3))


# In[152]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[153]:


print("r2:",R2.round(3))  #r2: 0.841


# In[154]:


Y = df_final["cc"]
X = df_final[["Age_08_04"]]


# In[155]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[156]:


Y_pred = LR.predict(X)
Y_pred


# In[157]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.0000005473633575


# In[158]:


Y = df_final["cc"]
X = df_final[["Quarterly_Tax"]]


# In[159]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[160]:


Y_pred = LR.predict(X)
Y_pred


# In[161]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF)  #Variance Influence Factor:  1.5933027300984508


# In[162]:


Y = df_final["cc"]
X = df_final[["HP"]]


# In[163]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[164]:


Y_pred = LR.predict(X)
Y_pred


# In[165]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.0016269295673288


# In[166]:


Y = df_final["cc"]
X = df_final[["KM"]]


# In[167]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[168]:


Y_pred = LR.predict(X)
Y_pred


# In[169]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.1473020467023822


# In[170]:


Y = df_final["cc"]
X = df_final[["Weight"]]


# In[171]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[172]:


Y_pred = LR.predict(X)
Y_pred


# In[173]:


r2 = r2_score(Y,Y_pred)
VIF = 1/(1-r2)
print("Variance Influence Factor: ",VIF) #Variance Influence Factor:  1.8431278653161849


# In[174]:


#MODEL 6
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc"]]


# In[175]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[176]:


LR.intercept_


# In[177]:


LR.coef_


# In[178]:


Y_pred = LR.predict(X)
Y_pred


# In[179]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))


# In[180]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3))


# In[181]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[182]:


print("r2:",R2.round(3))  #r2: 0.85


# In[183]:


""" as the Gears column consists of less correlation with all other independent variables multi collinearity doesnt exists
between any of them so we can consider Gears into model fitting"""

#MODEL 7
Y = df_final["Price"]
X = df_final[["Age_08_04","KM","Weight","HP","Quarterly_Tax","Doors","cc","Gears"]]


# In[184]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)


# In[185]:


LR.intercept_


# In[186]:


LR.coef_


# In[189]:


Y_pred = LR.predict(X)
Y_pred


# In[190]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("Mean Squared Error:",mse.round(3))                #Mean Squared Error: 0.149


# In[191]:


print(" Root Mean Squared Error:",np.sqrt(mse).round(3)) # Root Mean Squared Error:  0.386


# In[192]:


from sklearn.metrics import r2_score
R2 = r2_score(Y,Y_pred) 
R2


# In[193]:


print("r2:",R2.round(3))  #r2: 0.851


# In[ ]:


#From the above models we can conclude that from all the models applied above MODEL 6 can be considered as the best 
#model for the given dataset

