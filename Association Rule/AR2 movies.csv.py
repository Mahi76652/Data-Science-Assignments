#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
from sklearn.metrics import accuracy_score


# In[2]:


warnings.filterwarnings("ignore")


# In[3]:


from mlxtend.frequent_patterns import apriori,association_rules
import pandas as pd


# In[4]:


movies=pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Association Rule\\book.csv")


# In[5]:


df=movies.iloc[:,5:]


# In[6]:


df.head()


# In[7]:


# Data Exploration
df.iloc[:,:].sum()


# In[8]:


#checks the sum of each column to understand the frequency of each item in the dataset
df.info()


# In[9]:


for i in df.columns:
    print(i)
    print(df[i].value_counts())
    print()


# In[10]:


#Data Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[11]:


plt.rcParams['figure.figsize'] = (10, 8)


# In[12]:


wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(df.sum()))


# In[13]:


plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()


# In[14]:


#1. Association rules with 10% Support and 30% confidence
movies1 = apriori(df, min_support=0.1, use_colnames=True)
movies1


# In[15]:


rules = association_rules(movies1, metric="confidence", min_threshold=0.3)
rules


# In[16]:


rules.sort_values('lift',ascending=False)


# In[17]:


# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
lift=rules[rules.lift>1]
lift


# In[18]:


#Sorts the rules based on lift in descending order.
#Filters rules with a lift greater than 1, as lift measures the importance of the rule.
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


matrix = lift.pivot('antecedents','consequents','lift')


# In[20]:


plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[21]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])


# In[22]:


#2. Association rules with 20% Support and 50% confidence
movies2 = apriori(df, min_support=0.2, use_colnames=True)
movies2


# In[23]:


rules = association_rules(movies2, metric="confidence", min_threshold=0.5)
rules


# In[24]:


rules.sort_values('lift',ascending=False)


# In[25]:


lift=rules[rules.lift>1]
lift


# In[28]:


import matplotlib.pyplot as plt
matrix = lift.pivot('antecedents','consequents','lift')


# In[29]:


plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[30]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])


# In[31]:


#3. Association rules with 30% Support and 80% confidence
movies3 = apriori(df, min_support=0.3, use_colnames=True)
movies3


# In[32]:


rules = association_rules(movies3, metric="confidence", min_threshold=0.8)
rules


# In[33]:


rules.sort_values('lift',ascending=False)


# In[34]:


lift=rules[rules.lift>1]
lift


# In[35]:


import matplotlib.pyplot as plt
matrix = lift.pivot('antecedents','consequents','lift')


# In[36]:


plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[37]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])

"""Lift is a crucial metric in association rule mining. It measures how much more likely items are to be bought together compared to when they are bought independently. 

Lift > 1: Indicates that the occurrence of the antecedent increases the likelihood of the consequent occurring, suggesting a positive association.
Lift = 1: Implies that the antecedent and consequent are independent of each other.
Lift < 1: Suggests a negative association, meaning the occurrence of the antecedent decreases the likelihood of the consequent occurring."""


# In[ ]:




