#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the dataset
import pandas as pd
df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Association Rule\\book.csv",encoding='latin-1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# Exploratory Data Analysis (EDA)
# Display basic information about the dataset
print("Dataset Information:")
print(df.info())


# In[7]:


# Display summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())


# In[8]:


# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())


# In[9]:


# Visualize the distribution of numerical variables
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


plt.figure(figsize=(12, 8))


# In[11]:


for column in df.select_dtypes(include=['int64', 'float64']).columns:
    plt.subplot(2, 2, 2)
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')


# In[12]:


plt.tight_layout()
plt.show()


# In[13]:


# Visualize the correlation matrix for numerical variables
# This helps to identify relationships between variables
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[15]:


# Data Exploration
df.iloc[:,:].sum()
df.info()


# In[16]:


for i in df.columns:
    print(i)
    print(df[i].value_counts())
    print()


# In[17]:


"pip install wordcloud"


# In[18]:


pip install wordcloud


# In[19]:


from wordcloud import WordCloud


# In[20]:


plt.rcParams['figure.figsize'] = (10, 8)


# In[21]:


wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(df.sum()))


# In[22]:


plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()


# In[25]:


# 1. Association rules with 10% Support and 30% confidence
from mlxtend.frequent_patterns import apriori,association_rules


# In[24]:


get_ipython().system('pip install mlxtend')


# In[26]:


movies_10_30 = apriori(df, min_support=0.1, use_colnames=True)


# In[27]:


rules_10_30 = association_rules(movies_10_30, metric="confidence", min_threshold=0.3)


# In[28]:


# Display the association rules
print("Association Rules with 10% Support and 30% Confidence:")
print(rules_10_30)


# In[29]:


# Visualization of obtained rule
plt.figure(figsize=(10, 6))
sns.scatterplot(x=rules_10_30['support'], y=rules_10_30['confidence'])
plt.title('Association Rules (10% Support, 30% Confidence)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


# In[30]:


movies1 = apriori(df, min_support=0.1, use_colnames=True)
movies1


# In[31]:


rules = association_rules(movies1, metric="confidence", min_threshold=0.3)
rules


# In[32]:


rules.sort_values('lift',ascending=False)


# In[33]:


lift=rules[rules.lift>1]
lift


# In[34]:


matrix = lift.pivot('antecedents', 'consequents', 'lift')


# In[35]:


plt.figure(figsize=(12, 6), dpi=90)  # Adjusted the figsize
sns.heatmap(matrix, annot=True)
plt.title('HeatMap - For Lift Matrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()


# In[36]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])


# In[37]:


# Apriori algorithm is used to mine association rules with a minimum support of 10% and a minimum confidence of 30%.
# Association rules are displayed along with support, confidence, and lift values.
# A scatter plot visualizes the support-confidence relationship.


# In[38]:


# 2. Association rules with 15% Support and 50% confidence
movies2 = apriori(df, min_support=0.15, use_colnames=True)
movies2


# In[39]:


rules = association_rules(movies2, metric="confidence", min_threshold=0.5)
rules


# In[40]:


rules.sort_values('lift',ascending=False)


# In[41]:


lift=rules[rules.lift>1]
lift


# In[42]:


matrix = lift.pivot('antecedents','consequents','lift')


# In[43]:


plt.figure(figsize=(20,6),dpi=60)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[44]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])


# In[45]:


# Another set of association rules is mined with increased support (15%) and confidence (50%).
# Heatmap and scatter plot visualizations are provided for the obtained rules.


# In[46]:


#3. Association rules with 17% Support and 40% confidence
movies3 = apriori(df, min_support=0.17, use_colnames=True)
movies3


# In[47]:


rules = association_rules(movies3, metric="confidence", min_threshold=0.4)
rules


# In[48]:


rules.sort_values('lift',ascending=False)


# In[49]:


lift=rules[rules.lift>1]
lift


# In[50]:


matrix = lift.pivot('antecedents','consequents','lift')


# In[51]:


plt.figure(figsize=(20,6),dpi=250)
sns.heatmap(matrix,annot=True)
plt.title('HeatMap - ForLiftMatrix')
plt.yticks(rotation=0)
plt.xticks(rotation=90)


# In[52]:


# visualization of obtained rule
sns.scatterplot(x=rules['support'],y=rules['confidence'])


# In[ ]:


# Association rules are mined with different support (17%) and confidence (40%) thresholds.
# Heatmap and scatter plot visualizations are presented.

# Association rule mining is performed with varying support and confidence thresholds to discover interesting patterns in the data.
# The lift values in the association rules indicate the strength of relationships between items.
# Visualizations such as scatter plots and heatmaps help in better understanding the patterns and relationships revealed by the association rules.

