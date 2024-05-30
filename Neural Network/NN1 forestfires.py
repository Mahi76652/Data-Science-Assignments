#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Loading the dataset
import pandas as pd
df=pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Neural Network\\forestfires.csv")
df


# In[3]:


# Checking for missing values
df.isna().sum()


# In[4]:


# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[5]:


df["month"]=LE.fit_transform(df["month"])


# In[6]:


df["day"]=LE.fit_transform(df["day"])


# In[7]:


df["size_category"]=LE.fit_transform(df["size_category"])


# In[8]:


df


# In[9]:


# Exploratory Data Analysis (EDA)
# Computing the correlation matrix
df.corr()


# In[10]:


# Splitting the dataset into features (X) and target variable (Y)
X=df.iloc[:,0:30]
X


# In[11]:


Y=df["size_category"]
Y


# In[12]:


# Scaling the features using StandardScaler
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()


# In[13]:


X=SS.fit_transform(X)
X=pd.DataFrame(X)
X


# In[14]:


# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42)


# In[15]:


get_ipython().system('pip install tensorflow')


# In[16]:


import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64,input_dim=X_train.shape[1],activation='relu',kernel_initializer="uniform"),
    tf.keras.layers.Dense(32, activation="linear",kernel_initializer="uniform",),
    tf.keras.layers.Dense(1)
])


# In[17]:


#Second Dense Layer: tf.keras.layers.Dense(32, activation="linear", kernel_initializer="uniform")

#This is the second dense layer with 32 units/neurons.
#The activation function used is linear, which means it performs a linear operation (no activation function applied).
#Once again, the weights are initialized using a uniform distribution.

#Output Layer: tf.keras.layers.Dense(1)

#This is the output layer with 1 unit, which is common for regression tasks where the goal is to predict a continuous value.
#Kernel Initializer: kernel_initializer="uniform" specifies how the weights of the neurons are initialized. In this case, they are initialized using a uniform distribution.


# In[18]:


# Compiling the model
model.compile(optimizer="adam",loss="mean_squared_error",metrics=["accuracy"])


# In[19]:


# Training the model
history=model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_split=0.3)


# In[22]:


loss=model.fit(X_train,Y_train)
loss   # loss: 0.0632 - accuracy: 0.9535


# In[23]:


# Plotting the training and validation loss
import matplotlib.pyplot as plt


# In[24]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("echo")
plt.ylabel("Loss")
plt.legend(["Train","validation"],loc="upper left")
plt.show()


# In[25]:





# In[26]:


# Plotting the training and validation accuracy
import matplotlib.pyplot as plt


# In[27]:


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.xlabel("Echo")
plt.ylabel("Accuracy")
plt.legend(["Trained","Validation"],loc="upper left")
plt.show()


# In[ ]:


# The loss and accuracy plots help visualize the training process. 
# In the loss plot, it's common to observe a decrease in training loss over epochs, while the validation 
#loss might stabilize or increase, indicating potential overfitting.
# The accuracy plot provides insights into how well the model generalizes to unseen data during training.

