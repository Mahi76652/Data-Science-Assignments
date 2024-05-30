#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the dataset
import pandas as pd
df = pd.read_csv("C:\\Users\\PC-LENOVO\\Desktop\\ExcelR Assginments\\Neural Network\\gas_turbines.csv")


# In[2]:


df.info()


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


# Exploartory Data Analysis
# Display basic information about the dataset
print(df.info())


# In[6]:


# Display summary statistics
print(df.describe())


# In[7]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())


# In[8]:


# Data Visualisation
# Visualize the distribution of the target variable (TEY)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


plt.figure(figsize=(10, 6))
sns.histplot(df['TEY'], bins=30, kde=True)
plt.title('Distribution of TEY')
plt.xlabel('TEY')
plt.ylabel('Frequency')
plt.show()


# In[11]:


# Remove outliers from the TEY variable using IQR method
Q1 = df['TEY'].quantile(0.25)
Q3 = df['TEY'].quantile(0.75)


# In[12]:


IQR = Q3 - Q1


# In[13]:


df = df[(df['TEY'] >= Q1 - 1.5 * IQR) & (df['TEY'] <= Q3 + 1.5 * IQR)]


# In[14]:


# Visualize the distribution of the target variable after removing outliers
plt.figure(figsize=(10, 6))
sns.histplot(df['TEY'], bins=30, kde=True)
plt.title('Distribution of TEY (After Removing Outliers)')
plt.xlabel('TEY')
plt.ylabel('Frequency')
plt.show()


# In[15]:


# Display updated information about the dataset
print(df.info())


# In[16]:


# Pairplot to visualize relationships between variables
sns.pairplot(df[['TEY', 'AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO', 'NOX']])
plt.suptitle("Pairplot of Variables", y=1.02)
plt.show()


# In[17]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')


# In[18]:


# Define input features and target variable
X = df[['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO', 'NOX']]
Y = df['TEY']


# In[19]:


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[20]:


# Standardize the input features
from sklearn.preprocessing import StandardScaler


# In[21]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


# Build a neural network model
# Activation functions to compare
activation_functions = ['relu', 'sigmoid', 'softmax']


# In[23]:


# Results dictionary to store metrics for each activation function
results = {}


# In[24]:


import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[25]:


for activation_function in activation_functions:
    # Build a neural network model with different activation functions
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=activation_function, input_shape=(10,)),
        tf.keras.layers.Dense(32, activation=activation_function),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron for TEY prediction
    ])


# In[27]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[28]:


# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))


# In[29]:


# Evaluate the model
Y_pred = model.predict(X_test)    
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


# In[30]:


# Store results in the dictionary
results[activation_function] = {'mse': mse, 'mae': mae, 'r2': r2, 'history': history}


# In[31]:


# Visualize training and validation loss for each activation function
plt.figure(figsize=(12, 8))


# In[32]:


for activation_function in activation_functions:
    plt.plot(results[activation_function]['history'].history['val_loss'], label=f'{activation_function.capitalize()}')


# In[33]:


plt.title('Validation Loss for Different Activation Functions')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# In[34]:


# Print results
for activation_function in activation_functions:
    print(f"\nResults for {activation_function.capitalize()} Activation Function:")
    print("Mean Squared Error:", results[activation_function]['mse'])
    print("Mean Absolute Error:", results[activation_function]['mae'])
    print("R-squared:", (results[activation_function]['r2'] * 100).round(2))


# In[35]:


# The ReLU activation function performed very well, as indicated by the low MSE and MAE.
# The high R² value (99.86%) suggests that the model explains a significant portion of the variance in the target variable.
# The Sigmoid activation function also produced good results, with slightly higher MSE and MAE compared to ReLU.
# The R² value of 99.85% indicates that the model is effective in explaining the variance in the target variable.
# The Softmax activation function resulted in significantly worse performance compared to ReLU and Sigmoid.
# The extremely high MSE and MAE values, along with the negative R², suggest that the model with Softmax activation may not be suitable for this regression task.


# In[36]:


'''
Results for Relu Activation Function:
Mean Squared Error: 0.35738860106841275
Mean Absolute Error: 0.4350587044370937
R-squared: 99.86

Results for Sigmoid Activation Function:
Mean Squared Error: 0.38361762418854484
Mean Absolute Error: 0.43885821758432625
R-squared: 99.85

Results for Softmax Activation Function:
Mean Squared Error: 9275.121253038282
Mean Absolute Error: 94.99858753853655
R-squared: -3604.28'''


# In[37]:


# Both ReLU and Sigmoid activation functions are suitable for this regression task, with ReLU having a slight advantage in terms of lower MSE and MAE


# In[ ]:




