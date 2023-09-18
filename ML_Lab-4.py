#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

# Load the data table
df = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")
df=df.drop(['Filename'], axis=1)
df

# Calculate the entropy of each attribute
def calculate_entropy(df, f1):
  """Calculates the entropy of a given attribute in a DataFrame.

  Args:
    df: A Pandas DataFrame.
    attribute: The name of the attribute to calculate the entropy for.

  Returns:
    The entropy of the attribute.
  """

  # Get the unique values of the attribute.
  unique_values = df[f1].unique()

  # Calculate the proportion of data points in each class.
  class_proportions = df[f1].value_counts() / df.shape[0]

  # Calculate the entropy.
  entropy = -sum(class_proportions * np.log2(class_proportions))

  return entropy

# Calculate the information gain of each attribute
def calculate_information_gain(df, attribute):
  """Calculates the information gain of a given attribute in a DataFrame.

  Args:
    df: A Pandas DataFrame.
    attribute: The name of the attribute to calculate the information gain for.

  Returns:
    The information gain of the attribute.
  """

  # Calculate the entropy of the root node.
  root_entropy = calculate_entropy(df, 'buys_computer')

  # Calculate the entropy of each child node.
  child_entropies = []
  for unique_value in df[f1].unique():
    child_df = df[df[f1] == unique_value]
    child_entropy = calculate_entropy(child_df, 'buys_computer')
    child_entropies.append(child_entropy)

  # Calculate the weighted average of the child entropies.
  weighted_child_entropy = np.sum(np.array(child_entropies) * (child_df.shape[0] / df.shape[0]))

  # Calculate the information gain.
  information_gain = root_entropy - weighted_child_entropy

  return information_gain

# Calculate the entropy and information gain for each attribute
entropies = {}
information_gains = {}
for attribute in df.columns:
  entropies[f1] = calculate_entropy(df, f1)
  information_gains[f1] = calculate_information_gain(df, f1)

# Print the results
print('Entropies:')
print(entropies)
print('Information gains:')
print(information_gains)

# Identify the best attribute to split on
best_attribute = None
highest_information_gain = 0
for attribute, information_gain in information_gains.items():
  if information_gain > highest_information_gain:
    best_attribute = attribute
    highest_information_gain = information_gain

# Print the best attribute to split on
print('The best attribute to split on is:', best_attribute)


# In[ ]:




