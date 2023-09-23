#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

def entropy(probabilities):
    return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)

def information_gain(data, attri_index, target_index):
    T_instances = len(data)
    T_values = set(data[i][target_index] for i in range(T_instances))
    T_prob = [sum(1 for row in data if row[target_index] == value) / T_instances for value in T_values]
    entropy_be = entropy(T_prob)
    
    attri_values = set(data[i][attri_index] for i in range(T_instances))
    W_Entro_af = 0
    
    for value in attri_values:
        subset = [row for row in data if row[attri_index] == value]
        subset_size = len(subset)
        subset_target_probabilities = [sum(1 for row in subset if row[target_index] == target_value) / subset_size for target_value in T_values]
        W_Entro_af += (subset_size / T_instances) * entropy(subset_target_probabilities)
    
    info_gain = entropy_be - W_Entro_af
    return info_gain

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

target_index = 4
attributes = ["age", "income", "student", "credit_rating"]

information_gains = {}
for attribute_index, attribute in enumerate(attributes):
    gain = information_gain(data, attribute_index, target_index)
    information_gains[attribute] = gain

root_attribute = max(information_gains, key=information_gains.get)
root_information_gain = information_gains[root_attribute]

print(f"Information Gain of {root_information_gain:.3f} with Root node of '{root_attribute}' ")


# In[2]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

columns = ["age", "income", "student", "credit_rating", "buys_computer"]

data = pd.DataFrame(data, columns=columns)

X = data.drop("buys_computer", axis=1)
y = data["buys_computer"]

categorical_features = ["age", "income", "student", "credit_rating"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

pipeline.fit(X, y)

tree_depth = pipeline.named_steps["classifier"].get_depth()

print(f"Tree depth: {tree_depth}")


# In[3]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

columns = ["age", "income", "student", "credit_rating", "buys_computer"]

data = pd.DataFrame(data, columns=columns)

X = data.drop("buys_computer", axis=1)
y = data["buys_computer"]

cate_fea = ["age", "income", "student", "credit_rating"]
prepro = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), cate_fea)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", prepro),
    ("classifier", DecisionTreeClassifier())
])

pipeline.fit(X, y)

feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out(input_features=cate_fea)) + list(X.columns.drop(cate_fea))

plt.figure(figsize=(70, 20))
plot_tree(pipeline.named_steps["classifier"], filled=True, feature_names=feature_names, class_names=['no', 'yes'])
plt.show()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

X = data.drop(columns=['Filename', 'Label'])
y = data['Label']

Training_X, Testing_X, Training_y, Testing_y = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(Training_X, Training_y)

training_a = model.score(Training_X, Training_y)
print("Training Accuracy:", training_a)

testing_a = model.score(Testing_X, Testing_y)
print("Test Accuracy:", testing_a)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=[str(label) for label in model.classes_])
plt.show()


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)  

Training_X, Testing_X, Training_y, Testing_y = train_test_split(X, y, test_size=0.2, random_state=42)

cnames = data['Label'].unique().astype(str)  

model = DecisionTreeClassifier(max_depth=5)

model.fit(Training_X, Training_y)

train_a = model.score(Training_X, Training_y)
print("Training Accuracy :", train_a)

test_a = model.score(Testing_X, Testing_y)
print("Test Accuracy :", test_a)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=cnames.tolist())
plt.show()


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)

Training_X, Testing_X, Training_y, Testing_y = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=5)

model.fit(Training_X, Training_y)

train_a = model.score(Training_X, Training_y)
print("Training Accuracy :", train_a)

test_a = model.score(Testing_X, Testing_y)
print("Test Accuracy :", test_a)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=cnames.tolist())
plt.show()


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")


data = data.select_dtypes(include=[float, int])

X = data.drop(columns=['Label'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)

RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)

y_prediction_DT = DT.predict(X_test)
DT_accuracy = accuracy_score(y_test, y_prediction_DT)

y_prediction_RF = RF.predict(X_test)
RF_accuracy = accuracy_score(y_test, y_prediction_RF)


print("DT Matrix:\n", confusion_matrix(y_test, y_prediction_DT))
print("RF Matrix:\n", confusion_matrix(y_test, y_prediction_RF))
print("DT Report:\n", classification_report(y_test, y_prediction_DT))
print("RF Report:\n", classification_report(y_test, y_prediction_RF))
print("DT Accuracy:", DT_accuracy)
print("RF Accuracy:", RF_accuracy)

