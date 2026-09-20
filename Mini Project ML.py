#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[2]:


print("                                ~~~Welcome~~~")
dataset = input("Enter the name of the Dataset: ")


# In[3]:


df1 = pd.read_csv(dataset)
print(df1.columns)


# In[ ]:


print("Enter the unnecessary attributes to remove from the dataset, that you feel will not impact the Machine Learning model if removed!\n")
def remove_unnecessary_attributes():
    drop_columns = input("Enter the columns to drop, separated by commas: ")
    df2 = df1.drop(drop_columns.split(","), axis=1)
    print("\nUnnecessary Attributes removed!\n")
    return df2

uar = input("Enter if there are any unnecessary attributes (Y or N): ")
df2 = df1
if (uar == "Y" or uar == "y"):
    df2 = remove_unnecessary_attributes()
    
totalRows = df2.shape[0]
df2.head(5)


# In[5]:


def dataCleaning():
    percent1 = totalRows*0.01
    a = df2.isnull().sum()
    s = sum(list(a))
    if (s < percent1):
        global df3
        df3 = df2.dropna()
    else:
        print("Your data is missing beyond the Threshold of 1%, which might create discrepancies in model training. So, Clean the data and try again later!")
        exit()

df3 = df2
dataCleaning()
df3.head(5)


# In[6]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3 = df3.drop(['size'], axis=1)
df3.head(5)


# In[7]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[8]:


def convert_sqrt_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[9]:


def remove_Range():
    att = input("Enter name of such Attribute: ")
    print()
    df3_5 = (df3[~df3[att].apply(is_float)])
    print(df3_5.head(5))
    global df4
    df4 = df3.copy()
    df4[att] =df4[att].apply(convert_sqrt_to_num)
    print("\n\nThe range of the attribute has been modified to a single value as: \n")
    print(df4.head(3))
    
    
ch = input("\nEnter if an attribute contains a range of numerical value in a single row:(Y or N) ")
df4 = df3
if (ch=="y" or ch=="Y"):
    n1 = int(input("\nHow many such attributes are there? "))
    print()
    for i in range(n1):
        remove_Range()


# In[10]:


def remove_High_Dimensionality(column):
    global df5
    df5[col] = df5[col].apply(lambda x: x.strip())
    column_stats = df5.groupby(col)[col].agg("count").sort_values(ascending=False)
    column_stats_less_than_10 = column_stats[column_stats<=10]
    df5[col] = df5[col].apply(lambda x: 'other' if x in column_stats_less_than_10 else x)
   

df5 = df4.copy()
col = input("\nEnter the name of attribute whose values are to be converted to unique columns: ")
remove_High_Dimensionality(col)
df5.head(5)


# In[11]:


print("\nFor converting the data to Categorical Data, Hot-Encoding method is used!")
hot_encode_name = input("Enter the attribute name which you want to Hot-Encode: ")
dummies = pd.get_dummies(df5[hot_encode_name])
#print(dummies.head(5))
print("\n")
df6 = pd.concat([df5, dummies.drop('other', axis='columns')], axis='columns')
df7 = df6.drop(hot_encode_name, axis='columns')
#print(df7.head(5))


# In[12]:


l_null = list(df7.columns[df7.isna().any()])
if (len(l_null) != 0):
    print("Some columns in your Dataset still contain some Null Values as: ", l_null)
    print("\nIf the columns mentioned contain only Numerical values, then no need to worry at all.")
    null_ch = input("\nEnter if the Columns mentioned, contain only Numerical values (Y or N): ")
    if (null_ch == "y" or null_ch == "Y"):
        method = input("\nEnter the mathematical method that you want to use to fill missing values: Mean, Median or Mode? ")
        if (method == "mean" or method == "Mean" or method == "MEAN"):
            df7 = df7.fillna(df7.mean())
        elif (method == "mode" or method == "Mode" or method == "MODE"):
            df7 = df7.fillna(df7.mode())
        elif (method == "median" or method == "Median" or method == "MEDIAN"):
            df7 = df7.fillna(df7.median())
            
    else:
        print("However, if any one of the column contains Alphanumeric Values, then clean the Dataset manually and try again later :(")
        exit()


# In[13]:


dependant_variable = input("\nEnter the Dependant Variable (Column Name) that is to be predicted: ")
X = df7.drop(dependant_variable, axis='columns')
# X.shape
y = df7[dependant_variable]
# y.shape


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[15]:


score_accuracy = {
    "Linear Regression": 0,
    "Lasso Regression": 0,
    "Ridge Regression": 0,
    "Decision Tree Regression": 0,
    "Random Forest Regression": 0,
    "SVM": 0,
    "Gradient Boost": 0,
    "KNN": 0
}


# In[16]:


print("\nPLEASE WAIT, Be Patient while we are Processing your Data!\n")


# In[17]:


def Linear_Regression_model(X_train, y_train):
    from sklearn.linear_model import LinearRegression
    global lrm, model
    model = LinearRegression()
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Linear Regression"] = ms
    lrm = model
    
Linear_Regression_model(X_train, y_train)


# In[18]:


def Lasso_Regression_model(X_train, y_train):
    from sklearn.linear_model import Lasso
    global lsrm
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Lasso Regression"] = ms
    lsrm = model
    
Lasso_Regression_model(X_train, y_train)


# In[19]:


def Ridge_Regression_model(X_train, y_train):
    from sklearn.linear_model import Ridge
    global rrm
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Ridge Regression"] = ms
    rrm = model
    
Ridge_Regression_model(X_train, y_train)


# In[20]:


def DecisionTree_Regression_model(X_train, y_train):
    from sklearn.tree import DecisionTreeRegressor
    global drm
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Decision Tree Regression"] = ms
    drm = model
    
DecisionTree_Regression_model(X_train, y_train)


# In[21]:


def RandomForest_Regression_model(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    global rfrm
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Random Forest Regression"] = ms
    rfrm = model
    
RandomForest_Regression_model(X_train, y_train)


# In[22]:


def SupportVectorMachine_model(X_train, y_train):
    from sklearn.svm import SVR
    global svmm
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["SVM"] = ms
    svmm = model
    
SupportVectorMachine_model(X_train, y_train)


# In[23]:


def GradientBoosting_Regression_model(X_train, y_train):
    from xgboost import XGBRegressor
    global gbrm
    model = XGBRegressor()
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["Gradient Boost"] = ms
    gbrm = model
    
GradientBoosting_Regression_model(X_train, y_train)


# In[24]:


def KNN_Regression_model(X_train, y_train):
    from sklearn.neighbors import KNeighborsRegressor
    global knnrm
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    ms = model.score(X_test, y_test)*100
    score_accuracy["KNN"] = ms
    knnrm = model

KNN_Regression_model(X_train, y_train)


# In[25]:


m = max(score_accuracy.values())
mdl = max(zip(score_accuracy.values(), score_accuracy.keys()))[1]
print(score_accuracy)
print(mdl)


# In[34]:


import pickle
choice = input("\nDo you want model with Maximum Accuracy? (Y or N): ")
if (choice=="Y" or choice=="y"):
    if (mdl == "Linear Regression"):
        model = lrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "Lasso Regression"):
        model = lsrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "Ridge Regression"):
        model = rrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "Decision Tree Regression"):
        model = drm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "Random Forest Regression"):
        model = rfrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "SVM"):
        model = svmm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "Gradient Boost"):
        model = gbrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (mdl == "KNN"):
        model = knnrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)
            
else:
    print(score_accuracy, "\n")
    choice_model = input("\nEnter model of your choice from above dictionary: ")
    if (choice_model == "Linear Regression"):
        model = lrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "Lasso Regression"):
        model = lsrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "Ridge Regression"):
        model = rrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "Decision Tree Regression"):
        model = drm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "Random Forest Regression"):
        model = rfrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "SVM"):
        model = svmm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "Gradient Boost"):
        model = gbrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)

    elif (choice_model == "KNN"):
        model = knnrm
        with open('your_ML_model.pickle', 'wb') as f:
            pickle.dump(model,f)
            
import json
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps (columns))
    
print("\nCongratulations(:D), Your preferred model saved succesfully!")


# In[35]:


def predict_data(location, sqft, bath, bhk):
    loc_index = np.where (X.columns==location)[0][0]
    x = np.zeros(len (X. columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x]) [0]


# In[36]:


print(predict_data('1st Phase JP Nagar',1000, 2, 2))


# In[ ]:




