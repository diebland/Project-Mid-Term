#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


# In[ ]:


def renaming(df):
    #removing special characters & following the snake case
    
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    df.columns = df.columns.str.replace('#_','')
    return df.info()


# In[ ]:


def categorical_information (df):
    for col in df.select_dtypes('object'):
        print (df[col].nunique(), '\n')
        print(df[col].value_counts(), '\n')


# In[ ]:


def count_plot_cat(df):
    for col in df.select_dtypes('object'):
        sns.countplot(df[col])
        plt.show()


# In[ ]:


def numerical_plotting(df):
    decimaux = df.select_dtypes('float64')
    entiers = df.select_dtypes('int64')
    
    for col in decimaux:
        sns.distplot(continuous[col])
        plt.show()
        
    for col in entiers:
        sns.countplot(discrete[col])
        plt.show()


# In[ ]:


def vizualizing_outliers(df):
    for col in df._get_numeric_data():
            sns.boxplot(df_num[col])
            plt.show()


# In[ ]:


def corr_matrix(df):
    corr_matrix=df.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(corr_matrix, annot=True)
    plt.show()


# In[ ]:


def count_plot_hue_target(df,columns=[], target = ''):
    for col in columns: 
        plt.figure(figsize=(6,4))
        sns.countplot(x = col, hue = target, data = df)
    plt.show()   


# In[ ]:


# function to perform ChiSquare-test for all (categorical) variables
def chi_square_test(df, columns=[]):
    for i in columns:
        for j in columns:
            if i != j:
                data_crosstab = pd.crosstab(df[i], df[j], margins = False)
                print('ChiSquare test for ',i,'and ',j,': ')
                print(chi2_contingency(data_crosstab, correction=False), '\n')


# In[ ]:


#Boxcox transformation

def boxcox_transform(df):
    numeric_cols = df.select_dtypes('np.number').columns
    _ci = {column: None for column in numeric_cols}
    for column in numeric_cols:
        # since i know any columns should take negative numbers, to avoid -inf in df
        df[column] = np.where(df[column]<=0, np.NAN, df[column]) 
        df[column] = df[column].fillna(df[column].mean())
        transformed_data, ci = stats.boxcox(df[column])
        df[column] = transformed_data
        _ci[column] = [ci] 
    return df, _ci


# In[ ]:


#checking distribution of variables

def distribution_distplot(df):
    for col in df.select_dtypes('float64'):
        sns.distplot(df[col])
        # save the figure
        # plt.savefig('covariance_account_balance.png', dpi=100, bbox_inches='tight')
        plt.show()


# In[ ]:


def logistic_regression_model(X_train, X_test, y_train, y_test):

    # defining a function to apply the logistic regression model
    
    classification = LogisticRegression(random_state=42, max_iter=10000)
    classification.fit(X_train, y_train)
    
    # and to evaluate the model
    score = classification.score(X_test, y_test)
    print('The accuracy score is: ', score, '\n')
      
    predictions = classification.predict(X_test)
    confusion_matrix(y_test, predictions)  
   
    
    cf_matrix = confusion_matrix(y_test, predictions)
    group_names = ['True NO', 'False NO',
               'False YES', 'True YES']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    print (cf_matrix)


# In[ ]:


def KNN_classifier_model(X_train, y_train, X_test, y_test,n):
    
    # define a function to apply the KNN Classifier Model
    
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    
    # and to evaluate the model
    
    print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
    
    y_pred = knn.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True NO', 'False NO',
               'False YES', 'True YES']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    print (cf_matrix)


# In[ ]:


#choose the best key value
def best_K(X_train, y_train, X_test, y_test, r):
    scores = []
    for i in r:
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
        
    plt.figure(figsize=(10,6))
    plt.plot(r,scores,color = 'blue', linestyle='dashed',
             marker='*', markerfacecolor='red', markersize=10)
    plt.title('accuracy scores vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Accuracy') 


# In[ ]:




