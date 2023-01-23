#!/usr/bin/env python
# coding: utf-8

# In[212]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.preprocessing
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
import ast  
from sklearn import metrics
import re
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter


# In[213]:


file = pd.read_excel(r"FinalData.xlsx")


# In[214]:


file.drop(columns='What is your name?', inplace=True)
file.drop(columns='What is your gender?', inplace=True)
file.drop(columns='What was your course in UG?', inplace=True)
file.drop(columns='What was the average CGPA or Percentage obtained in under graduation?', inplace=True)
#file.drop(columns='Have you done masters after undergraduation? If yes, mention your field of masters.(Eg, Masters in Mathematics)', inplace=True)


# In[215]:


file.rename(columns = {file.columns[0] :'UG_Course',
                       file.columns[1] :'Interest',
                       file.columns[2] :'Skills',
                       file.columns[3] :'Certification',
                       file.columns[4] :'Certificate_name',
                       file.columns[5] :'Working',
                       file.columns[6] :'JobTitle',
                       file.columns[7] :'Masters'
                      },inplace = True)


# In[216]:


file = file[file['Skills'].notna()]
file = file[file['Interest'].notna()]
file = file[file['Certificate_name'].notna()]
file = file[file['UG_Course'].notna()]


# In[217]:


df = file


# In[218]:


le = LabelEncoder()


# In[219]:


label_encoder = preprocessing.LabelEncoder()
LabelEncoder()


# In[220]:


def list_labelencode(column,file):
    column1 = list(file[column].explode().unique())
    label_encoder.fit(column1)
    file[column + '_encoded'] = file[column].apply(lambda x:label_encoder.transform(x))
    file[column + '_encoded']
    
def labelencoding(column,file):
    file[column + '_encoded'] = le.fit_transform(file[column])
    file[column + '_encoded'] 


# In[221]:


def recommend(d):
    global df
    
    d = pd.DataFrame([d])
    df = pd.concat([df, d], axis=0, ignore_index=True)
    
    df_last=df.iloc[[-1],]
    df['factors'] = df[['Interest','Skills']].apply("-".join, axis=1)
    
    i=0
    for i in range (len(df)):
        m_to_b = df.Certification[i]
        certi = df.Certificate_name[i]
        if m_to_b.startswith('Y') ==True:
            df.factors[i] = df.factors[i] + '-' + certi       
        i=i+1  
        
    i=0
    for i in range (len(df)):
        m_to_b = df.Working[i]
        job = df.JobTitle[i]
        if m_to_b.startswith('Y') ==True:
            df.factors[i] = df.factors[i] + '-' + job      
        i=i+1
        
    i=0
    for i in range (len(df)):
        m_to_b = df.Masters[i]
        if m_to_b.startswith('N') ==True:
            continue
        else:
            df.factors[i] = df.factors[i] + '-' + m_to_b     
        i=i+1  
        
    df_f = df.factors
    df_f.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    df_f = df_f.to_frame()
    df_f.columns=['factors']
    
    df_f['factors'] = df_f['factors'].str.lower()
    df_f['factors'] = df_f['factors'].replace(r'\s+', ' ', regex=True)
    df_f['factors'] = df_f['factors'].str.split(' ')
    
    df['UG_Course'].replace(" ","",regex=True, inplace=True)
    
    df_new = pd.merge(df, df_f, left_index=True, right_index=True)
    df_new.drop(df_new.columns[[1,2,3,4,5,6,7,8]], axis=1, inplace=True)
    df_new.rename(columns = {df_new.columns[1] :'factors'},inplace = True)
    
    labelencoding("UG_Course",df_new)
    list_labelencode("factors",df_new)
    df_new.drop(df_new.columns[[0,1]], axis=1, inplace=True)
    df_new.rename(columns = {df_new.columns[0] :'UG_Course',df_new.columns[1] :'factors'},inplace = True)
    
    df_new.replace("[^a-zA-Z0-9]"," ",regex=True, inplace=True)
    df_new['UG_Course'].replace(" ","",regex=True, inplace=True)
    
    df_d = pd.get_dummies(df_new['factors'].explode()).sum(level=0)
    
    df_new = pd.merge(df_new, df_d, left_index=True, right_index=True)
    df_new.drop(df_new.columns[[1]], axis=1, inplace=True)
    
    x = df_new.iloc[:,1:]
    y = df_new.iloc[:,[0]]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_train)
    x_pred = clf.predict(x_test)
    x_pred = list(x_pred)
    out = Counter(x_pred)

    output = sorted(out.items(), key=lambda item: item[1],reverse=True)
    output = output[:5]
    output = [item[0] for item in output]
    
    def out(output,col,file):
        le.fit(file[col])
        return le.inverse_transform(output)

    for i in out(output,'UG_Course',file):
        print(i)


# In[222]:


d={}
print("Enter your details:  \n")
nm = input("Enter your name:  ")
gen = input("Enter your gender:  ")
d['UG_Course']=''
d['Interest']=input("Enter the interests:  ")
d['Skills']=input("Enter the skills :  ")
d['Certification']=input("Did you do any certification courses additionally? :  ")
if (d['Certification']=='yes' or d['Certification']=='Yes'):
    d['Certificate_name']=input("If yes, please specify your certificate course title:  ")
d['Working']=input("Are you currently working or have you been emplyed before ? :  ")
if (d['Working']=='yes' or d['Working']=='Yes'):
    d['JobTitle']=input(" If yes, please enter Job Title :  ")
d['Masters']=input("Are you currently pursuing masters or have you persued higher education before :  ")


# In[223]:


a = recommend(d)

