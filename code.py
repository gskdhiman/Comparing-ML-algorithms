# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:03:28 2019

@author: GU389021

"""
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
import eli5
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix #,cohen_kappa_score
import pandas as pd
import numpy as np
import re

train_filepath = r'../dataset/titanic/train.csv'
test_filepath = r'../dataset/titanic/test.csv'
train_dataset = pd.read_csv(train_filepath)
test_dataset = pd.read_csv(test_filepath)
train_dataset.head()
test_dataset.head()

'''See how many null values in each feature'''
train_dataset.isnull().sum()
test_dataset.isnull().sum()

test_data_ID = test_dataset['PassengerId']

'''plots to visualise the dependencies'''
sns.jointplot(x='Age', y='Survived', data=train_dataset,kind = 'hex') 
sns.distplot(train_dataset['Fare'],kde=False,color = 'green')
sns.stripplot(x='Sex', y='Age', data=train_dataset, jitter=True, hue='Survived',dodge = True)


'''handling embarked field'''
values = {"S": 0, "C": 1, "Q": 2}
for df in [train_dataset,test_dataset]:
    top_value = df['Embarked'].describe()['top']
    df['Embarked'] = df['Embarked'].fillna(top_value)
    df['Embarked'] = df['Embarked'].map(values)


'''handle SibSp and Parch columns'''
for df in [train_dataset,test_dataset]:
    df['relatives'] = df['SibSp']+ df['Parch']
    df['relatives'] = df['relatives'].apply(lambda rel:1 if rel>1 else 0) 


'''handle passengerId column'''
#train_dataset.drop(['SibSp','Parch','PassengerId'],axis =1,inplace = True,errors ='ignore')
for df in [train_dataset,test_dataset]:
    df.drop(['PassengerId'],axis =1,inplace = True,errors ='ignore')


'''Handling age column'''
for df in [train_dataset,test_dataset]:
    #mean_Age = train_dataset['Age'].mean()

    mean_Age = df['Age'].describe()['mean'] 
    std_Age = df['Age'].describe()['std']
    null_Age_count = df['Age'].isnull().sum()
    age_null_indices = list(np.where(df['Age'].isna())[0])
    rand_fill = list(np.random.randint(mean_Age - std_Age, mean_Age + std_Age,size = null_Age_count))
    fill_na_custom = dict(zip(age_null_indices, rand_fill))
    df['Age'] = df['Age'].fillna(fill_na_custom)


'''lets see how cabin values play a role'''

for df in [train_dataset,test_dataset]:
    df['Cabin'].unique()
    df['Cabin'] = df['Cabin'].fillna(0)
    p = re.compile(r'([0-9]+)')
    df['Cabin_idx'] = df['Cabin'].apply(lambda x: int(p.search(str(x)).group(0)) if p.search(str(x)) else 0)
    df['Cabin_idx'] = df['Cabin_idx'].apply(lambda x : x//10)
    p = re.compile(r'([a-zA-z])')
    df['Deck'] = df['Cabin'].apply(lambda x: p.search(str(x)).group(0) if p.search(str(x)) else 'U')


z = list(train_dataset['Deck'].unique())
z.sort()

for df in [train_dataset,test_dataset]:
    cabin_transformer = dict(zip(z,list(range(len(z)))))
    df['Deck'] = df['Deck'].map(cabin_transformer)

sns.stripplot(x='Sex', y='Cabin_idx', data = train_dataset, jitter=True, hue='Survived',dodge = True)
sns.stripplot(x='Sex', y='Deck', data = train_dataset, jitter=True, hue='Survived',dodge = True)


'''handling sex field'''
gender_available = list(train_dataset['Sex'].unique())
for df in [train_dataset,test_dataset]:
    df['Sex'].describe()
    df['Sex'] = df['Sex'].map(dict(zip(gender_available,list(range(len(gender_available))))))

    df.drop(['Ticket'],axis =1,inplace = True,errors ='ignore')
    df.drop(['Name'],axis =1,inplace = True,errors ='ignore')
    df.drop(['Cabin'],axis =1,inplace = True,errors ='ignore')
    df['Fare'].fillna(value = df['Fare'].mean(),inplace  = True)
    fare = np.array(df['Fare'])
    df['Fare'] = normalize([fare]).T
    df.drop(['Fare'],axis =1,inplace = True,errors ='ignore')
    
    
train_features = train_dataset.drop("Survived", axis=1)
train_labels = train_dataset["Survived"]

test_features = test_dataset


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_features,train_labels)

perm = PermutationImportance(random_forest, random_state=1).fit(train_features,train_labels)
eli5.show_weights(perm, feature_names = train_features.columns.tolist())

Y_pred = random_forest.predict(test_features)

c_matrix = confusion_matrix(train_features,train_labels)
out_df = pd.DataFrame(columns = ['PassengerId','Survived'])
out_df['PassengerId'] =test_data_ID
out_df['Survived'] = Y_pred

submission_filepath ='gender_submission.csv'
out_df.to_csv(submission_filepath,index = False)

sns.heatmap(c_matrix.T,square = True,annot = True,
            fmt = 'g',cbar = True)
plt.xlabel('true labels')
plt.ylabel('predicted labels')
