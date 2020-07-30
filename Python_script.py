#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

df1 = pd.read_csv('train/First_Health_Camp_Attended.csv')
df2 = pd.read_csv('train/Health_Camp_Detail.csv')
df3 = pd.read_csv('train/Patient_Profile.csv')
df4 = pd.read_csv('train/Second_Health_Camp_Attended.csv')
df5 = pd.read_csv('train/Third_Health_Camp_Attended.csv')
df6 = pd.read_csv('train/Train.csv')


df1.dropna(axis=1, inplace=True)

df2['Camp_Start_Date'] = pd.to_datetime(df2['Camp_Start_Date'])
df2['Camp_End_Date'] = pd.to_datetime(df2['Camp_End_Date'])

df3['First_Interaction'] = pd.to_datetime(df3['First_Interaction'])

df6['Registration_Date'] = pd.to_datetime(df6['Registration_Date'])
df6.fillna('NaN', inplace=True)


# - No null values in First_Health_Camp_Attended
# - No null values in Health_Camp_Detail
# - NULL Values in Patient_Profile : filled NaN values of City_type and Employer_Category with 'unknown' value
# - No null in Second_Health_Camp_Attended
# - No null in Third_Health_Camp_Attended
# - No null in First_Health_Camp_Attended


#transforming categorical data to numerical using label Encoder in df2

le1 = preprocessing.LabelEncoder()
le1.fit(df2['Category1'])
df2['Category1'] = le1.transform(df2['Category1'])


le2 = preprocessing.LabelEncoder()
le2.fit(df2['Category2'])
df2['Category2'] = le2.transform(df2['Category2'])


#filled NaN values of City_type and Employer_Category with 'NaN' value
#Patient profile

df3['City_Type'].fillna('NaN', inplace=True)
df3['Employer_Category'].fillna('NaN', inplace=True)


attend_first = df1.groupby(['Patient_ID'], as_index=False).Health_Score.count()
attend_first.rename(columns={'Health_Score':'count'}, inplace=True)
# attend_first.head()

attend_second = df4.groupby(['Patient_ID'], as_index=False)['Health Score'].count()
attend_second.rename(columns={'Health Score':'count'}, inplace=True)
# total_attended_second.head()

attend_third = df5.groupby(['Patient_ID'], as_index=False).Number_of_stall_visited.count()
attend_third.rename(columns={'Number_of_stall_visited':'count'}, inplace=True)
# total_attended_third.head()

temp = pd.concat([attend_first, attend_second, attend_third], ignore_index=False)
new_temp = temp.groupby(['Patient_ID'], as_index=False)['count'].sum()

# Counting total Registrations for each Patient
temp = df6.groupby(['Patient_ID'], as_index=False).Registration_Date.count()
temp.rename(columns={'Registration_Date': 'Total_Registrations'}, inplace=True)
df6 = pd.merge(df6, temp, left_on='Patient_ID', right_on='Patient_ID', how='left')

# Counting total attended for each Patient

total_attended = pd.merge(df6, new_temp, left_on='Patient_ID', right_on='Patient_ID', how='left')
total_attended.fillna(0, inplace=True)
total_attended['count'] = total_attended['count'].astype(int)
total_attended.head()

total_attended['probability'] = total_attended['count']/total_attended['Total_Registrations']
total_attended.drop(columns=['Registration_Date', 'Total_Registrations', 'count'], inplace=True)
total_attended.head()


total_attended = pd.merge(total_attended, df2, left_on='Health_Camp_ID', right_on='Health_Camp_ID', how='left')
total_attended.head()

total_attended = pd.merge(total_attended, df3, left_on='Patient_ID', right_on='Patient_ID', how='left')
# total_attended.head()



data = total_attended[['Patient_ID', 'Health_Camp_ID', 'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'probability', 'Online_Follower', 
                       'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Category1', 'Category2', 'Category3']]


# # ############-------------MODEL BUILDING----------###############

# # Models
# 
# - Logistic Regression
# - Ridge Regression
# - Lasso Regression
# - Random Forest
# - SVM
# - NN
# - Decision Trees
# - MLPRegressor
# - LightGBM
# - GradientBoostingRegressor

from sklearn.model_selection import train_test_split

X = data.drop(columns=['probability', 'Patient_ID', 'Health_Camp_ID'])
y = data['probability']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# ## 1. Linear Regression


from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

model = LinearRegression().fit(X[['Category1', 'Var1', 'Category2', 'Var5', 'LinkedIn_Shared']], y)

y_pred = model.predict(X_val[['Category1', 'Var1', 'Category2', 'Var5', 'LinkedIn_Shared']])
y_score = mean_squared_error(y_val, y_pred)


# # 2. Ridge Regression

from sklearn.linear_model import Ridge

model = Ridge().fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)

# # 3. Random forest Classifier
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=10,
                              criterion='mse',
                              n_estimators =100)

model.fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)


# # 4. SVM

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

scale_X = StandardScaler()
scale_y = StandardScaler()

X = scale_X.fit_transform(X_train)
y=y_train

model = SVR(kernel='rbf', gamma='auto')
model.fit(X, y)
model.score(X, y)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)


# # 5. Decision Tree Regressor


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)


# # 6. Neural Network

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = data.drop(columns=['probability', 'Patient_ID', 'Health_Camp_ID'])
y = data['probability']

X = np.array(X[['Category1', 'Var1', 'Category2', 'Var5', 'LinkedIn_Shared']])
y = np.array(y)

model = keras.Sequential()
model.add(Dense(10, input_dim=5, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')
# model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
hist = model.fit(X, y, epochs=50, batch_size=1024,  callbacks=[callback], verbose=0)


# # 7. Lasso Regression

from sklearn import linear_model

model = linear_model.Lasso()

model.fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)

# # 8. MLP Regressor

from sklearn.neural_network import MLPRegressor

model = MLPRegressor()

model.fit(X_train, y_train)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)

# # 9. LightGBM regressor

import lightgbm

model = lightgbm.LGBMRegressor(learning_rate=0.001,
                               boosting_type='gbdt')

model.fit(X, y)
model.score(X_train, y_train)

y_pred = model.predict(X_val)
y_score = mean_squared_error(y_val, y_pred)

# # 10. GradientBoostingRegressor

from sklearn import ensemble

params = {'n_estimators': 100,
          'learning_rate': 0.001,
          'loss': 'ls'}


model = ensemble.GradientBoostingRegressor()

model.fit(X[['Category1', 'Var1']], y)


# # ---------Submission File------------


df_test = pd.read_csv('test.csv')
df_test = pd.merge(df_test, df2, left_on='Health_Camp_ID', right_on='Health_Camp_ID', how='left')
df_test = pd.merge(df_test, df3, left_on='Patient_ID', right_on='Patient_ID', how='left')


X_test = df_test.drop(columns=['Patient_ID', 'Health_Camp_ID', 'Registration_Date', 'Camp_Start_Date', 'Camp_End_Date', 
                               'Income', 'Education_Score', 'Age', 'First_Interaction', 'City_Type', 'Employer_Category'])


df_test['Outcome'] = model.predict(X_test[['Category1', 'Var1']])

sub = df_test.copy()
sub = sub[['Patient_ID', 'Health_Camp_ID', 'Outcome']]
sub['Outcome'] = sub['Outcome'].apply(lambda x: 0 if x<0 else x)
# sub['Outcome'] = round(sub['Outcome'], 3)
sub.to_csv('submission.csv', index=False)


# In[30]:





# In[32]:





# In[ ]:




