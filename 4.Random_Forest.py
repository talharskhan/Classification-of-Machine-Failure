import pandas as pd
import numpy as np
import streamlit as st 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from pickle import dump
from pickle import load


file=pd.read_csv("maintenance.csv")
col = ['UDI', 'Product ID', 'Rotational speed [rpm]', 'HDF', 'TWF', 'PWF', 'OSF', 'RNF']
file.drop(col, inplace = True, axis = 1)
file.head()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df =file
df.iloc[:, 0] = labelencoder.fit_transform(df.iloc[:, 0])
# st.subheader('User Input parameters')
# st.write(df)

smk = SMOTE()
X = df.iloc[:, 0:5]
y = df.iloc[:, 5]

X_res,y_res=smk.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25)

st.subheader('User Input parameters')
st.write(X_test)
X_test.shape,y_test.shape

model = RandomForestClassifier()
model.fit(X_train, y_train)

# save the model to disk
dump(model, open('Model.sav', 'wb'))

# load the model from disk
loaded_model = load(open('Model.sav', 'rb'))
prediction = loaded_model.predict(X_test)
prediction_proba = loaded_model.predict_proba(X_test)

result = ('Yes' if prediction_proba[0][1] > 0.5 else 'No')

# st.subheader('Predicted Result')
# st.write(result)

st.sidebar.subheader('Prediction Probability')
st.sidebar.write(prediction_proba)