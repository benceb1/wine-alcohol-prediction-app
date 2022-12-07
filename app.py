import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Red-Wine Alcohol Prediction App
""")
st.write('---')

# I found this method to download from my github
dataset = pd.read_csv('https://raw.githubusercontent.com/benceb1/csv_and_others/master/winequality-red.csv')
X = dataset.drop('alcohol', axis=1)
Y = dataset['alcohol']


st.sidebar.header('Specify Input Parameters')

def create_object_field_and_slider_value(column_name):
  return { column_name: st.sidebar.slider(column_name, float(int(X[column_name].min())), float(round(X[column_name].max())), float(round(X[column_name].mean())), 0.01) }

def user_input_features():
  field_elements = map(create_object_field_and_slider_value, X.columns)
  data = {}

  for element in field_elements:
    data.update(element)

  features = pd.DataFrame(data, index=[0])

  return features

df = user_input_features()


# Main Panel


st.header('Specified Input parameters')
st.write(df)
st.write('---') 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)


model = RandomForestRegressor()
model.fit(X_train, y_train)


prediction = model.predict(df)
y_pred = model.predict(X_test)

st.header('Prediction of ALCOHOL')
st.write(prediction)
st.write('---')

st.header('Mean Squared Error:')
st.write(mean_squared_error(y_test, y_pred))
st.write('---')

