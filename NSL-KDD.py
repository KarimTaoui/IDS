import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image
import pydotplus
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
import openpyxl



st.set_page_config(
    page_title="NSL-KDD",   
    layout="wide",
    page_icon=":chart_with_upwards_trend:"
)




st.header("Dataset description")
data_description=pd.read_excel("Description_features.xlsx")
data_description = data_description.set_index('#', drop=True)
data_description = data_description.drop(data_description.index[-1])
st.write(data_description)

data=pd.read_csv("kdd_train.csv")

st.write(f"Les valeurs nuls : {data.isnull().any().sum()}")
st.write(f"Les valeurs dupliqués : {data.duplicated().sum()}")
le = LabelEncoder()
# Encode non-numeric values as integers
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])

st.header('Our Data')
st.write(data)

st.header("Informations about Our Data")
st.write(data.describe())
s = data.drop('class', axis=1)
y = data['class']


# Select the k best features based on chi-squared test
k =21  # number of top features to select
selector = SelectKBest(chi2, k=k)
selector.fit(s, y)

# Get the indices of the selected features
selected_features_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features_names = s.columns[selected_features_indices]

# Create a new data frame with only the selected features
X = data[selected_features_names]


features = X.columns
# Create a StandardScaler object
scaler = StandardScaler()
# Fit the scaler on the data
scaler.fit(X[features])
# Scale the data and replace the original values with the scaled values
X[features] = scaler.transform(X[features])
st.header("Dataset after features selections and scalling")
st.write(X)



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Build a decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", accuracy)

# input_data=(0,1,30,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,144,8,1.0,0.0,0.06,0.05,0.0,255,8,0.03,0.05,0.0,0.0)
# changed_data=np.asarray(input_data)
# reshaped_data=changed_data.reshape(1,-1)
# prediction=model.predict(reshaped_data)
# st.write(prediction)

text_representation = tree.export_text(model)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

dot_data = tree.export_graphviz(model, out_file=None, 
                      feature_names=X.columns,  
                      class_names=['normal', 'anomaly'],  
                      filled=True, rounded=True,  
                      special_characters=True)

# Visualize the decision tree
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('decision_tree', view=True)