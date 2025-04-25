import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load or train the model (you can replace this with a saved model in future)
@st.cache_data
def train_model():
    data = load_iris()
    X, y = data.data, data.target
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf, data

model, iris = train_model()

# UI
st.title("🌸 Iris Flower Classifier (Decision Tree)")
st.write("Enter the flower measurements below:")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_class = iris.target_names[prediction]

st.success(f"Predicted Iris Species: **{predicted_class}**")
