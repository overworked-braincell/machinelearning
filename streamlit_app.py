import streamlit as st
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#title
st.title(" (☞ﾟヮﾟ)☞ ML Project ☜(ﾟヮﾟ☜) ")
st.write("streamlit version = {}".format(st.__version__))
st.write("[docs.streamlit.io](https://docs.streamlit.io/).")

#load dataset
st.write("upload your dataset:")
file = st.file_uploader("Choose a CSV file", type="csv")
df = pd.read_csv(file)
st.write("Dataset Preview", df.head())

st.write("Drop Coloumns")
df = df.drop(df.columns[[0, 1, 2, 7, 8, 10, 11, 13, 14, 15,17, 18, 19]], axis=1)

st.write("Preview", df.head())

# st.write("Dataset Info", df.info())
st.write("Dataset Describe", df.describe())
# st.write("Dataset Correlation", df.corr())



# if file:
#     df = pd.read_csv(file)
#     st.write("Dataset Preview", df.head())

	#split data -- could ask to split the data
	# X = df.iloc[:, :-1]
	# y = df.iloc[:, -1]
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	#train model -- train model points off if you fit the wrong data
	# model = RandomForestClassifier()
	# model.fit(X_train, y_train)

	#Predict
	# st.write("Model Accuracy", model.score(X_test, y_test))

