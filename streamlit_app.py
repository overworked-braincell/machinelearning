import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split


#title
st.title(" (☞ﾟヮﾟ)☞ ML Project ☜(ﾟヮﾟ☜) ")
st.write("streamlit version = {}".format(st.__version__))
st.write("[docs.streamlit.io](https://docs.streamlit.io/).")

#load dataset
# st.write("upload your dataset:")
# file = st.file_uploader("Choose a CSV file", type="csv")
# df = pd.read_csv(file)

url = 'https://github.com/btran17/machinelearning/blob/main/data/application_record.csv?raw=true'
df = pd.read_csv(url)

st.write("Dataset Preview", df.head())

# st.write("Drop Coloumns")
# df = df.drop(df.columns[[]], axis=1)

st.write("Dataset Info", df.info())
st.write("Dataset Describe", df.describe())
# st.write("Dataset Correlation", df.corr())

#convert categorical values to numerical
# df['M'] = np.where(df['CODE_GENDER'] == 'M', 1, 0)
# df['Y'] = np.where(df['FLAG_OWN_CAR'] == 'N', 1, 0)
# df['Y'] = np.where(df['FLAG_OWN_REALTY'] == 'N', 1, 0)

# df['CODE_GENDER'].replace([0,1],['F','M'],inplace=True)
# df['FLAG_OWN_CAR'].replace([0,1],['Y','N'],inplace=True)
# df['FLAG_OWN_REALTY'].replace([0,1],['Y','N'],inplace=True)

df['CODE_GENDER'] = df['CODE_GENDER'].replace({'M': 0, 'F': 1})
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace({'N': 0, 'Y': 1})
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace({'N': 0, 'Y': 1})

st.write("Categorical to Numerical", df.head(5))






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

