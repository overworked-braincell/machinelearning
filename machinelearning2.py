import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Intern Game Day", layout="wide")
st.title("(☞ﾟヮﾟ)☞ Machine Learning ☜(ﾟヮﾟ☜)")
st.caption("streamlit version = {}".format(st.__version__))
# st.write("[docs.streamlit.io](https://docs.streamlit.io/)")
st.write("  =================================================================  ")
st.header(" 🧠 Credit Score Classification 🧠 ")
st.write("  =================================================================  ")

st.subheader("\n 0-1. Define the url and copy the correct link from your repository\n")
st.caption("HINT: https://github.com/username/repo/blob/branch/filename.csv")

if "df" not in st.session_state:
    st.session_state.df = None
if "preprocessed_df" not in st.session_state:
    st.session_state.preprocessed_df = None
if "link" not in st.session_state:
    st.session_state.link = ""

def github_url(url):
    if "github.com" in url and "blob" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
    return url

link = st.text_input("TODO: Paste the GitHub CSV Link")
st.session_state.link = link  # update session state with the current input

if st.button("Load Data") and link:
    try:
        url = github_url(st.session_state.link)
        df = pd.read_csv(url)
        ###########################################################################

        # df.describe(include='object').T
        # df.dtypes
        # st.write('total missing values', df.isnull().sum())  # total missing values

        df = df.replace("_", "", regex=True)
        # df[df.eq('').any(axis=1)] #find empty strings

        # drop columns that won't be used
        df.drop(['Interest_Rate', 'Payment_of_Min_Amount', 'Credit_Utilization_Ratio', 'Credit_History_Age',
                 'Num_Credit_Inquiries', 'Month', 'Name', 'SSN', 'Changed_Credit_Limit',
                 'Monthly_Inhand_Salary', 'Total_EMI_per_month'], axis=1, inplace=True)

        # Age - convert object to int
        # st.write('Age: ', df['Age'].value_counts())
        # st.write('Missing Age Values: ', df['Age'].isnull().sum())  # total missing values
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df = df.dropna(subset=['Age'])
        df['Age'] = df['Age'].astype(int)

        # Occupation - replace blank values with 'other'
        # st.write('occupation: ', df.Occupation.value_counts())
        # st.write('Occupation Data at 8: ', df['Occupation'][8])
        df['Occupation'].replace(to_replace='', value='Other', inplace=True)
        # df['Occupation'].replace(to_replace='_______', value='Other', inplace=True)
        df['Occupation_Code'] = df['Occupation']
        df['Occupation'], unqiue = pd.factorize(df['Occupation'])
        # st.write('Occupation Code: ', df['Occupation_Code'].value_counts())
        # st.write('Occupation : ', df['Occupation'].value_counts())
        df.drop('Occupation_Code', axis=1, inplace=True)

        # Annual_Income - convert object to ?
        # st.write('Annual_Income Values: ', df['Annual_Income'].value_counts(ascending=True))
        # st.write('Annual_Income Data Type: ', df['Annual_Income'].dtype)
        df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')
        df['Annual_Income'] = df['Annual_Income'].round(2)
        # st.write('Annual_Income NA Values: ', df['Annual_Income'].isna().sum())
        # st.write('Annual_Income < 0', df[df['Annual_Income'] < 0]['Annual_Income'].count())

        # Monthly_Inhand_Salary
        # st.write('Monthly_Inhand_Salary Values: ', df['Monthly_Inhand_Salary'].value_counts(ascending=True))
        # st.write('Monthly_Inhand_Salary Data Type: ', df['Monthly_Inhand_Salary'].dtype)
        # df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].round(2)
        # st.write('Monthly_Inhand_Salary NA Values: ', df['Monthly_Inhand_Salary'].isna().sum())
        # st.write('Monthly_Inhand_Salary < 0', df[df['Monthly_Inhand_Salary'] < 0]['Monthly_Inhand_Salary'].count())

        # Num_of_Loan - convert object to int; remove anything above x amount that seems unreasonable
        # st.write('Num_of_Loan Values: ', df['Num_of_Loan'].value_counts(ascending=True))
        # st.write('Num_of_Loan NA Values: ', df['Num_of_Loan'].isna().sum())
        df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce')
        df['Num_of_Loan'] = df['Num_of_Loan'].astype(int)
        df.drop(df[df['Num_of_Loan'] > 50].index, inplace=True)
        df.drop(df[df['Num_of_Loan'] < 0].index, inplace=True)
        # st.write('Num_of_Loan Values: ', df['Num_of_Loan'].value_counts(ascending=True))

        # Num_of_Delayed_Payment - replace missing values with median (too many outliers); convert object to int
        # st.write('Num_of_Delayed_Payment: ', df['Num_of_Delayed_Payment'].value_counts())
        df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce')
        # fig, ax = plt.subplots()
        # sns.boxplot(data=df, x=df['Num_of_Delayed_Payment'])
        # st.pyplot(fig)
        df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'] \
            .fillna(df['Num_of_Delayed_Payment'].median()).astype(
            int)  # fill missing data with median b/c too many outliers

        # Credit_Mix - replace missing/blank values with 'unknown'
        # df['Credit_Mix'].value_counts()
        # st.write('Credit_Mix Null Values: ', df['Credit_Mix'].isnull().sum())
        df['Credit_Mix'].replace(to_replace='', value='Unknown', inplace=True)
        df['Credit_Mix'].replace(to_replace='_', value='Unknown', inplace=True)
        # st.write('Credit_Mix', df['Credit_Mix'].value_counts())
        df['Credit_Mix_Code'] = df['Credit_Mix']
        df['Credit_Mix'], unqiue = pd.factorize(df['Credit_Mix'])
        # st.write('Credit_Mix Code: ', df['Credit_Mix_Code'].value_counts())
        # st.write('Credit_Mix : ', df['Credit_Mix'].value_counts())
        df.drop('Credit_Mix_Code', axis=1, inplace=True)

        # Outstanding_Debt
        # st.write('Outstanding_Debt: ', df['Outstanding_Debt'].value_counts())
        # st.write('Outstanding_Debt NA Values: ', df['Outstanding_Debt'].isna().sum())
        # st.write('Annual_Income Data Type: ', df['Annual_Income'].dtype)

        # Amount_invested_monthly - missing values; convert object to numeric (float)
        # st.write('Amount_invested_monthly: ', df['Amount_invested_monthly'].value_counts())
        # st.write('Amount_invested_monthly Values: ', df['Amount_invested_monthly'].value_counts(ascending=False))
        # st.write('Amount_invested_monthly NA Values: ', df['Amount_invested_monthly'].isna().sum())
        df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'], errors='coerce')
        # st.write('data type: Amount_invested_monthly: ', df['Amount_invested_monthly'].dtype)
        # fig, ax = plt.subplots()
        # ax.set_title("Amount_invested_monthly")
        # ax.set_xlabel("Amount_invested_monthly")
        # ax.set_ylabel("Frequency")
        # sns.histplot(df['Amount_invested_monthly'], bins=10, kde=True)
        # st.pyplot(fig)
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'] \
            .fillna(df['Amount_invested_monthly'].median()).astype(float)  # fill missing value with the median
        df['Amount_invested_monthly'] = df['Amount_invested_monthly'].round(2)
        # st.write('Amount_invested_monthly NA Values: ', df['Amount_invested_monthly'].isna().sum())
        # st.write('Amount_invested_monthly: ', df['Amount_invested_monthly'].head())

        # Payment_Behavior
        # st.write('Payment_Behaviour: ', df['Payment_Behaviour'].value_counts())
        df['Payment_Behaviour'].replace(to_replace='!@9#%8', value='Undetermined', inplace=True)
        # st.write('Payment_Behaviour: ', df['Payment_Behaviour'].value_counts())
        # st.write('Payment_Behaviour NA Values: ', df['Payment_Behaviour'].isna().sum())
        df['Payment_Behaviour_Code'] = df['Payment_Behaviour']
        df['Payment_Behaviour'], unqiue = pd.factorize(df['Payment_Behaviour'])
        # st.write('Payment_Behaviour Code: ', df['Payment_Behaviour_Code'].value_counts())
        # st.write('Payment_Behaviour : ', df['Payment_Behaviour'].value_counts())
        df.drop('Payment_Behaviour_Code', axis=1, inplace=True)

        # Monthly_Balance - missing values; convert object to numeric
        # st.write('Monthly_Balance Values: ', df['Monthly_Balance'].value_counts(ascending=True))
        # st.write('Monthly_Balance Data Type: ', df['Monthly_Balance'].dtype)
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
        df['Monthly_Balance'] = df['Monthly_Balance'].round(2)
        # st.write('Monthly_Balance Data Type: ', df['Monthly_Balance'].dtype)
        # st.write('Monthly_Balance: ', df['Monthly_Balance'].value_counts(ascending=False))
        # st.write('Monthly_Balance NA Values: ', df['Monthly_Balance'].isna().sum())
        # fig, ax = plt.subplots()
        # sns.boxplot(data=df, x=df['Monthly_Balance'])
        # st.pyplot(fig)
        # st.write('Monthly_Balance < 0', df[df['Monthly_Balance'] < 0]['Monthly_Balance'].count())
        df['Monthly_Balance'] = df['Monthly_Balance'].fillna(df['Monthly_Balance'].median()).astype(int)

        # Credit_Score - convert values to numbers
        # st.write('Credit_Score Values: ', df['Credit_Score'].value_counts())
        # st.write('Credit_Score NA Values: ', df['Credit_Score'].isna().sum())
        df['Credit_Score_Code'] = df['Credit_Score']
        df['Credit_Score'], unqiue = pd.factorize(df['Credit_Score'])
        # st.write('Credit Score Code: ', df['Credit_Score_Code'].value_counts())
        # st.write('Credit Score: ', df['Credit_Score'].value_counts())
        df.drop('Credit_Score_Code', axis=1, inplace=True)

        ### INT Data Type

        # Num_Bank_Accounts - drop values over 100 and below 0
        # st.write('Num_Bank_Accounts Values: ', df['Num_Bank_Accounts'].value_counts(ascending=False))
        # st.write('Num_Bank_Accounts NA Values: ', df['Num_Bank_Accounts'].isna().sum())
        df.drop(df[df['Num_Bank_Accounts'] > 100].index, inplace=True)
        df.drop(df[df['Num_Bank_Accounts'] < 0].index, inplace=True)

        # Num_Credit_Card
        # st.write('Num_Credit_Card Values: ', df['Num_Credit_Card'].value_counts(ascending=False))
        # st.write('Num_Credit_Card NA Values: ', df['Num_Credit_Card'].isna().sum())
        # st.write('Num_Credit_Card less than 0: ', (df['Num_Credit_Card'] < 0).sum())
        df.drop(df[df['Num_Credit_Card'] > 100].index, inplace=True)
        # st.write('Num_Credit_Card >100: ', df['Num_Credit_Card'])

        # Delay_from_due_date - drop negative numbers
        # st.write('Delay_from_due_date less than 0: ', (df['Delay_from_due_date'] < 0).sum())
        df.drop(df[df['Delay_from_due_date'] < 0].index, inplace=True)

        ###############################################################################

        st.session_state.df = df.copy()  # save session state
        st.session_state.preprocessed_df = df.copy()
        st.caption("🚀🚀🚀 Loaded Successfully 🚀🚀🚀")
    except Exception as e:
        st.error({e})
        st.stop()

# check data load
if st.session_state.df is None:
    st.info("Please paste a valid GitHub URL and load the data")
    st.stop()

df = st.session_state.df.copy()

# st.write("Session State Debug:", st.session_state)

st.header("1️⃣ Data Preview & Exploration")
# st.write("Here's a preview of the dataset:")
# st.dataframe(df.head())

st.subheader("You will be using the variable 'df' when working with the data set")
st.caption("For example: ")
viewdf = st.text_area("To view your dataset: enter: df.head() ",
                            value="""""", height=68)
st.session_state["viewdf_input"] = viewdf
if st.button("Preview"):
    try:
        # df = st.session_state.df.copy()
        local_env = {"df": df.copy()}
        res = eval(viewdf, {}, local_env)
        if isinstance(res, pd.DataFrame):
            st.dataframe(res)
        else:
            st.write(res)
    except Exception as e:
        st.error({e})

df_sum = st.text_area("To view descriptive statistics of the data, enter: df.describe()",
                            value="""""", height=68)
st.session_state["df_sum_input"] = df_sum
if st.button("Describe"):
    try:
        local_env = {"df": df.copy()}
        res = eval(df_sum, {}, local_env)
        if isinstance(res, pd.DataFrame):
            st.dataframe(res)
        else:
            st.write(res)
    except Exception as e:
        st.error({e})

# code = st.text_area("Code Area: ", value="""""", height=68)
# if st.button("Run Code"):
#     try:
#         local_env = {"df": df.copy()}
#         res = eval(code, {}, local_env)
#         if isinstance(res, pd.DataFrame):
#             st.write(res)
#             st.session_state.res = res
#         else:
#             st.write(res)
#     except Exception as e:
#         st.error({e})

st.subheader("Example code for references: ")
code_section = {
    "Value Counts": {
        "description": "Return the frequency of each unique row in the DataFrame.",
        "code": """
        dataframe.value_counts()
        """},
    "Drop Functionality": {
    "description": "Drop Specified Columns and Values",
    "code": """

    ###  Drop specified columns from the DataFrame ###

    ## @param axis - axis=1 drops columns; axis=0 drops rows  
    
    # You can drop the column via the name
    dataframe.drop(['Month', 'Name', 'SSN'], axis=1)

    # Or you can drop the column by the index
    dataframe = dataframe.drop(dataframe.columns[[0, 4, 8]], axis=1)

    ###  Drop any value in 'Num_of_Loan' that is greater than 50  ###

    ## .index extracts the index (row labels) of those filtered
    ## @param inplace=True - modifies the original DataFrame directly (no new object)

    dataframe.drop(dataframe[dataframe['Num_of_Loan'] > 50].index, inplace=True)

    """},
    "Convert Object to Numeric Type": {
    "description": "This code section converts the 'Age' column, which was originally an object type into a numeric type.",
    "code": """
    ## pd.to_numeric converts the values to a numeric value (e.g., int, float, etc)
    ## @param errors='coerce' - values that can't be converted into a number because NaN (not a number)
    
    dataframe['Age'] = pd.to_numeric(dataframe['Age'], errors='coerce')
    """},
    "Remove Missing Values": {
    "description": "This code uses .dropna() to remove the missing values (NaN) from the 'Age' column.",
    "code": """
    dataframe = dataframe.dropna(subset=['Age'])
    """},
    "Convert to Int Type": {
        "description": "This code section converts the 'Age' column to Int Type.",
        "code": """
        ## .astype(int) converts the values into integers
        
        dataframe['Age'] = dataframe['Age'].astype(int)
        """},
    "Replace Blank Values": {
    "description": "This code section replaces any blank cells with the value 'Other'",
    "code": """
    ## .replace - replaces the values in the column
    ## @param to_replace - specifies which value to find
    ## @param value - specifies what to replace it with
    ## @param inplace=True - modifies the original DataFrame directly (no new object)
    
    dataframe['Occupation'].replace(to_replace='', value='Other', inplace=True)
    """},
    "Factorize Categorical Values": {
    "description": "Converts categorical values into integer codes",
    "code": """
    ##  pd.factorize will assign a numeric value to the categorical label 
    ## (e.g., Student = 0; Doctor = 2; Other = 3; etc)
    
    dataframe['Occupation_Code'] = dataframe['Occupation'] # copy the old column into 'Occupation_Code'
    dataframe['Occupation'], unqiue = pd.factorize(dataframe['Occupation'])
    """},
    "Replace A Specific Values": {
    "description": "This line finds the string '!@9#%8' in the dataset and replaces the value with 'Undetermined'",
    "code": """
    ## .replace - replaces the values in the column
    ## @param to_replace - specifies which value to find
    ## @param value - specifies what to replace it with
    ## @param inplace=True - modifies the original DataFrame directly (no new object)
    
    dataframe['Payment_Behaviour'].replace(to_replace='!@9#%8', value='Undetermined', inplace=True)
    """},
    "Fill Missing Data with Median Value": {
    "description": "This line of code fills any missing data with the median value of the column"
                   "Medians can be used if the data is skewed or has too many outliers",
    "code": """
    ## .fillna - replaces any missing value (NaN) in the column
    ## .median() - calculates the median vlaue of the column
    ## .astype(int) - converts the column values to integer type
    ## .round(2) - ensures only 2 decimal places
    
    dataframe['Monthly_Balance'] = dataframe['Monthly_Balance'].fillna(dataframe['Monthly_Balance'].median()).astype(int)
    dataframe['Monthly_Balance'] = dataframe['Monthly_Balance'].round(2)
    """}
}

# Display each code section with its description inside an expander
for title, content in code_section.items():
    with st.expander(f"Show Code: {title}"):
        st.markdown(content["description"])
        st.code(content["code"], language="python")

st.header("2️⃣ Data Cleaning (Preprocessing)")
st.subheader("\n 2-1. Drop the following columns: 'ID', 'Customer_ID', 'Type_of_Loan' \n")

st.markdown(""" REMINDER: 'df' is the name of your dataframe. """)

st.write("Current Columns: ")
cols = list(df.columns)
st.write(cols)

df_original = st.session_state.preprocessed_df.copy()

# reset to the original loaded data
if st.button("Reset Columns"):
    if st.session_state.preprocessed_df is not None:
        st.session_state.df = st.session_state.preprocessed_df.copy()
        df = st.session_state.df.copy()
        st.success("Data has been reset.")
        st.write("Data: ", df.head(1))

drop_columns = st.text_area("TODO: Drop the columns: 'ID', 'Customer_ID', 'Type_of_Loan' ",
                            value="""""")
st.session_state["drop_columns_input"] = drop_columns

st.caption("\n You may need to reset if you have to run the query again. \n")

if st.button("Drop Columns"):
    try:
        local_env = {"df": df.copy()}
        st.write('Before: ', st.session_state.preprocessed_df.head(1))
        ols = list(df.columns)
        st.text(cols)

        exec(drop_columns, {}, local_env)
        updated_df = local_env.get("df", None) # updated df
        # check
        dropped_index = df.columns[[0,1,8]].tolist()
        dropped = ['ID', 'Customer_ID', 'Type_of_Loan']
        cols = all(col not in df.columns for col in dropped)
        if updated_df is not None and cols or dropped_index:
            st.session_state.df = updated_df.copy()
            st.success("Columns Successfully Dropped.")
            st.write('After: ', updated_df.head(1))
            cols = list(updated_df.columns)
            st.text(cols)
            df = st.session_state.df.copy()
        else:
            st.error("Make sure you are assigning the result to `df` and the columns are dropped")
    except Exception as e:
        st.error({e})
        st.write("Session State Debug:", st.session_state)

st.subheader("In this dataset, there are values in the 'Age' column that makes no sense.\n")
st.caption("See the histogram and boxplot below: \n\n")
options = ["Select Graph", "Histogram", "Boxplot"]
dropdown = st.selectbox("Choose a Visualization: ", options)
# if st.button("Visualize"):
if dropdown == "Histogram":
    fig, ax = plt.subplots()
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    sns.histplot(st.session_state.preprocessed_df['Age'], bins=10, kde=True)
    st.pyplot(fig)
elif dropdown == 'Boxplot':
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='Age', data=st.session_state.preprocessed_df, ax=ax1)
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age")
    st.pyplot(fig1)

# reset to the original loaded data
if st.button("Reset Age"):
    if st.session_state.df is not None:
        st.session_state.df = st.session_state.preprocessed_df.copy()
        df = st.session_state.df.copy()
        st.success("Data has been reset.")
        st.write("Data: ", st.session_state.preprocessed_df['Age'].describe())

st.subheader("\n 2-2. Drop the values that are greater than 100 [>100] and less than 1 [< 1] "
         "\n See your changes below with the diagram of your choice."
         "\n Plot the new values with the diagram of your choice.")
st.caption("\n You may need to reset if you have to run the query again. \n")
drop_age = st.text_area("TODO: Drop the all values greater than 100 and less than 1: ")
st.session_state["drop_age_input"] = drop_age

if st.button("Drop Age"):
    try:
        local_env = {"df": df.copy()}
        st.write("Before: ", st.session_state.preprocessed_df['Age'].describe())

        exec(drop_age, {}, local_env)
        updated_df = local_env.get("df", None) # updated df
        if updated_df is not None:
            st.session_state.df = updated_df.copy()
            st.success("Age Values Dropped.")
            st.write("After: ", updated_df['Age'].describe())
            df = st.session_state.df.copy()
        else:
            st.error("Could not update the DataFrame. \nMake sure your code modifies the variable `df` "
                     "and does not contain any errors.")
    except Exception as e:
        st.error({e})
        st.write("Session State Debug:", st.session_state)

ageDiagram = ["Select a chart", "Histogram", "Boxplot"]

choice = st.selectbox("Choose a Visualization: ", ageDiagram)
# if st.button("Visualize"):
if choice == "Histogram":
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Age'], bins=10, kde=True)
        ax2.set_title("Age Distribution")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
elif choice == "Boxplot":
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='Age', data=df, ax=ax3)
        ax3.set_title("Age Distribution")
        ax3.set_xlabel("Age")
        st.pyplot(fig3)

st.header("Data visualizations")

visualizations = ["Select a Visualization", "Heatmap",
           "Countplot: Occupation vs Credit Score",
           "Countplot: Payment Behaviour vs Credit Score"]

select = st.selectbox("Choose a Visualization: ", visualizations)

if select == "Heatmap":
    heatmap = plt.figure(figsize=(18,10))
    sns.heatmap(df.select_dtypes(include='number').corr(),annot=True,linewidths=True)
    st.pyplot(heatmap)
elif select == "Countplot: Occupation vs Credit Score":
    countplot, ax4 = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Occupation', hue='Credit_Score', data=df)
    plt.xlabel('Occupation')
    plt.xticks(rotation=50)
    plt.title('Occupation vs Credit Score')
    plt.legend(title='Occupation vs Credit Score')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(countplot)
elif select == "Countplot: Payment Behaviour vs Credit Score":
    countplot2, ax5 = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Payment_Behaviour', hue='Credit_Score', data=df)
    plt.xlabel('Payment_Behaviour')
    plt.xticks(rotation=20)
    plt.title('Payment Behaviour vs Credit Score')
    plt.legend(title='Payment Behaviour vs Credit Score')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(countplot2)

# reset to the original loaded data
if st.button("Reset"):
    if st.session_state.df is not None:
        st.session_state.df = st.session_state.preprocessed_df.copy()
        df = st.session_state.df.copy()
        st.success("Data has been reset.")

st.header("3️⃣ Data Modeling")
st.caption("Training a Machine Learning Model")

st.subheader("\n 3-1. Split the data into Train and Test sets. "
             "\n\n Using the 70/30 split -- 70% (train) /30% (test)")

st.markdown("X - Features  \n y - Target  \n test_size - percent that goes to the validation set  \n random_state - set a seed for reproducibilty  \n\n"
            "X_train: training features (columns used to train the model)  "
            "\nX_test: testing features (same columns, but for evaluating the model)  "
            "\ny_train & y_test: the corresponding target (label) values for each  ")
training_data = st.text_area("Hint: X_train, X_test, y_train, y_test = train_test_split(?, ?, test_size=??, random_state=??)",
                             value="""from sklearn.model_selection import train_test_split
X = df[df.columns[:-1]].values # all values except the last column
y = df[df.columns[-1]].values
\n## TODO: Enter your code below""", height=200)
st.session_state["training_data_input"] = training_data

if st.button("Split Data"):
    try:
        local_env = {"df": df.copy()}
        exec(training_data, {}, local_env)
        updated_df = local_env.get("df", None) # updated df

        X = local_env.get("X", None)
        y = local_env.get("y", None)
        X_train = local_env.get("X_train", None)
        y_train = local_env.get("y_train", None)
        X_test = local_env.get("X_test", None)
        y_test = local_env.get("y_test", None)

        if updated_df is not None:
            st.session_state.df = updated_df.copy()
            st.success("Training & Test Data have been split successfully")

            st.session_state.X = X
            st.session_state.y = y
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            # st.write("data: ", df.head(1))
            # X = df[df.columns[:-1]].values  # all values except the last column
            # st.write("X Columns: ", X)
            # y = df[df.columns[-1]].values
            # st.write("y Column: ", y)

            if X_train is not None and X_test is not None:
                st.write("X_train shape (rows, columns):", X_train.shape)
                st.write("X_test shape (rows, columns):", X_test.shape)
            df = st.session_state.df.copy()
        else:
            st.error("X should be all your features; y should be your predicting value--Credit Score")
    except Exception as e:
        st.error({e})
        st.write("Session State Debug:", st.session_state)

st.subheader("Model References")

model_section = {
    "Logistic Regression": {
        "description": "[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html): "
                       "Study or prediction of qualitative variables; the probability of an event occurring given the independent variables  "
                       "\n » Best for Linear, Binary Classification Problems  "
                       "\n » Downside: May struggle with non-linear relationships  "
                       "\n\n **Other Parameters:**  "
                       "\n>*solver*: algorithm to optimize the logistic regression function.  "
                       "\n*max_iter*: maximum iterations to converge  "
                       "\n*multi_class*: strategy for multiclass classification  ",
        "code": """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000) # multinomial for non-binary logistic regression
model.fit()
"""},
    "Decision Tree Classifier": {
        "description": "[Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html): Segmenting or splitting the predictor space into a number of regions "
                       "\n » Best for Non-Linear, interpretable decision-making  "
                       "\n » Downside: Can over fit if not pruned or regularized; thus, lower accuracy"
                       "\n\n **Other Parameters:**  "
                       "\n>*random_state*: for reproducibility; ensures the same training result is obtained every time you run the code  "
                       "\n*max_leaf_nodes*: pre-pruning to help with overfitting  "
                       "\n*max_depth*: limit the depth of each tree (helps prevent overfitting)  "
                       "\n*min_samples_split*: minimum number of samples required to split a node"
                       "\n*min_samples_leaf*: minimum number of samples required at a leaf node  "
                       "\n*max_features*: number of features to consider when looking for the best split ('auto', 'sqrt', 'log2')  "
                        "\n*splitter*: choose the split at each node (options: 'best' (default); 'random')  "
        ,
        "code": """
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42, max_leaf_nodes=6)
        # train
        model.fit()
        # test
        model.predict()
        """},
    "Random Forest Classifier": {
        "description": "[Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier): Ensemble of decision trees  "
                       "\n » Reduces overfitting by using many decision trees  "
                       "\n » Downside: Slower than logistic regression or a single decision tree  \n"
                        "\n\n **Other Parameters:**  "
                        "\n>*random_state*: for reproducibility; ensures the same training result is obtained every time you run the code  "
                        "\n*n_estimators*: set the number of decision trees in the forest  "
                        "\n*max_depth*: limit the depth of each tree (helps prevent overfitting)  "
                        "\n*min_samples_split*: minimum number of samples required to split a node  "
                        "\n*min_samples_leaf*: minimum number of samples required at a leaf node  "
                        "\n*max_features*: number of features to consider when looking for the best split ('auto', 'sqrt', 'log2')  "
                        "\n>*bootstrap*: whether bootstrap samples are used (default: True)  "
                        "\n"
                       "\n`NOTE: More trees = better performance (up to a point)`  "
                       "\n`NOTE: Too many trees = longer training/prediction time`  "
        ,
        "code": """
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # train
        model.fit()
        # test
        model.predict()
        """},
    "K-Nearest Neighbors (KNN)": {
        "description": "[KNN](https://scikit-learn.org/stable/modules/neighbors.html): Classifier that uses proximity to classify or predict a grouping of an individual data point"
                       "\n » No training step (lazy learner)"
                       "\n » Downside: Gets slow on larger datasets, sensitive to feature scaling"
                       "\n\n **Other Parameters:**  "
                        "\n>*n_neighbors*: number of nearest neighbors to use (k in KNN)  "
                       "\n `NOTE: Smaller values = more flexible but can overfit; larger values = smoother decision boundary`  "
                        "\n*weights*: 'uniform', 'distance', 'callable'  "
                        "\n*algorithm*: 'auto', 'ball_tree', 'kd_tree', 'brute'  "
                        "\n*leaf_size*: affects the speed of the tree-based algorithm  "
                        "\n `NOTE: Lower values = faster queries but higher memory`  "
                        "\n*metric*: 'euclidean', 'manhatten', 'chebyshev'  ",
        "code": """
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
        # train
        model.fit()
        # test
        model.predict()
        """}
}

for title, content in model_section.items():
    with st.expander(f"Show Model: {title}"):
        st.markdown(content["description"])
        st.code(content["code"], language="python")

st.subheader("4️⃣ Train/Test a Model")
st.caption("WARNING: DO NOT USE YOUR TEST DATA WHEN TRAINING YOUR MODEL")
st.caption("Keep in mind: Predictor Values: 1 = 'Standard'; 2 ='Poor'; 3='Good'")
st.subheader("Logistic Regression")
logr = st.code("from sklearn.linear_model import LogisticRegression"
        "\nfrom sklearn.metrics import accuracy_score, confusion_matrix, classification_report"
        "\nlogr = LogisticRegression(solver='lbfgs',max_iter=1000)"
        "\n\n# Train the Logistic Regression Model"
        "\nlogr.fit(X_train, y_train)"
        "\n\n# Evaluate on training data"
        "\nytrain_pred = logr.predict(X_train)"
        "\naccuracy = accuracy_score(y_train, ytrain_pred)")

if st.button('View Logistic Regression Model'):
    try:
        df = st.session_state.get("df")
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")
        # Check if data is present
        if any(x is None for x in (X_train, X_test, y_train, y_test)):
            st.warning("Missing training or test data.")
        else:
            logr = LogisticRegression(solver='lbfgs', max_iter=1000)
            logr.fit(X_train, y_train)
            ytrain_pred = logr.predict(X_train)
            logr_accuracy = accuracy_score(y_train, ytrain_pred)
            st.write("Accuracy: ", logr_accuracy)

            st.session_state['logr'] = logr  # Save model in session state
            st.session_state['logr_model_accuracy'] = logr_accuracy
    except Exception as e:
            st.error({e})

logr_score = st.code("\ny_pred = logr.predict(X_test)"
                     "\naccuracy = accuracy_score(y_test, y_pred)"
                     "")

if st.button('Score Logistic Regression Model'):
    try:
        df = st.session_state.get("df")
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        # if hasattr(logr, 'coef_'):  # Checks if the model has been fitted
        #     st.write("Model is fitted.")
        # else:
        #     st.write("Model is not fitted.")

        st.write("Train Accuracy: ", st.session_state['logr_model_accuracy'])

        logr = st.session_state['logr']
        y_pred = logr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Test Accuracy: ", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)
        rep_df = pd.DataFrame(report).transpose()
        st.text("Classification Report:")
        st.caption("Predictor Values:  \n1 = 'Standard'  \n2 = 'Poor'  \n3 = 'Good'  "
                   "\nPrecision - how many predicted positives were corrects?  "
                   "\nRecall - how many actual positives were there?  "
                   "\nF1-score - mean of precision and recall.  "
                   "\nSupport - number of actual occurrences in each class in the data.  ")
        st.dataframe(rep_df.style.format("{:.2f}"))
    except Exception as e:
        st.error({e})

st.subheader("TODO: Train a model or two of your choice!")

st.caption("Example Skeleton: \n"
           "\n model = model(parameters) \n"
           "\n model.fit(?, ?)  \n"
           "\n ytrain_pred = model.predict(?)  \n"
           "\n accuracy = accuracy_score(?, ?)  \n")

st.write("\n")
st.subheader("Decision Tree Classifier")
dtc = st.text_area("TODO: Enter your code below for the: Decision Tree Classifier\n"
                "\n\t Parameters:"
                "\n\t random_state: for reproducibility; ensures the same training result is obtained every time you run the code  "
                "\n\t max_leaf_nodes: pre-pruning to help with overfitting  "
                "\n\t max_depth: limit the depth of each tree (helps prevent overfitting)  "
                "\n\t min_samples_split: minimum number of samples required to split a node "
                "\n\t min_samples_leaf: minimum number of samples required at a leaf node  "
                "\n\t max_features: number of features to consider when looking for the best split ('auto', 'sqrt', 'log2')  "
                "\n\t splitter: choose the split at each node (options: 'best' (default); 'random')"
                   ,value="""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
dtc = 
""", height=150)

st.session_state["dtc_input"] = dtc

if st.button("Submit Decision Tree Model"):
    try:
        df = st.session_state.get("df")
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        exec(dtc, {}, local_env)

        model = local_env.get("dtc", None)  # Extract trained model

        if model is None:
            st.error("Model `dtc` not found. Make sure your code defines it.")
        else:
            # Evaluate
            ytrain_pred = model.predict(X_train)
            dtc_accuracy = accuracy_score(y_train, ytrain_pred)
            acc = accuracy_score(y_train, ytrain_pred)

            st.write("Train Accuracy: ", acc)

            # Visualize Tree
            fig2, ax2 = plt.subplots(figsize=(30, 20))
            plot_tree(model, filled=True, feature_names=df.columns[:-1], class_names=True, rounded=True, fontsize=10)
            st.pyplot(fig2)

            # Save model to session
            st.session_state['dtc'] = model
            st.session_state['dtc_model_accuracy'] = dtc_accuracy
    except Exception as e:
            st.error({e})

st.write("\n")
dtc_score = st.text_area("Decision Tree Classifier", value="""""", height=68)
st.session_state["dtc_score_input"] = dtc_score

if st.button("Score Decision Tree Model"):
    try:
        model = st.session_state.get("dtc", None)
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        if model is None:
            st.warning("No trained Decision Tree model found. Submit your model first.")
        else:
            st.write("Train Accuracy: ", st.session_state['dtc_model_accuracy'])
            dtc = st.session_state['dtc']
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Test Accuracy: {accuracy:.2f}")

            # Visualize Tree
            fig2, ax2 = plt.subplots(figsize=(20, 10))
            plot_tree(model, filled=True, feature_names=df.columns[:-1], class_names=True, rounded=True, fontsize=10)
            st.pyplot(fig2)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Test Accuracy: ", accuracy)

            st.write("(Optional) Classification Report for Decision Tree")
            report = classification_report(y_test, y_pred, output_dict=True)
            rep_df = pd.DataFrame(report).transpose()
            st.text("Classification Report:")
            st.caption("Predictor Values:  \n1 = 'Standard'  \n2 = 'Poor'  \n3 = 'Good'  "
                       "\nPrecision - how many predicted positives were corrects?  "
                       "\nRecall - how many actual positives were there?  "
                       "\nF1-score - mean of precision and recall.  "
                       "\nSupport - number of actual occurrences in each class in the data.  ")
            st.dataframe(rep_df.style.format("{:.2f}"))
    except Exception as e:
        st.error({e})

st.write("\n")
st.subheader("Random Forest Classifier")
rfc = st.text_area("TODO: Enter your code below for the: Random Forest Classifier\n"
                "\n\t Parameters:"
                "\n\t random_state: for reproducibility; ensures the same training result is obtained every time you run the code "
                "\n\t n_estimators: set the number of decision trees in the forest  "
                "\n\t max_depth: limit the depth of each tree (helps prevent overfitting)  "
                "\n\t min_samples_split: minimum number of samples required to split a node  "
                "\n\t min_samples_leaf: minimum number of samples required at a leaf node  "
                "\n\t max_features: number of features to consider when looking for the best split ('auto', 'sqrt', 'log2')  "
                "\n\t bootstrap: whether bootstrap samples are used (default: True)"
                   ,value="""from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
rfc =
""", height=180)
st.session_state["rfc_input"] = rfc

if st.button("Submit Random Forest Model"):
    try:
        df = st.session_state.get("df")
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        exec(rfc, {}, local_env)

        model = local_env.get("rfc", None)  # Extract trained model

        if model is None:
            st.error("Model `rfc` not found. Make sure your code defines it.")
        else:
            # Evaluate
            ytrain_pred = model.predict(X_train)
            acc = accuracy_score(y_train, ytrain_pred)
            rfc_accuracy = accuracy_score(y_train, ytrain_pred)
            st.write("Train Accuracy: ", acc)

            # # Visualize
            # fig, ax = plt.subplots(figsize=(20, 10))
            # plot_tree(model.estimators_[0], feature_names=df.columns[:-1],
            #           class_names=model.classes_.astype(str), filled=True)
            # st.pyplot(fig)

            sig = model.feature_importances_
            sig_df = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': sig})
            sig_df = sig_df.sort_values(by='Importance', ascending=False)

            # Plot
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=sig_df, ax=ax2)
            plt.title('Feature Importance')
            st.pyplot(fig2)

            # Save model to session
            st.session_state['rfc'] = model
            st.session_state['rfc_model_accuracy'] = rfc_accuracy
    except Exception as e:
            st.error({e})

rfc_score = st.text_area("Score Random Forest Classifier", value="""""", height=68)
st.session_state["rfc_score_input"] = rfc_score

if st.button("Score Random Forest Model"):
    try:
        model = st.session_state.get("rfc", None)
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        if model is None:
            st.warning("No trained Random Forest model found. Submit your model first.")
        else:
            st.write("Train Accuracy: ", st.session_state['rfc_model_accuracy'])
            rfc = st.session_state['rfc']
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Test Accuracy: {accuracy:.2f}")

            sig = model.feature_importances_
            sig_df = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': sig})
            sig_df = sig_df.sort_values(by='Importance', ascending=False)

            # Plot
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=sig_df, ax=ax2)
            plt.title('Feature Importance')
            st.pyplot(fig2)

            st.write("(Optional) Classification Report for Random Forest")
            report = classification_report(y_test, y_pred, output_dict=True)
            rep_df = pd.DataFrame(report).transpose()
            st.text("Classification Report:")
            st.caption("Predictor Values:  \n1 = 'Standard'  \n2 = 'Poor'  \n3 = 'Good'  "
                       "\nPrecision - how many predicted positives were corrects?  "
                       "\nRecall - how many actual positives were there?  "
                       "\nF1-score - mean of precision and recall.  "
                       "\nSupport - number of actual occurrences in each class in the data.  ")
            st.dataframe(rep_df.style.format("{:.2f}"))

    except Exception as e:
        st.error({e})

st.write("\n")
st.subheader("K-Nearest Neighbor (KNN)")
knn = st.text_area("TODO: Enter your code below for the: K-Nearest Neighbor (KNN) Model\n"
                "\n\t Parameters:"
                "\n\t n_neighbors: number of nearest neighbors to use (k in KNN) "
                "\n\t weights: 'uniform', 'distance', 'callable'  "
                "\n\t algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'  "
                "\n\t leaf_size: affects the speed of the tree-based algorithm; default 30  "
                "\n\t metric: 'euclidean', 'manhatten', 'chebyshev'  "
                   , value="""from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
knn = 
""", height=180)
st.session_state["knn_input"] = knn

if st.button("Submit KNN Model"):
    try:
        df = st.session_state.get("df")
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        exec(knn, {}, local_env)

        model = local_env.get("knn", None)  # Extract trained model

        if model is None:
            st.error("Model `knn` not found. Make sure your code defines it.")
        else:
            # Evaluate
            ytrain_pred = model.predict(X_train)
            acc = accuracy_score(y_train, ytrain_pred)
            knn_accuracy = accuracy_score(y_train, ytrain_pred)

            st.write("Train Accuracy: ", acc)

            # Save model to session
            st.session_state['knn'] = model
            st.session_state['knn_model_accuracy'] = knn_accuracy
    except Exception as e:
            st.error({e})

knn_score = st.text_area("Score K-Nearest Neighbor (KNN)", value="""""", height=68)
st.session_state["knn_score_input"] = knn_score

if st.button("Score KNN Model"):
    try:
        model = st.session_state.get("knn", None)
        X_train = st.session_state.get("X_train")
        X_test = st.session_state.get("X_test")
        y_train = st.session_state.get("y_train")
        y_test = st.session_state.get("y_test")

        local_env = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        if model is None:
            st.warning("No trained KNN model found. Submit your model first.")
        else:
            st.write("Train Accuracy: ", st.session_state['knn_model_accuracy'])
            knn = st.session_state['knn']
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Test Accuracy: {accuracy:.2f}")

            st.write("(Optional) Classification Report for KNN")
            report = classification_report(y_test, y_pred, output_dict=True)
            rep_df = pd.DataFrame(report).transpose()
            st.text("Classification Report:")
            st.caption("Predictor Values:  \n1 = 'Standard'  \n2 = 'Poor'  \n3 = 'Good'  "
                       "\nPrecision - how many predicted positives were corrects?  "
                       "\nRecall - how many actual positives were there?  "
                       "\nF1-score - mean of precision and recall.  "
                       "\nSupport - number of actual occurrences in each class in the data.  ")
            st.dataframe(rep_df.style.format("{:.2f}"))

    except Exception as e:
        st.error({e})

st.caption(" (╯°□°）╯︵ ┻━┻ ")

st.subheader("Model Accuracy Comparison")

# Collect all accuracies
model_accuracies = {
    "Logistic Regression": st.session_state.get("logr_model_accuracy"),
    "Decision Tree": st.session_state.get("dtc_model_accuracy"),
    "Random Forest": st.session_state.get("rfc_model_accuracy"),
    "KNN": st.session_state.get("knn_model_accuracy")
}

# Remove any models that weren't trained (i.e., have None as accuracy)
filtered = {k: v for k, v in model_accuracies.items() if v is not None}

if filtered:
    # Optional textual summary
    st.markdown("Trained Models & Accuracies:")
    for model, acc in filtered.items():
        st.write(f"- **{model}**: {acc:.2f}")

    acc_df = pd.DataFrame(list(filtered.items()), columns=["Model", "Accuracy"])

    # Plot bar chart
    fig, ax = plt.subplots()
    sns.barplot(x="Accuracy", y="Model", data=acc_df, palette="viridis", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy Comparison Across Models")

    # Annotate bars with accuracy values
    for i, (model, accuracy) in enumerate(filtered.items()):
        ax.text(accuracy + 0.01, i, f"{accuracy:.2f}", va='center')

    st.pyplot(fig)

    # Table
    st.write("Accuracy Table")
    st.dataframe(acc_df.style.format({"Model Accuracy": "{:.2f}"}))

else:
    st.info("Train at least one model to compare accuracies.")

from fpdf import FPDF
import datetime


def generate_pdf():
    pdf = FPDF(format='A4', orientation='P')
    pdf.add_page()
    pdf.set_font("courier", size=12)

    pdf.cell(0, 10, txt="Model Results Report", ln=True, align='C')
    pdf.ln(10)

    # include accuracies if available
    models = ["logr_model_accuracy", "dtc_model_accuracy", "rfc_model_accuracy", "knn_model_accuracy"]

    for model_key in models:
        accuracy = st.session_state.get(model_key)
        if accuracy is not None:
            model_name = model_key.replace("_model_accuracy", "").upper()
            pdf.cell(0, 10, txt=f"{model_name} Accuracy: {accuracy}", ln=True, align='C')

    # user input
    text_inputs = {
        "LINK": st.session_state.get("link", ""),
        "VIEW DATASET": st.session_state.get("viewdf", ""),
        "DESCRIBE DATASET": st.session_state.get("df_sum", ""),
        "DROP COLUMN": st.session_state.get("drop_columns", ""),
        "DROP VALUES": st.session_state.get("drop_age", ""),
        "SPLIT DATA": st.session_state.get("training_data", ""),
        # "LOGISTIC REGRESSION TRAIN": st.session_state.get("logr", ""),
        "DECISION TREE TRAIN": st.session_state.get("dtc", ""),
        "RANDOM FOREST TRAIN": st.session_state.get("rfc", ""),
        "KNN TRAIN": st.session_state.get("knn", ""),
        # "LOGISTIC REGRESSION TEST": st.session_state.get("logr_score", ""),
        "DECISION TREE TEST": st.session_state.get("dtc_score", ""),
        "RANDOM FOREST TEST": st.session_state.get("rfc_score", ""),
        "KNN TEST": st.session_state.get("knn_score", ""),
    }

    for title, code in text_inputs.items():
        if isinstance(code, str) and code.strip():
            pdf.set_font("courier", style='B', size=12)
            pdf.multi_cell(0, 6, txt=f"{title}:\n{code}", align='L')
            pdf.set_font("courier", size=12)
            pdf.multi_cell(0, 6, txt=code, align='L')
            pdf.ln(3)

    # add session values
    for key in st.session_state:
        if isinstance(st.session_state[key], (int, float, str)):
            pdf.cell(0, 10, txt=f"{key}: {st.session_state[key]}", ln=True)

    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

if st.button("Generate PDF"):
    file_path = generate_pdf()
    with open(file_path, "rb") as file:
        st.download_button("Download Report", data=file, file_name="model_report.pdf", mime="application/pdf")
