import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression


# --------------------------------
# Application Configuration & Info
# --------------------------------

# Setting to widescreen
st.set_page_config(layout="wide")

# Displaying application summary information to user
st.title("Machine Learning Application")
st.markdown("""
### :spiral_note_pad: About the Application
This application allows you to apply various machine learning models to an uploaded dataset. 
With abilities to choose from various machine learning models, experiment with hyperparameters, 
and compare the performance between models, this application allows for easy data exploration
whether you use a sample dataset or upload your own. The application's comparative layout also
helps you quickly find models that fit your needs based on the performance metrics you care about.
            
:exclamation: NOTE: This application works best on a fullscreen desktop view.
""")

# Configuring app interface with a column layout
col1, col2, col3 = st.columns([1, 1, 3])

# -----------------
# Dataset Selection
# -----------------

with col1:
    st.subheader(":one: Select and View Your Dataset")

    dataset = st.selectbox('Dataset selection', ['Titanic', 'California Housing', 'Upload Your Own'])

    if dataset == 'Titanic':
        df = sns.load_dataset('titanic')
    elif dataset == 'California Housing':
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
    elif dataset == 'Upload Your Own':
        file = st.file_uploader('Upload your .csv file here:')
        if file is not None:
            df = pd.read_csv(file)
        else:
            st.warning("Please upload a CSV file.")
            st.stop()

# ------------------------
# Select Feature Variables
# ------------------------

with col1:
    st.subheader(":two: Select Feature Variables")

    feature_vars = st.multiselect('Feature Variables', df.columns, default=df.columns)

# ------------------
# Data Preprocessing
# ------------------

if 'df_clean' not in st.session_state:
    st.session_state.df_clean = df.copy()

with col2:
    st.subheader(":three: Preprocess Data")
    
    # Selecting column to clean
    column = st.selectbox("Choose a column to fill", df.select_dtypes(include=['number', 'bool', 'object']).columns)
    method = st.radio("Choose a method", [
        "Original DF", 
        "Drop Rows", 
        "Drop Columns (>50% Missing)",
        "Impute Mean", 
        "Impute Median", 
        "Impute Zero",
        "Convert to Numeric",
        "Get Dummies"
    ])

# Copy of df in order to preserve the original data
df_clean = st.session_state.df_clean

# Apply modifications
if method == "Original DF":
    st.session_state.df_clean = st.session_state.df_clean = df.copy()
elif method == "Drop Rows":
    # Remove all rows that contain any missing values.
    st.session_state.df_clean = st.session_state.df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    # Drop columns where more than 50% of the values are missing.
    st.session_state.df_clean = st.session_state.df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean() > 0.5])
elif method == "Impute Mean":
    # Replace missing values in the selected column with the column's mean.
    st.session_state.df_clean[column] = st.session_state.df_clean[column].fillna(df[column].mean())
elif method == "Impute Median":
    # Replace missing values in the selected column with the column's median.
    st.session_state.df_clean[column] = st.session_state.df_clean[column].fillna(df[column].median())
elif method == "Impute Zero":
    # Replace missing values in the selected column with zero.
    st.session_state.df_clean[column] = st.session_state.df_clean[column].fillna(0)
elif method == "Convert to Numeric":
    bool_true = True
    bool_false = False
    st.session_state.df_clean[column] = st.session_state.df_clean[column].map({bool_true:1, bool_false:0})
elif method == "Get Dummies":
    st.session_state.df_clean = pd.get_dummies(st.session_state.df_clean, columns=[column], drop_first=True)
    
# ---------------
# Model Selection
# ---------------

with col2:
    st.subheader(":four: Select Your ML Models and Target")
    ml_1 = st.selectbox('Left Model', ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Dimensionality Reduction'])
    ml_2 = st.selectbox('Right Model', ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors', 'Dimensionality Reduction'])
    target_var = st.selectbox('Target Variable', st.session_state.df_clean.columns)
    target_var_df = target_var
    

# -----------------
# Dataframe Viewing
# -----------------

# Allows users to view their chosen dataframe before 
# applying machine learning models, which may help them 
# decide whether pre-processing is needed and select a 
# model based on its applicability to the dataset

with col3:
    st.subheader("Selected Dataframe View:")
    st.markdown(""":exclamation: Note: Dropping feature variables in step 2 will affect both the original dataset 
                and the modified dataset, but any modifications made in step 3 will not affect the
                original dataset.""")
    df_view = st.selectbox('Choose view', ['Original Dataframe', 'Cleaned Dataframe'])
    if df_view == 'Original Dataframe':
        st.dataframe(df)
    else:
        st.dataframe(st.session_state.df_clean)



# ------------------
# ML Modeling - Left
# ------------------

# Splitting the selected target variable from the feature variables
if target_var in feature_vars:
    feature_vars.remove(target_var)

if ml_1 == 'Logistic Regression':
    X = st.session_state.df_clean[feature_vars]
    y = df[target_var]
    # Split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = .2,
                                                        random_state = 42)
    # Initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, cmap = "Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show() # graph shows a slight bias toward predicting non-survival
    # Display classification report
    st.write(classification_report(y_test, y_pred))
    # Extract coefficients and intercept
    coef = pd.Series(model.coef_[0], index = feature_vars)
    intercept = model.intercept_[0]
    # Display coefficients
    st.write(coef)
    st.write(intercept)
    # a coefficient of 0 is a 50/50 chance

# -----------------
# ML Model Displays
# -----------------

col1_2, col2_2 = st.columns(2)

with col1_2:
    st.subheader(ml_1)

with col2_2:
    st.subheader(ml_2)
