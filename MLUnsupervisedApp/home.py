import streamlit as st

"""
get a piece of paper and keep working [delete when done]
"""


# ---------------------------------
# PAGE CONFIGURATION & INFO DISPLAY
# ---------------------------------

st.set_page_config(layout="wide")

st.title("Unsupervised ML Application")

st.markdown("""
This application allows you to apply unsupervised machine learning models to either an uploaded or selected sample dataset.
Once you've uploaded and previewed your dataset, you can navigate to either the K-means Clustering, Hierarchical Clustering,
or Principal Component Analysis (PCA) model and begin exploratory data analysis. This app is fully equipped with the ability to
adjust hyperparameters and observe their effects on essential model performance metrics. """)

# ---------------------------------------------------------
# ---------------------------------------------------------
# v FIX EVERYTHING BELOW THIS LINE [REMOVE WHEN COMPLETE] v
# ---------------------------------------------------------
# ---------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


col1, col2 = st.columns([1,3])

# -----------------
# DATASET SELECTION
# -----------------

# Step 1: Upload or Select a Dataset

with col1:
    st.subheader(":one: Upload or Select a Dataset")
    dataset = st.selectbox('Dataset selection', ['Diabetes', 'Breast Cancer', 'Iris', 'Wine', 'Upload Your Own'])

    if dataset == 'Diabetes': 
        appropriate_model_type = 'regression'
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Breast Cancer':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Iris':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Wine':
        appropriate_model_type = 'classification'
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset == 'Upload Your Own':
            appropriate_model_type = 'none'
            uploaded_file = st.file_uploader("Upload a valid .csv file.", type=["csv"])
            df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a dataset or use the sample dataset to proceed.")
        st.stop()

# ---------------
# DATASET PREVIEW
# ---------------

# Show dataset
with col2:
    st.subheader(f"Dataset Preview: {dataset}")
    if dataset == 'Upload Your Own':
        st.info('Developer\'s Note: You must prepare the dataset for use in appropriate machine learning models prior to uploading. Otherwise, you may encounter errors when using the application\'s machine learning algorithms.')
    st.dataframe(df)

# ------------------
# DATA PREPROCESSING
# ------------------

with col1:
    st.subheader(":two: Select and Adjust Model")
    # Select target and features
    columns = df.columns.tolist()
    target_col = st.selectbox("Select the target column", columns)
    features = [col for col in columns if col != target_col]

X = df[features]
y = df[target_col]
