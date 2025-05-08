import streamlit as st

# ---------------------------------
# PAGE CONFIGURATION & INFO DISPLAY
# ---------------------------------

# Wide Page Layout
st.set_page_config(layout="wide")

# Page Header
st.title("About")
st.markdown("Thank you for using Unsoupervised, an unsupervised machine learning application. ")

# ---------------------------------
# OVERVIEW OF UPSUPERVISED LEARNING
# ---------------------------------

st.write("   ")
st.subheader("Overview of Unsupervised Learning")
st.markdown("""
    Unsupervised Machine Learning uses unlabeled data (data with only features) in order to identify structures, discover patterns, group similar observations, and/or reduce dimensionality in the dataset.
    It is best for exploration, rather than prediction. Supervised Machine Learning is better suited for tasks where the goal is to make predictions for a target variable.
    Types of Unsupervised ML:
    * **Dimensionality Reduction:** Simplifies high-dimensional data (data with lots of features) into fewer features. Good for visualization or for removing largely unneccesary features.
    * **Clustering:** Grouping data points into clusters of similar observations""")

# ------------
# DATA SOURCES
# ------------

st.write("   ")
st.subheader("Data")
st.markdown("""
This project uses the following datasets from Kaggle:
* Country-data.csv [learn more [here](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?resource=download)]
* Wholesale Customers Data [learn more [here](https://www.kaggle.com/code/farhanmd29/unsupervised-learning/input)]

This project also uses the following sample dataset from sklearn.datasets:
* Breast Cancer Wisconsin [learn more [here]]

Learn more about the dataset [here](https://scikit-learn.org/stable/api/sklearn.datasets.html) on the Scikit Learn API Reference.

Information about the datasets is also available in-app. To view information about each demo dataset, select the dataset in the left sidebar, then under "Dataset Preview" hit the 'ℹ️ Info' tab.
""")

# -------
# LICENSE
# -------

st.write("   ")
st.subheader("License")
st.markdown("""This project is part of a portfolio released under the MIT License. See the portfolio license file **[here](https://github.com/cozette-brown/BROWN-Data-Science-Portfolio/blob/d7c128186047d453de9f2491894e4fd0fa3da77d/LICENSE.md)** for details.""")

# ----------------
# ACKNOWLEDGEMENTS
# ----------------

st.write("   ")
st.subheader("Acknowledgements")
st.markdown("""
This project was created with help from the following sources:
* Lectures and .ipynb files from Professor David Smiley, University of Notre Dame
* Saini, A. (2021). ["How to make a great Streamlit app: Part II"](https://blog.streamlit.io/designing-streamlit-apps-for-the-user-part-ii/)

Plus assistance from these other resources I recommend:
* [Geeks for Geeks](https://geeksforgeeks.org)
* [W3 Schools](https://www.w3schools.com)
* [Streamlit API cheat sheet](https://docs.streamlit.io/develop/quick-reference/cheat-sheet)
* [Streamlit cheat sheet (Streamlit App)](https://cheat-sheet.streamlit.app/)
* [Streamlit Emoji Shortcodes](https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/)

Plus a plethora of resources via documentation from:
* Streamlit
* scikit-learn

Special thanks to @baselhusam on GitHub, and his app ClickML available on [GitHub](https://github.com/baselhusam/ClickML/tree/main) and [Streamlit Community Cloud](https://clickml.streamlit.app/?ref=streamlit-io-gallery-other). The design and code of his application informed the making of this machine learning app.
""")

# ------
# CREDIT
# ------

st.write("   ")
with st.container(border=True):
    st.subheader("Credit")
    st.markdown("""Unsoupervised was created by Cozette Brown, a data science student at the University of Notre Dame. View her data science portfolio and learn more at [github.com/cozette-brown](https://github.com/cozette-brown)""")