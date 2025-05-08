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

st.subheader("Overview of Unsupervised Learning")
st.markdown("""
    Unsupervised Machine Learning uses unlabeled data (data with only features) in order to identify structures, discover patterns, group similar observations, and/or reduce dimensionality in the dataset.
    It is best for exploration, rather than prediction. Supervised Machine Learning is better suited for tasks where the goal is to make predictions for a target variable.
    Types of Unsupervised ML:
    * **Dimensionality Reduction:** Simplifies high-dimensional data (data with lots of features) into fewer features. Good for visualization or for removing largely unneccesary features.
    * **Clustering:** Grouping data points into clusters of similar observations""")

# -----------------------
# AVAILABLE ML ALGORITHMS
# -----------------------

st.subheader("Available ML Algorithms")
st.markdown("""\n This application makes three different algorithms available:
    \n **Dimensionality Reduction**
    1. Principal Component Analysis (PCA)
    \n **Clustering**
    \n 2. KMeans Clustering
    \n 3. Hierarchical Clustering
    \n You can learn more about each algorithm below.
""")

# Tab Layout Configuration
tab_titles = ['Principal Component Analysis (PCA)', 'KMeans Clustering', 'Hierarchical Clustering']
tabs = st.tabs(tab_titles)

# Tab 0: Principal Component Analysis (PCA)
with tabs[0]:
    st.subheader("Principal Component Analysis (PCA)")

# Tab 1: KMeans Clustering
with tabs[1]:
    st.markdown("<h2 style='text-align:center; 'Principal Component Analysis (PCA)</h2>", unsafe_allow_html=True)
    st.markdown("""
    """)

# Tab 2: Hierarchical Clustering
with tabs[2]:
    st.subheader("Hierarchical Clustering")

# -------------------
# DATASET INFORMATION
# -------------------

# -----------
# SOURCE CODE
# -----------

# Creator

# REFERENCES
"""
tab_variable_name = st.tabs(['tab names'])

with tab_variable_name[0]:
with tab_variable_name[1]:
"""
