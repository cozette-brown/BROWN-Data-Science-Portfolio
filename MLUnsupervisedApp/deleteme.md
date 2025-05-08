# Kaggle Datasets

**Country-data**
https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?resource=download

**Wholesale Customers Data**
https://www.kaggle.com/code/farhanmd29/unsupervised-learning/input

# Discarded Bits of Code

"""
with col1:
    st.subheader(":two: Select and Adjust Model")
    # Select target and features
    columns = df.columns.tolist()
    target_col = st.selectbox("Select the target column", columns)
    features = [col for col in columns if col != target_col]

X = df[features]
y = df[target_col]
"""

# Things to Reference
st.container() and st.table() and st app design

theming information for .toml files

https://docs.streamlit.io/develop/quick-reference/cheat-sheet

https://www.geeksforgeeks.org/python-sklearn-sklearn-datasets-load_breast_cancer-function/

https://www.geeksforgeeks.org/breast-cancer-wisconsin-diagnostic-dataset/

https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set

https://www.kaggle.com/code/farhanmd29/unsupervised-learning/input

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer

David Smiley's Presentations and Notebooks

https://www.turing.com/kb/guide-to-principal-component-analysis

ClickML






**Principal Component Analysis (PCA)** is a method of Dimensionality Reduction that seeks to capture the maximum variance of a dataset via principal components. By reducing the number of dimensions *without* harming variance, you can make datasets easier to work with and interpret. 

Benefits include:
* Improved computational efficiency
* Enhanced data visualizations
* Better model performance

PCA is an excellent tool for creating visualizations (2D scatterplots) alongside other unsupervised or supervised algorithms. Within this app, you can explore your dataset using PCA alone or use it alongside unsupervised clustering methods. 

:bulb: **Tip:** You selected "Principal Component Analysis (PCA)" for your machine learning algorithm, but you can use it with other algorithms by hitting "Enable PCA" *after* selecting KMeans Clustering or Hierarchical Clustering.


Scroll down to view metrics and visualizations created from your 

Removed sanity check visual from hierarchical clustering:

# -------------------
    # SANITY-CHECK VISUAL
    # -------------------

    with st.container(border=True):
        st.subheader("Sanity-Check Visual")

        sanity_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

        with sanity_tabs[0]:
            # Grabs number of features 
            num_features = numeric_df.shape[1]

            # Calculates a layout for subplot grid
            cols = 3
            rows = int(np.ceil(num_features / cols))
            
            # Creates subplots
            fig8, axes = plt.subplots(rows, cols)
            axes = axes.flatten() # for indexing with a single loop

            # Plot each feature
            for i, column in enumerate(numeric_df.columns):
                numeric_df[column].hist(ax=axes[i], edgecolor='k', bins=15, color="tomato")
                axes[i].set_title(column)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig8.delaxes(axes[j])

            # Display plots
            fig8.suptitle("Distribution of each numeric feature")
            plt.tight_layout()
            st.pyplot(fig8)

        with sanity_tabs[1]:
            st.write("Insert Here")


