# File Path Libraries
import os
from pathlib import Path

# Read in .toml file - hasn't been working but not essential for app functioning.
config_dir = Path(__file__).resolve().parent / ".streamlit"
config_path = config_dir / "config.toml"

# Import libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_breast_cancer

from scipy.cluster.hierarchy import linkage, dendrogram

# ---------------------------------
# PAGE CONFIGURATION & INFO DISPLAY
# ---------------------------------

st.set_page_config(layout="wide")

# Display Logo and Title 
script_dir = os.path.dirname(__file__)
image_path = os.path.join(script_dir, 'asset', 'logo.png')
st.image(image_path)

st.markdown("""
**Unsoupervised** allows you to apply unsupervised machine learning models to either an uploaded or selected sample dataset.
Once you've chosen and previewed your dataset, you can select from either the Principal Component Analysis (PCA),
KMeans Clustering, or Hierarchical Clustering models and begin exploratory data analysis. This app is equipped with the ability to
adjust hyperparameters and observe their effects on essential model performance metrics using complete dataframes and clear visualizations. """, unsafe_allow_html=True)

# Invisible Divider
st.write("   ")
st.write("   ")

# Display Quick Start Instructions
with st.container(border=True):
    st.subheader("Quick Start Instructions")
    st.markdown("""This application works best in fullscreen desktop view. With the left sidebar **open,** follow the steps to upload a dataset and begin machine learning! As you adjust your settings in the sidebar, you'll see changes to the main body of the app. Scroll down on this section to preview your dataset and view metrics and visualizations for your selected model.""")
    st.markdown(""":bulb: **Tip:** Click the "About" page (accessible via the left sidebar) to view a brief overview of unsupervised machine learning. Or, for more in-depth information, there will also be helpful hints provided wherever you see an ":information_source: Info" tab""")

# --------------------------
# UPLOAD OR SELECT A DATASET
# --------------------------

# Build file paths for demo datasets
base_dir = Path(__file__).resolve().parent  # Path to the directory where the script is located
country_data_path = base_dir / "data" / "Country-data.csv"
customer_data_path = base_dir / "data" / "Wholesale-customers-data.csv"

# Selection UI
with st.sidebar:
    st.subheader("1. Upload or Select a Dataset")
    dataset = st.selectbox('Dataset selection', ['Country Data', 'Customer Data', 'Breast Cancer Data', 'Upload Your Own'])
 
    # Loads 'Country-data.csv'
    if dataset == 'Country Data': 
        data = pd.read_csv(country_data_path)
    # Loads 'Wholesale-customers-data.csv'
    elif dataset == 'Customer Data':
        data = pd.read_csv(customer_data_path)
    elif dataset == 'Breast Cancer Data':
        breast_cancer = load_breast_cancer()
        data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    # Allows user to upload their own dataset, then loads it
    elif dataset == 'Upload Your Own':
        # Select
        uploaded_file = st.file_uploader("Upload a valid .csv file.", type=["csv"])
        # Load
        data = pd.read_csv(uploaded_file)
    # Warning if nothing loaded correctly (probably won't appear but added just in case)
    else:
        st.warning("Please upload a dataset or use the sample dataset to proceed.")
        st.stop()

# ---------------
# DATASET PREVIEW
# ---------------

# Invisible Divider 
st.write("   ")
st.write("   ")

# Section Header
st.subheader(f"Dataset Preview: {dataset}")

# Configure Tab Layout
dataset_preview_tabs = st.tabs(['🗃 Dataset', 'ℹ️ Info'])
with dataset_preview_tabs[0]:
    # Display Dataframe
    st.dataframe(data)
with dataset_preview_tabs[1]:
    with st.container(border=True):
        # Display information about demo datasets, OR display a warning about uploaded dataset
        # Country Dataset
        if dataset == 'Country Data':
            st.subheader('About the Dataset')
            st.markdown("""
            The Country Data dataset compiles socio-economic and health factors from the different countries of the world. It contains the following features:

            * **country:** Name of the country  
            * **child_mort:** Death of children under 5 years of age per 1000 live births  
            * **exports:** Exports of goods and services per capita. Given as a percentage of the GDP per capita  
            * **health:** Total health spending per capita. Given as a percentage of GDP per capita  
            * **imports:** Imports of goods and services per capita. Given as a percentage of the GDP per capita  
            * **income:** Net income per person  
            * **inflation:** Measurement of the annual growth rate of the Total GDP  
            * **life_expec:** Average number of years a newborn child would live if the current mortality patterns remain the same  
            * **total_fer:** Number of children that would be born to each woman if the current age-fertility rates remain the same  
            * **gdpp:** GDP per capita. Calculated as the Total GDP divided by the total population  

            *'Country-data.csv' was accessed via Kaggle at [this link](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?resource=download). It has been used in this application for demonstration purposes only.*
            """)
        # Customer Dataset
        elif dataset == 'Customer Data':
            st.subheader('About the Dataset')
            st.markdown("""
            The Customer Data dataset compiles data from the clients of a wholesale distributor. All spending is tracked in "monetary units" (m.u.) with no currency specified. While not specified, it seems to be simulating data modeled after real-world places in Portugal. It contains the following features:

            * **Channel:** Whether the customer is a Hotel/Restaurant/Cafe or a Retail channel
            * **Region:** Lisnon, Oporto, or Other
            * **Fresh:** Annual spending on fresh products in m.u.
            * **Milk:** Annual spending on milk products in m.u.  
            * **Grocery:** Annual spending on grocery products in m.u.
            * **Frozen:** Annual spending on frozen products in m.u.
            * **Detergents_Paper:** Annual spending on detergents and paper products in m.u.
            * **Delicassen:** Annual spending on delicatessen products in m.u. 

            *'Wholesale customers data.csv' was accessed via Kaggle at [this link](https://www.kaggle.com/datasets/binovi/wholesale-customers-data-set). It has been used in this application for demonstration purposes only.*
            """)
        # Breast Cancer Dataset
        elif dataset == 'Breast Cancer Data':
            st.subheader('About the Dataset')
            st.markdown("""
            The Breast Cancer dataset compiles data from observations of breast tumors, either malignant or benign. It contains the following features:

            * **mean radius:** Mean of distances from center to points on the perimeter
            * **mean texture:** Standard deviation of gray-scale values
            * **mean perimeter:** Perimeter of the tumor
            * **mean area:** Area of the tumor
            * **mean smoothness:** Variation in radius lengths
            * **mean compactness:** Perimeter^2 / Area - 1.0
            * **mean concavity:** Severity of concave portions of the contour
            * mean concave points:** Number of concave portions of the contour
            * mean symmetry:*8 Symmetry of the cell nuclei
            * mean fractal dimension: "Coastline approximation" - 1

            *The dataset was accessed using load_breast_cancer() via sklearn. It has been used in this application for demonstration purposes only.*
            """)
        # Uploaded Dataset
        else:
            st.subheader('Note on Uploaded Datasets')
            st.markdown("""You must prepare the dataset for use in appropriate machine learning models prior to uploading. Otherwise, you may encounter errors when using the application\'s machine learning algorithms.""")

# -----------------------------
# SELECT MACHINE LEARNING MODEL
# -----------------------------

# Select Model
with st.sidebar:
    st.subheader("2. Select a Machine Learning Algorithm")
    model_types = ["Principal Component Analysis (PCA)", "KMeans Clustering", "Hierarchical Clustering"]
    selected_model = st.selectbox("Select an algorithm", model_types)

# Invisible Divider
st.write("   ")
st.write("   ")

# Display Selected Model Type
st.subheader(f"{selected_model}")
st.markdown(":bulb: **Tip:** You may change your machine learning algorithm in the sidebar to the left.")

# Tab Configuration
model_and_info = st.tabs(['📊 Model Performance', 'ℹ️ Info'])

# Display preview of metrics and visualizations for various models
with model_and_info[0]:
    # PCA
    if selected_model == 'Principal Component Analysis (PCA)':
        st.markdown("""
        Keep scrolling to see:
        * Explained Variance
        * Scatter Plot of PCA Scores
        * Scree Plot
        * Bar Plot
        * Screen & Bar Plots Combined
        """)
    # KMeans Clustering
    elif selected_model == 'KMeans Clustering':
        st.markdown("""
        Keep scrolling to see:
        * Elbow Plot for Optimal k
        * Silhouette Scores for Optimal k
        * 2D Clustering Results *(Note: PCA must be enabled)*
        """)
    # Hierarchical Clustering
    else:
        st.markdown("""
        Keep scrolling to see:
        * Hierarchical Tree Dendrogram
        * Assigned Clusters
        * Silhouette Analysis
        * 2D Clustering Results *(Note: PCA must be enabled)*
        """)
with model_and_info[1]:
    # Display Info about each algorithm type
    with st.container(border=True):
        # PCA
        if selected_model == 'Principal Component Analysis (PCA)':
            # About PCA
            st.subheader("About Principal Component Analysis (PCA)")
            st.markdown("""
            **PCA** is a method of Dimensionality Reduction that seeks to capture the maximum variance of a dataset via principal components. By reducing the number of dimensions *without* harming variance, you can make datasets easier to work with and interpret. 
                
            Benefits include:
            * Improved computational efficiency
            * Enhanced data visualizations
            * Better model performance

            PCA is also an excellent tool for creating visualizations (2D scatterplots) alongside other algorithms. Within this app, you can explore your dataset using PCA alone or use it alongside unsupervised clustering methods. 

            :bulb: **Tip:** You selected "Principal Component Analysis (PCA)" for your machine learning algorithm, but you can use it with other algorithms as well. Simply open the left sidebar and hit "Enable PCA" *after* selecting KMeans Clustering or Hierarchical Clustering to begin.
            """)
            
            # Adjustable Settings
            st.write("   ")
            st.write("   ")
            st.subheader("Adjustable Settings")
            st.markdown("""
            Unsoupervised allows you to adjust the following setting for your PCA model:
            \n* **Desired Number of Components:** Adjust the number of Principal Components, with a minimum of 2 and a maximum of the original number of components
            \n:bulb: **Tip:** Adjustable settings can be found in the sidebar to the left.
            """)
            
            # Included Metrics and Visualizations
            st.write("   ")
            st.write("   ")
            st.subheader("In-app Metrics and Visualizations")
            st.markdown("""
            Unsoupervised includes the following metrics and visualizations:
            * **Explained Variance:** A dataframe showing the Principal Component, the variance explained by that component (Explained Variance Ratio), and the Cumulative Variance of that component and all other included components
            * **Scatter Plot of PCA Scores:** Maps the data on a scatter plot when the number of components equals 2. *(Note: If your selected number of components is higher than 2, the scatter plot will still display.)* May be used to visually identify distinct clusters in the data, which can be helpful for deciding to use Clustering methods on your data.
            * **Scree Plot of Variance Explained:** Plots the cumulative explained variance versus the number of components, which may help you decide the optimal number of components needs to capture the most variance in the data. Look for an "elbow" (a distinct bend in the plotted curve) to decide which number of components is needed
            * **Bar Plot of Variance Explained:** Unlike the scree plot, plots the variance explained by *each* Principal Component using a bar plot. Similarly to the scree plot, this may help you decide where the addition of another Principal Component begins to "fall off" and lack usefulness
            * **Scree & Bar Plots Combined:** Overlays the scree plot and bar plot for comparative purposes, so you may balance the scree plot with the bar plot to make a decision about how many Principal Components are needed
            """)
        
        # KMeans Clustering
        elif selected_model == 'KMeans Clustering':

            # About KMeans Clustering
            st.subheader("About KMeans Clustering")
            st.markdown("""
            **KMeans Clustering** is a method of Clustering that seeks to group data points in a dataset in order to discover hidden structures, sort large datasets into meaningful subgroups, and identify relevant patterns or outliers for further analysis.
                
            Benefits include:
            * Scaling well to large datasets
            * Easily interpretable results
            * Reliable dataset segmentation
            """)

            # Adjustable Settings
            st.write("   ")
            st.write("   ")
            st.subheader("Adjustable Settings")
            st.markdown("""
            Unsoupervised allows you to adjust the following settings for your KMeans Clustering model:
            \n* **Desired Number of Clusters:** Adjust the number of clusters, with a minimum of 2 and a maximum of 10
            \n* **Enable PCA for Visualization:** Enable Principal Component Analysis, so that you can display a 2D scatterplot of your clustered dataset 
            \n:bulb: **Tip:** Adjustable settings can be found in the sidebar to the left.
            """)

            # Included Metrics and Visualizations
            st.write("   ")
            st.write("   ")
            st.subheader("In-app Metrics and Visualizations")
            st.markdown("""
            Unsoupervised includes the following metrics and visualizations:
            * **Elbow Plot for Optimal k:** Plots the number of clusters v. the within-cluster sum of squares on a curve, which may help you decide the most appropriate number of clusters for the dataset. Look for an "elbow" (a distinct bend in the plotted curve) to decide which number of clusters is most optimal
            * **Silhouette Scores for Optimal k:** Plots the average Silhouette Score for each possible model of 2-10 clusters, which may help you decide the most optimal number of clusters. Look for the highest silhouette score to find the most optimal number of clusters
            * **2D Clustering Results:** When PCA is enabled, sorts dataset observations into your selected number of clusters on a color-coded 2D scatter plot, in order to visualize the results of your clustering model
            """)

        # Hierarchical Clustering
        else:
            # About Hierarchical Clustering
            st.subheader("About Hierarchical Clustering")
            st.markdown("""
            **Hierarchical Clustering** is a method of Clustering that builds a tree of nested clusters in order to uncover multi-level structures, segment datasets into flexible variable-sized groups, and identify relevant patterns or outliers for further analysis.
                
            Benefits include:
            * Forms clusters from bottom-up, so observations are a "cluster until proven otherwise"
            * Ability to choose number of clusters *after* you see the dendrogram
            * Full merge history visible for exploratory insight
            """)

            # Adjustable Settings
            st.write("   ")
            st.write("   ")
            st.subheader("Adjustable Settings")
            st.markdown("""
            Unsoupervised allows you to adjust the following settings for your Hierarchical Clustering model:
            \n* **ID Column:** Select the appropriate column with which to label individual observations, and use the rest of the numeric columns as features
            \n* **Truncate Results:** Toggle to display either all observations or a truncated version on the dendrogram
            \n* **Select k:** Select k clusters after viewing your dendrogram
            \n* **Enable PCA for Visualization:** Enable Principal Component Analysis, so that you can display a 2D scatterplot of your clustered dataset 
            \n:bulb: **Tip:** Adjustable settings can be found in the sidebar to the left.
            """)

            # Included Metrics and Visualizations
            st.write("   ")
            st.write("   ")
            st.subheader("In-app Metrics and Visualizations")
            st.markdown("""
            Unsoupervised includes the following metrics and visualizations:
            * **Hierarachical Tree:** Displays hierarchical clustering process on a dendrogram
            * **Assigned Clusters:** A dataframe showing each observation and its assigned cluster
            * **Silhouette Analysis:** Plots the average Silhouette Score for each possible model of 2-10 clusters, which may help you decide the most optimal number of clusters. Look for the highest silhouette score to find the most optimal number of clusters
            * **2D Clustering Results:** When PCA is enabled, sorts dataset observations into your selected number of clusters on a color-coded 2D scatter plot, in order to visualize the results of your clustering model
            """)

# ----------------------------
# PRINCIPAL COMPONENT ANALYSIS
# ----------------------------

if selected_model == 'Principal Component Analysis (PCA)':

    # -----------------------
    # PREPROCESS DATA FOR PCA
    # -----------------------

    # Grab numeric data from dataset
    numeric_df = data.select_dtypes(include=['number'])
    
    # Scale the Data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(numeric_df)

    # Select Number of Components
    with st.sidebar:
        # Display Selected Model Type
        st.subheader("3. Enter the Desired Number of Components")
        component_max = len(numeric_df.columns)
        components = st.number_input("Enter an integer value number of components", 2, component_max)
    
    # -----------
    # COMPUTE PCA
    # -----------

    pca = PCA(n_components=components)
    X_pca = pca.fit_transform(X_std)

    # -------------------------
    # VARIANCE AND SCATTER PLOT
    # -------------------------
    
    # Invisible Divider
    st.write("   ")
    st.write("   ")

    # Configure Column Layout
    varcol, scattercol = st.columns(2)

    # ----------------
    # DISPLAY VARIANCE
    # ----------------

    with varcol:
        with st.container(border=True):
            st.subheader("Explained Variance")

            # Compute the Explained Variance Ratio and Cumulative Variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            # Format Variance as a DataFrame
            variance_df = pd.DataFrame({
                'PC': [f"{i+1}" for i in range(len(explained_variance))],
                'Explained Variance Ratio': explained_variance,
                'Cumulative Variance': cumulative_variance
                })

            # Configure tab layout
            variance_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display formatted dataframe
            with variance_tabs[0]:
                st.dataframe(variance_df)
                st.markdown("""            
                :bulb: **Tip:** To enlarge the dataframe for viewing, hover over the dataframe then hit the fullscreen button which appears in the dataframe's top right corner.
                """)
            # Display info about dataframe
            with variance_tabs[1]:
                st.subheader("About the Dataframe")
                st.markdown("""
                The Explained Variance dataframe displays the following information:
                * **PC:** The Principal Component, from the first component to the user-selected number of components                    
                * **Explained Variance Ratio:** The variance explained by that Principal Component, expressed as a decimal rounded to four spots
                * **Cumulative Variance Ratio:** The total variance explained by the included Principal Components
                
                :bulb: **Tip:** By default, the dataframe will only display information for the first two Principal Components. You may display information for additional Principal Components by adjusting the "Desired Number of Components" in the sidebar to the left, up to the maximum number of components.
                """)

    # --------------------
    # DISPLAY SCATTER PLOT
    # --------------------

    with scattercol:
        with st.container(border=True):
            st.subheader("Scatter Plot of PCA Scores")
    
            # Configure tab layout
            pca_score_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display Scatter Plot
            with pca_score_tabs[0]:
                plt.figure(figsize=(8,6))
                plt.scatter(
                    X_pca[:, 0], # First principal component
                    X_pca[:, 1], # Second principal component
                    color = ['tomato'],
                    alpha=0.7,
                    edgecolor='k',
                    s=60                    
                )
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.grid(True)
                plt.title("2D Projection of Data")
                st.pyplot(plt)
                st.markdown("""
                :bulb: **Tip:** To enlarge the plot for viewing, hover over the plot then hit the fullscreen button which appears in the plot's top right corner.
                
                *This scatter plot displays data for when there are two Principal Components. It will remain on screen even if you adjust the number of components to be above 2.*
                """)
            # Display info about Scatter Plot
            with pca_score_tabs[1]:
                st.subheader("About the Plot")
                st.markdown("""
                The scatter plot of PCA scores displays the following information:
                * **[x-axis] Principal Component 1:** PC1 after standardization                   
                * **[y-axis] Principal Component 2:** PC2 after standardization
                """)
                st.write("   ")
                st.markdown("""
                **Interpretation:**
                * Each point represents an observation from the dataset
                * Points close together indicate similar datapoints. This is useful to visualize for clustering methods       
                """)

    # Setup for side-by-side Scree Plot, Bar Plot, and combined Scree/Bar Plot

    with st.container(border=True):
        scree_bar_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

        with scree_bar_tabs[0]:

            pltcol1, pltcol2 = st.columns([1, 2])

            with pltcol1:
                
                # ------------------
                # DISPLAY SCREE PLOT
                # ------------------

                st.subheader("Scree Plot")
                pca_full = PCA(n_components=component_max).fit(X_std)
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                fig1, ax1 = plt.subplots()
                ax1.plot(
                    range(1, len(cumulative_variance)+1),
                    cumulative_variance,
                    marker='o',
                    color='tomato'
                )
                ax1.set_xlabel('Number of Components')
                ax1.set_ylabel('Cumulative Explained Variance')
                ax1.set_title('PCA Variance Explained')
                ax1.set_xticks(range(1, len(cumulative_variance)+1))
                ax1.grid(True)
                st.pyplot(fig1)

                # Set explained and components variables
                explained = pca_full.explained_variance_ratio_ * 100  # individual variance (%) per component
                components = np.arange(1, len(explained) + 1)
                
                # ----------------
                # DISPLAY BAR PLOT
                # ----------------

                st.subheader("Bar Plot")
                fig2, ax2 = plt.subplots()
                ax2.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='darkseagreen')
                ax2.set_xlabel('Principal Component')
                ax2.set_ylabel('Variance Explained')
                ax2.set_title('Variance Explained by Each Principal Component')
                ax2.set_xticks(components)
                ax2.grid(True, axis='y')
                st.pyplot(fig2)
                
            with pltcol2: 

                # ----------------------
                # DISPLAY COMBINED PLOTS
                # ----------------------

                # Set cumulative variable
                cumulative = np.cumsum(explained)

                # Combined Plots
                st.subheader("Scree & Bar Plots Combined")
                fig3, ax3 = plt.subplots()

                # Bar plot for individual variance explained
                bar_color = 'darkseagreen'
                ax3.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
                ax3.set_xlabel('Principal Component')
                ax3.set_ylabel('Individual Variance Explained (%)', color=bar_color)
                ax3.tick_params(axis='y', labelcolor=bar_color)
                ax3.set_xticks(components)
                ax3.set_xticklabels([f"{i}" for i in components])

                # Add percentage labels on each bar
                for i, v in enumerate(explained):
                    ax3.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')

                # Create a second y-axis for cumulative variance explained
                ax4 = ax3.twinx()
                line_color = 'tomato'
                ax4.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
                ax4.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
                ax4.tick_params(axis='y', labelcolor=line_color)
                ax4.set_ylim(0, 100)

                # Remove grid lines
                ax3.grid(False)
                ax4.grid(False)

                # Combine legends from both axes and position the legend inside the plot (middle right)
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax4.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))

                plt.title('PCA: Variance Explained', pad=20)
                plt.tight_layout()
                st.pyplot(plt)

                st.markdown(""":bulb: **Tip:** To enlarge the plots for viewing, hover over each plot then hit the fullscreen button which appears in the plot's top right corner.""")

        # Display info about Scree, Bar, and Combined Plots
        with scree_bar_tabs[1]:
            # Describe Scree Plot
            st.subheader("About the Scree Plot")
            st.markdown("""
            The scree plot displays the following information:
            * **[x-axis] Number of Components:** How many components are included in the new model after PCA
            * **[y-axis] Cumulative Explained Variance:** The total amount of variance captured by the new model, for all included components together
            """)
            st.write("   ")
            st.markdown("""**Interpretation:**
            \n* Each point on the plot shows the variance captured by the number of retained components
            \n* A scree plot helps determine how many components to retain when applying PCA. When plotted, it will display a curve. Look for the "elbow" in the curve, or the place where it noticeably bends and begins to level off, to decide what number of components is best for your dataset
            """)
            # Describe Bar Plot
            st.write("   ")
            st.subheader("About the Bar Plot")
            st.markdown("""
            The axes of the bar plot are similar to the scree plot, with one key difference:
            * **[x-axis] Number of Components:** How many components are included in the new model after PCA
            * **[y-axis] Variance Explained:** The ratio of variance explained by a *singular* component. Not to be confused with the cumulative variance, which is the variance caputured by the component *and* all previously retained components
            """)
            st.write("   ")
            st.markdown("""
            **Interpretation:**
            * Each bar shows the variance captured by component number x (as seen on the x-axis)
            * Similarly to the scree plot, you'll want to pay attention to where the bar plot begins to "fall off" to decide where to stop adding components            
            """)
            # Describe Combined
            st.write("   ")
            st.subheader("About the Combined Plots")
            st.markdown("""The combined plot shows the scree plot and bar plot together, making it easier to compare results from both plots. It also displays percent variance captured by each component above each bar, so you can see individual variance expressed as a percent. In this plot, there are two y-axes:
            \n* **[left] Individual Variance Explained (%):** Variance captured by an individual component, as a percent
            \n* **[right] Cumulative Variance Explained (%):** Cumulative variance captured by the component and all previously retained components, as a percent""")

# -----------------
# KMEANS CLUSTERING
# -----------------

elif selected_model == 'KMeans Clustering':
    
    # -------------------------------------
    # PREPROCESS DATA FOR KMEANS CLUSTERING
    # -------------------------------------

    # Grab numeric data from dataset
    numeric_df = data.select_dtypes(include=['number'])
    
    # Scale the Data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(numeric_df)

    # Select Number of Clusters
    with st.sidebar:
        # Display Selected Model Type
        st.subheader("3. Enter the Desired Number of Clusters")
        cluster_max = 10
        k = st.number_input("Enter an integer value (Note: Maximum 10)", 2, cluster_max)

    # -------------------------
    # COMPUTE KMEANS & CLUSTERS
    # -------------------------

    kmeans = KMeans(n_clusters=k, random_state=42) # random_state for consistency
    clusters = kmeans.fit_predict(X_std)

    # --------------------------------
    # EVALUATE BEST NUMBER OF CLUSTERS
    # --------------------------------

    # Define range of k to try
    ks = range(2, 11) # 2 through 10 clusters

    # Store Within-Cluster Sum of Squares for each k value
    wcss = []

    # Store Silhoutte scores for each k
    silhouette_scores = []

    # Loop over range of k values to get wcss and silhouette_scores
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42) # using random state for consistency
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))

    # -------------------------------------------
    # VISUALIZE RESULTS (BEST NUMBER OF CLUSTERS)
    # -------------------------------------------

    st.write("   ")
    st.write("   ")

    numclustcol1, numclustcol2 = st.columns(2)

    with numclustcol1:
        with st.container(border=True):
            # Elbow Method Plot
            st.subheader("Elbow Plot")
            elbow_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])
            with elbow_tabs[0]:
                fig5, ax5 = plt.subplots()
                ax5.plot(ks, wcss, marker='o', color='tomato')
                ax5.set_title('Elbow Plot for Optimal k')
                ax5.set_xlabel('Number of Clusters (k)')
                ax5.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
                ax5.grid(True)
                st.pyplot(fig5)
                st.markdown(""":bulb: **Tip:** To enlarge the dataframe for viewing, hover over the dataframe then hit the fullscreen button which appears in the dataframe's top right corner.""")
            with elbow_tabs[1]:
                st.subheader("About the Elbow Plot")
                st.markdown("""
                The elbow plot for optimal k displays the following information:
                * **[x-axis] Number of Clusters (k):** The number of possible clusters from 2-10                  
                * **[y-axis] Within-Cluster Sum of Squares (WCSS):** The sum of the squared distance between each point and the centroid in a cluster. Measures how tightly points are clustered around a centroid, with a lower WCSS indicating more tightly-clustered groups
                """)
                st.write("   ")
                st.markdown("""
                **Interpretation:**
                * Each point on the plot represents the WCSS for each number of clusters 2-10
                * The elbow plot helps you decide the optimal number of clusters. Look for the "elbow" in the curve, or the the place where decreases most noticeably begin to level off, to determine the optimal number of clusters   
                """)
        
    with numclustcol2:
        with st.container(border=True):
            # Silhouette Score Plot
            st.subheader("Silhouette Scores")
            silhouette_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])
            with silhouette_tabs[0]:
                fig6, ax6 = plt.subplots()
                ax6.plot(ks, silhouette_scores, marker='o', color='darkseagreen')
                ax6.set_title('Silhouette Scores for Optimal k')
                ax6.set_xlabel('Number of Clusters (k)')
                ax6.set_ylabel('Silhouette Score')
                ax6.grid(True)
                st.pyplot(fig6)
                st.markdown(""":bulb: **Tip:** To enlarge the dataframe for viewing, hover over the dataframe then hit the fullscreen button which appears in the dataframe's top right corner.""")

            with silhouette_tabs[1]:
                st.subheader("About the Silhouette Scores Plot")
                st.markdown("""
                The silhouette scores plot for optimal k displays the following information:
                * **[x-axis] Number of Clusters (k):** The number of possible clusters from 2-10                  
                * **[y-axis] Silhouette Score:** The silhouette score for clustered observations, which quantifies how similar an observation is to other observations in its own cluster compared to other clusters. A higher silhouette score indicates better clustering
                """)
                st.write("   ")
                st.markdown("""
                **Interpretation:**
                * Each point on the plot represents the silhouette score for each number of clusters 2-10
                * This plot helps you decide the optimal number of clusters. Look for the highest silhouette score to determine the optimal number of clusters   
                """)

    # -------------------------------------------------------
    # PRINCIPAL COMPONENT ANALYSIS (FOR VISUALIZING CLUSTERS)
    # -------------------------------------------------------

    # Enable/Disable Principal Component Analysis
    with st.sidebar:
        st.subheader("4. Enable Principal Component Analysis for Visualization?")
        pca_true = st.toggle("Enable PCA")
        with st.container(border=True):
            st.markdown("<small>:bulb: **Tip:** Enabling will reduce the number of dimensions to 2 in order to visualize clustering on a 2D scatter plot, but *only* for that visualization. Scatter plot will not display if disabled.</small>", unsafe_allow_html=True)

    # Principal Component Analysis

    if pca_true == True:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_std)

    # ----------------------------------------------------------------
    # DISPLAY SCATTER PLOT OF CLUSTERING RESULTS (PCA MUST BE ENABLED)
    # ----------------------------------------------------------------

    with st.container(border=True):
        st.subheader("2D Clustering Results")
        pca_projection_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

        # Display Scatter Plot
        with pca_projection_tabs[0]:
            if pca_true == True:
                fig7, ax7 = plt.subplots()

                colors = ['tomato', 'darkseagreen', 'cadetblue', 'goldenrod', 'plum',
                    'cornflowerblue', 'sandybrown', 'pink', 'seagreen', 'sienna']

                # Initialize for dynamic legend that responds to user inputted 'k'
                handles = []
                labels = []

                # Plot clusters
                for cluster_label in range(k):
                    ax7.scatter(
                        X_pca[clusters == cluster_label, 0],
                        X_pca[clusters == cluster_label, 1],
                        color=colors[cluster_label],
                        alpha = 0.7,
                        edgecolor = 'k',
                        s = 60,
                        label=f"Cluster {cluster_label}"
                    )
                    ax7.set_xlabel('Principal Component 1')
                    ax7.set_ylabel('Principal Component 2')
                    ax7.set_title('KMeans Clustering: 2D PCA Projection')
                    ax7.legend(loc='best')
                    ax7.grid(True)

                st.pyplot(fig7)
                st.markdown(""":bulb: **Tip:** To enlarge the dataframe for viewing, hover over the dataframe then hit the fullscreen button which appears in the dataframe's top right corner.""")

            else:
                st.markdown(""":bulb: **Tip:** PCA must be enabled to display results. You may enable PCA in the left sidebar.""")

        with pca_projection_tabs[1]:
            st.subheader("About the Plot")
            st.markdown("""
            The scatter plot of the PCA projection scores displays the following information:
            * **[x-axis] Principal Component 1:** PC1 after standardization                   
            * **[y-axis] Principal Component 2:** PC2 after standardization
            """)
            st.write("   ")
            st.markdown("""
            **Interpretation:**
            * Each point represents an observation from the dataset
            * Each color represents a cluster. The number of clusters displayed is determined by the desired number of clusters set in the sidebar to the left     
            """)

# -----------------------
# HIERARCHICAL CLUSTERING
# -----------------------

else:

    # -------------------------------------------
    # PREPROCESS DATA FOR HIERARCHICAL CLUSTERING
    # -------------------------------------------

    # Grab numeric data from dataset
    numeric_df = data.select_dtypes(include=['number'])

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)


    # ----------------
    # HIERACHICAL TREE
    # ----------------

    # Select label from dataframe
    with st.sidebar:
        # Selecting Label Column
        st.subheader("3. Select Correct ID Column from Dataset")
        columns = data.columns.tolist()
        label = st.selectbox('Column Selection', columns)

        # Option to Truncate
        st.subheader("4. Truncate Dendrogram Results?")
        st.markdown("<small>If disabled, all observations will be shown on the dendrogram. If enabled, a truncated version will be shown. Enable for readability. </small>", unsafe_allow_html=True)
        truncate_true = st.toggle("Truncate Results")
    
    # Column configuration
    treecol, clustercol = st.columns([5,3])

    with treecol:
        with st.container(border=True):
            st.subheader("Hierarchical Tree")
            tree_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display dendrogram
            with tree_tabs[0]:
                # Define colors for tree
                threshold = 10
                colors = ['tomato', 'darkseagreen', 'cadetblue', 'goldenrod', 'plum',
                            'cornflowerblue', 'sandybrown', 'pink', 'seagreen', 'sienna']
                
                # Set color cycle to custom colors
                plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

                # Standardize the numeric features for centering and scaling
                Z = linkage(X_scaled, method="ward")
                
                fig9, ax9 = plt.subplots(figsize=(15,7))

                if truncate_true == True:
                    dendrogram(Z,
                            truncate_mode="lastp",
                            labels = data[label].to_list(),
                            ax=ax9)
                else:
                    dendrogram(Z,
                            labels = data[label].to_list(),
                            ax=ax9)   
                                     
                ax9.set_title(f"Hierarchical Clustering Dendrogram ({label})")
                ax9.set_xlabel(f"{label}")
                ax9.set_ylabel("Distance")

                
                # Display the tree
                st.pyplot(fig9)

                st.markdown(""":bulb: **Tip:** To enlarge the dendrogram for viewing, hover over the dendrogram then hit the fullscreen button which appears in the plot's top right corner.""")

            # Display info about dendrogram
            with tree_tabs[1]:
                st.subheader("About the Dendrogram")
                st.markdown("""
                The dendrogram displays the following information:
                * **[x-axis] Observation IDs:** Displays the ID of each observation using information from the column selected by a user in step 3 of the left sidebar. Can be truncated. Observations will be ordered according to clustering process             
                * **[y-axis] Distance:** The distance, or dissimilarity, between clusters
                
                **Interpretation:**
                * Splits in the tree graph show divisions from a singular cluster into smaller clusters, and all the way down to individual observations
                * Each observation will be clustered into color-coded groups, which you can view in full or in a truncated version
                * Pay attention to the number of colors that are *touching* the x-axis. This may indicate the appropriate k-value (for instance, if there are three colors, k=3)""")
    
    # --------------------------
    # CHOOSE K & ASSIGN CLUSTERS
    # --------------------------

    # Choose k
    with st.sidebar:
        st.subheader("5. Select k ")
        st.markdown("<small>:bulb: Tip: Select AFTER visually inspecting the dendrogram.</small>", unsafe_allow_html=True)
        k = st.number_input("Enter an integer value number for k", 2, 10)

    # Being Agglomerative clustering for Scatter Plot 
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    data["Cluster"] = agg.fit_predict(X_scaled)

    # Display Assigned Clusters Dataframe
    with clustercol:
        with st.container(border=True):
            st.subheader("Assigned Clusters")
            cluster_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display dataframe
            with cluster_tabs[0]:
                clustered_df = data[[label, "Cluster"]]
                st.dataframe(clustered_df)

                cluster_labels = data["Cluster"].to_numpy()

                st.markdown(""":bulb: **Tip:** To enlarge the dataframe for viewing, hover over the dataframe then hit the fullscreen button which appears in the plot's top right corner.""")
                st.write("   ")
                st.markdown(""":bulb: **Tip:** Hit "Cluster" on the dataframe to sort results by cluster.""")

            # Display information about dataframe
            with cluster_tabs[1]:
                st.subheader("About the Dataframe")
                st.markdown("""
                The Assigned Clusters dataframe displays the following information:
                * **Selected Label:** An identifier for each observation, as pulled from a user-selected column                    
                * **Cluster:** The cluster each observation has been assigned to. By clicking "Cluster" you can sort the dataframe by cluster
                """)

    # ----------------------------------------
    # DISPLAYING OPTIMAL K W/ SILHOUETTE ELBOW
    # ----------------------------------------

    # Column configuration
    silhouettecol, clusteringcol = st.columns(2)

    # Elbow Plot
    with silhouettecol:
        with st.container(border=True):
            st.subheader("Silhouette Analysis")

            silanalysis_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display Elbow Plot
            with silanalysis_tabs[0]:
                k_range = range(2, 11)
                sil_scores = []

                for k in k_range:
                    # Fit hierarchical clustering with Ward linkage
                    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
                    # Silhouette: +1 = dense & well‑separated, 0 = overlapping, −1 = wrong clustering
                    score = silhouette_score(X_scaled, labels)
                    sil_scores.append(score)
                    
                # Plot the curve
                fig11, ax11 = plt.subplots()
                ax11.plot(list(k_range), sil_scores, marker="o")
                ax11.set_xticks(list(k_range))
                ax11.set_xlabel("Number of Clusters (k)")
                ax11.set_ylabel("Average Silhouette Score")
                ax11.set_title("Silhouette Analysis for Agglomerative (Ward) Clustering")
                ax11.grid(True, alpha=0.3)

                st.pyplot(fig11)

                st.markdown(""":bulb: **Tip:** To enlarge the plot for viewing, hover over the plot then hit the fullscreen button which appears in the plot's top right corner.""")

            # Display information about silhouette analysis plot
            with silanalysis_tabs[1]:
                st.subheader("About the Silhouette Analysis Plot")
                st.markdown("""
                The silhouette analysis plot displays the following information:
                * **[x-axis] Number of Clusters (k):** The number of possible clusters from 2-10                  
                * **[y-axis] Silhouette Score:** The average silhouette score for clustered observations, which quantifies how similar an observation is to other observations in its own cluster compared to other clusters. A higher silhouette score indicates better clustering
                """)
                st.write("   ")
                st.markdown("""
                **Interpretation:**
                * Each point on the plot represents the silhouette score for each number of clusters 2-10
                * This plot helps you decide the optimal number of clusters. Look for the highest silhouette score to determine the optimal number of clusters   
                """)

    # -------------------------------------------------------
    # PRINCIPAL COMPONENT ANALYSIS (FOR VISUALIZING CLUSTERS)
    # -------------------------------------------------------

    # Enable Principal Component Analysis
    with st.sidebar:
        # Enable/Disable PCA
        st.subheader("6. Enable Principal Component Analysis for Visualization?")
        st.markdown("<small>:bulb: **Tip:** Enabling will reduce the number of dimensions to 2 in order to visualize clustering on a 2D scatter plot, but *only* for that visualization. Scatter plot will not display if disabled.</small>", unsafe_allow_html=True)
        pca_true = st.toggle("Enable PCA")

    # Principal Component Analysis

    if pca_true == True:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)


    # ----------------------------------------------------------------
    # DISPLAY SCATTER PLOT OF CLUSTERING RESULTS (PCA MUST BE ENABLED)
    # ----------------------------------------------------------------

    with clusteringcol:
        with st.container(border=True):
            st.subheader("2D Clustering Results")

            clustering_tabs = st.tabs(['🔎 View', 'ℹ️ Info'])

            # Display Clustering Scatter Plot
            with clustering_tabs[0]:
                if pca_true == True:
                
                    fig10, ax10 = plt.subplots()

                    # List of colors to use for each cluster
                    colors = ['tomato', 'darkseagreen', 'cadetblue', 'goldenrod', 'plum',
                            'cornflowerblue', 'sandybrown', 'pink', 'seagreen', 'sienna']

                    # Plot each cluster separately
                    for cluster_label in range(k):
                        ax10.scatter(
                            X_pca[cluster_labels == cluster_label, 0],  # X-axis data for the current cluster
                            X_pca[cluster_labels == cluster_label, 1],  # Y-axis data for the current cluster
                            color=colors[cluster_label],  # Assign color based on the cluster
                            alpha=0.7,
                            edgecolor='k',
                            s=60,
                            label=f"Cluster {cluster_label}"
                        )

                    ax10.set_xlabel('Principal Component 1')
                    ax10.set_ylabel('Principal Component 2')
                    if dataset == 'Upload Your Own':
                        ax10.set_title(f'Agglomerative Cluster on Uploaded Dataset via PCA')
                    else:
                        ax10.set_title(f'Agglomerative Cluster on {dataset} via PCA')
                    ax10.legend(title="Clusters")
                    ax10.grid(True)

                    st.pyplot(fig10)

                    st.markdown(""":bulb: **Tip:** To enlarge the plot for viewing, hover over the plot then hit the fullscreen button which appears in the plot's top right corner.""")

                else:
                    st.markdown(""":bulb: **Tip:** PCA must be enabled to display results. You may enable PCA in the left sidebar.""")

            # Display information about Scatter Plot
            with clustering_tabs[1]:
                st.subheader("About the Scatter Plot")
                st.markdown("""
                The scatter plot of agglomerative clustering displays the following information:
                * **[x-axis] Principal Component 1:** PC1 after standardization                   
                * **[y-axis] Principal Component 2:** PC2 after standardization
                """)
                st.write("   ")
                st.markdown("""
                **Interpretation:**
                * Each point represents an observation from the dataset
                * Each color represents a cluster. The number of clusters displayed is determined by the desired number of clusters (k) set in the sidebar to the left     
                """)

