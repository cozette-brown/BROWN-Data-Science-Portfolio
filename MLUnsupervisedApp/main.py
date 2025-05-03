import streamlit as st

pages = {
    "Main": [
        st.Page("home.py", title="Home"),
        st.Page("about.py", title="About")
    ],
    "Modeling": [
        st.Page("k_means_clustering.py", title="K-means Clustering"),
        st.Page("hierarchical_clustering.py", title="Hierarchical Clustering"),
        st.Page("principal_component_analysis.py", title="Principal Component Analysis (PCA)")
    ],
}

pg = st.navigation(pages)
pg.run()