import streamlit as st

pages = {
    "Navigation": [
        st.Page("home.py", title="Home"),
        st.Page("about.py", title="About")
    ]
}

pg = st.navigation(pages)
pg.run()