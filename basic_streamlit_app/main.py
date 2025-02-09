import streamlit as st
import pandas as pd

# =====================
# DISPLAY CONFIGURATION
# =====================

st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([3, 1, 1])


# ====================================
# READ CSV AND PREPARE IT FOR DISPLAY
# ====================================

df = pd.read_csv("data/data.csv")
df["Avg. User Rating - IMDB"] = pd.to_numeric(df["Avg. User Rating - IMDB"])
df["Avg. User Rating - Rotten Tomatoes"] = pd.to_numeric(df["Avg. User Rating - Rotten Tomatoes"])
# Sample data credit: SILFHJASDLFJKHASDKLJFHASDLF

# ============
# MAIN DISPLAY 
# ============

with col1:
    st.title("James Bond in Film") 
    st.write("Display everything you\'d ever want to know about James Bond movies!")
    st.dataframe(df) 

# =============
# APPLY FILTERS
# =============

with col2:
    st.subheader("Filters")
    # Year Slider
    min_year, max_year = st.slider("Choose a year range:", min_value= df["Year"].min(), max_value=df["Year"].max(), value=(df["Year"].min(), df["Year"].max()))
    # Gross Sliders
    min_us, max_us = st.slider("Choose the U.S. Gross:", min_value= df["U.S. Gross"].min(), max_value=df["U.S. Gross"].max(), value=(df["U.S. Gross"].min(), df["U.S. Gross"].max()))
    min_usadj, max_usadj = st.slider("Choose the Adjusted U.S. Gross:", min_value= df["Adjusted U.S. Gross"].min(), max_value=df["Adjusted U.S. Gross"].max(), value=(df["Adjusted U.S. Gross"].min(), df["Adjusted U.S. Gross"].max()))
    min_world, max_world = st.slider("Choose the World Gross:", min_value= df["World Gross"].min(), max_value=df["World Gross"].max(), value=(df["World Gross"].min(), df["World Gross"].max()))
    min_worldadj, max_worldadj = st.slider("Choose the Adjusted World Gross:", min_value= df["Adjusted World Gross"].min(), max_value=df["Adjusted World Gross"].max(), value=(df["Adjusted World Gross"].min(), df["Adjusted World Gross"].max()))
    # Budget Slider
    min_budget, max_budget = st.slider("Choose the Budget:", min_value= df["Budget"].min(), max_value=df["Budget"].max(), value=(df["Budget"].min(), df["Budget"].max()))
    min_budgetadj, max_budgetadj = st.slider("Choose the Adjusted Budget:", min_value= df["Adjusted Budget"].min(), max_value=df["Adjusted Budget"].max(), value=(df["Adjusted Budget"].min(), df["Adjusted Budget"].max()))
    # Film Length Slider
    min_length, max_length = st.slider("Choose the Film Length:", min_value= int(df["Film Length"].min()), max_value=int(df["Film Length"].max()), value=(int(df["Film Length"].min()), int(df["Film Length"].max())))
    # Ratings Slider
    min_rateimdb, max_rateimdb = st.slider("Choose the IMDB Rating:", min_value=float(df["Avg. User Rating - IMDB"].min()), max_value=float(df["Avg. User Rating - IMDB"].max()), value=(float(df["Avg. User Rating - IMDB"].min()), float(df["Avg. User Rating - IMDB"].max())))
    
with col3:
    # Ratings Slider Continued
    min_ratert, max_ratert = st.slider("Choose the Rotten Tomatoes Rating:", min_value= df["Avg. User Rating - Rotten Tomatoes"].min(), max_value=df["Avg. User Rating - Rotten Tomatoes"].max(), value=(df["Avg. User Rating - Rotten Tomatoes"].min(), df["Avg. User Rating - Rotten Tomatoes"].max()))
    # Kills Slider
    min_bondkills, max_bondkills = st.slider("Choose the number of kills by Bond:", min_value= df["Kills by Bond"].min(), max_value=df["Kills by Bond"].max(), value=(df["Kills by Bond"].min(), df["Kills by Bond"].max()))
    min_otherkills, max_otherkills = st.slider("Choose the number of kills by others:", min_value= df["Kills by Others"].min(), max_value=df["Kills by Others"].max(), value=(df["Kills by Others"].min(), df["Kills by Others"].max()))
    # Multiselect Filters
    bond = st.multiselect("Choose your Bonds:", df["Bond"].unique())
    director = st.multiselect("Choose your director(s):", df["Director"].unique())
    bond_car = st.multiselect("Choose your Bond's Car Manufacturer:", df["Bond Car Manufacturer"].unique())
    martinis = st.multiselect("Choose your Martinis Consumed:", df["Martinis Consumed"].unique())
    bjb = st.multiselect('Choose the number of times that \"Bond, James Bond\" is said:', df["Bond James Bond Count"].unique())

# ===============
# FILTER RESULTS
# ===============

df = df[
    (df["Year"] >= min_year) & (df["Year"] <= max_year) & 
    (df["U.S. Gross"] >= min_us) & (df["U.S. Gross"] <= max_us) & 
    (df["Adjusted U.S. Gross"] >= min_usadj) & (df["Adjusted U.S. Gross"] <= max_usadj) & 
    (df["World Gross"] >= min_world) & (df["World Gross"] <= max_world) & 
    (df["Adjusted World Gross"] >= min_worldadj) & (df["Adjusted World Gross"] <= max_worldadj) & 
    (df["Budget"] >= min_budget) & (df["Budget"] <= max_budget) & 
    (df["Adjusted Budget"] >= min_budgetadj) & (df["Adjusted Budget"] <= max_budgetadj) & 
    (df["Film Length"] >= min_length) & (df["Film Length"] <= max_length) & 
    (df["Avg. User Rating - IMDB"] >= min_rateimdb) & (df["Avg. User Rating - IMDB"] <= max_rateimdb) & 
    (df["Avg. User Rating - Rotten Tomatoes"] >= min_ratert) & (df["Avg. User Rating - Rotten Tomatoes"] <= max_ratert) & 
    (df["Kills by Bond"] >= min_bondkills) & (df["Kills by Bond"] <= max_bondkills) &
    (df["Kills by Others"] >= min_otherkills) & (df["Kills by Others"] <= max_otherkills)]

if bond:
    df = df[df["Bond"].isin(bond)]

if director:
    df = df[df["Director"].isin(director)]

if bond_car:
    df = df[df["Bond Car Manufacturer"].isin(bond_car)]

if martinis:
    df = df[df["Martinis Consumed"].isin(martinis)]

if bjb:
    df = df[df["Bond James Bond Count"].isin(bjb)]   

# ===============
# DISPLAY RESULTS
# ===============

with col1:
    st.subheader("Filtered Results")
    st.dataframe(df) 

