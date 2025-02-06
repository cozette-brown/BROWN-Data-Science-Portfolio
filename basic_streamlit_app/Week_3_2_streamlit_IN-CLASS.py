# Import the Streamlit library
import streamlit as st
import pandas as pd

# NAVIGATING YOUR TERMINAL
# ls: shows all files in your directory (make sure you're in the right directory, aka your data science portfolio)
# cd ___: navigate to another area (such as cd Documents) (to shorten this, type cd then the beginning of the folder, then hit tab to autocomplete)
# cd: go back from your subfolders
# CD INTO YOUR BASIC_STREAMLIT_APP

# Display a simple text message
st.title("Hello, streamlit!")
st.markdown("This is my first streamlit app!")

# Display a large title on the app

# ------------------------
# INTERACTIVE BUTTON
# ------------------------
if st.button("Click me!"): # if you click the button
    st.write("You clicked the button.")
else: # otherwise
    st.write("Go ahead...click the button. I dare you.")

# PLAYING AROUND
st.text('Fixed width text')
st.markdown('_Markdown_') # see #*
st.caption('Balloons. Hundreds of them...')
st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.write('Most objects') # df, err, func, keras!
st.write(['st', 'is <', 3]) # see *
st.title('My title')
st.header('My header')
st.subheader('My sub')
st.code('for i in range(8): foo()')

st.button('Hit me')
st.checkbox('Check me out')
st.radio('Pick one:', ['nose','ear'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.color_picker('Pick a color')

# * optional kwarg unsafe_allow_html = True

# Create a button that users can click.
# If the button is clicked, the message changes.

# ------------------------
# COLOR PICKER WIDGET
# ------------------------

color = st.color_picker("Pick a Color!", "#000000")
st.write(f'the current color is {color}.')

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

st.title('Dataset')

# Import pandas for handling tabular data

# Display a section title

# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().