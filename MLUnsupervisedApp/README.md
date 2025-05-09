# Unsupervised Machine Learning Streamlit Application: Unsoupervised

This ***Unsupervised Machine Learning Streamlit Application,*** which I call "Unsoupervised," allows users to select and train unsupervised machine learning models using sample datasets or their own uploaded .csv file. Users can preview their dataset, select a model, adjust model settings, and view changes to their model performance metrics as they adjust.<br><br>

This app includes the following machine learning models:
* Principal Component Analysis (PCA)
* KMeans Clustering (with optional PCA integration)
* Hierarchical Clustering (with optional PCA integration)

My end objective is to give users the ability to quickly train unsupervised machine learning models and view important performance visualizations, all without having to write their own code.

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Screenshots](#screenshots)

## **Installation**
  
This project is available on the Streamlit Community Cloud at 
[cozettebrown-machine-learning-app.streamlit.io](cozettebrown-unsoupervised.streamlit.io)

To install **MLUnsupervisedApp**, follow these steps:
1. Clone the repository: **`git clone https://github.com/cozette-brown/BROWN-Data-Science-Portfolio/MLUnsupervisedApp.git`**
2. Navigate to the project directory: **`cd MLUnsupervisedApp`**
3. Install dependencies: **`pip install -r requirements.txt`**
4. Run the project: **`streamlit run main.py`**

Requirements for installation include:
* matplotlib==3.7.2
* numpy==1.24.3
* pandas==2.0.3
* scikit_learn==1.6.1
* scipy==1.15.3
* streamlit==1.41.1
* python==3.11
  
## **Usage**

To use **MLUnsupervisedApp**, follow these steps:

1. **Run the Streamlit App:** Navigate to the MLUnsupervisedApp directory and run the project with **`streamlit run main.py`
2. **Upload or Select a Dataset** Select a sample dataset from sklearn.datasets or upload your own .csv file *(NOTE: At this time, the application only supports files which are already prepared for machine learning models. Please tidy and preprocess data before uploading.)*
5. **Select a Machine Learning Model:** Choose from Principal Component Analysis (PCA), KMeans Clustering, or Hierarchical Clustering.
5. **Adjust Settings:** Adjust hyperparameters and option settings in a left sidebar. More information on the available settings can be found in-app, but some basic settings include selecting the number of principal components (for Dimensionality Reduction), setting k number of clusters (for Clustering), or enabling PCA for clustering models.
6. **View Model Performance Metrics:** Analyze visualizations such as 2D scatter plots, or model-specific plots such as scree plots, silhouette scores, and dendrograms. More information is available in-app.

## Data

This project uses the following datasets from Kaggle:
* Country-data.csv [learn more [here](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data?resource=download)]
* Wholesale Customers Data [learn more [here](https://www.kaggle.com/code/farhanmd29/unsupervised-learning/input)]

This project also uses the following sample dataset from sklearn.datasets:
* Breast Cancer Wisconsin [learn more [here]]

Learn more about the dataset [here](https://scikit-learn.org/stable/api/sklearn.datasets.html) on the Scikit Learn API Reference.

Information about the datasets is also available in-app. To view information about each demo dataset, select the dataset in the left sidebar, then under "Dataset Preview" hit the 'ℹ️ Info' tab.

## License

This project is part of a portfolio released under the MIT License. See the portfolio license file **[here](https://github.com/cozette-brown/BROWN-Data-Science-Portfolio/blob/d7c128186047d453de9f2491894e4fd0fa3da77d/LICENSE.md)** for details.

## Acknowledgements

I created this project with help from the following sources:
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

Special thanks to @baselhusam on GitHub, and his app ClickML available on [GitHub](https://github.com/baselhusam/ClickML/tree/main) and [Streamlit Community Cloud](https://clickml.streamlit.app/?ref=streamlit-io-gallery-other). I consulted much of his code and design for the making of my own machine learning app.

## Screenshots
