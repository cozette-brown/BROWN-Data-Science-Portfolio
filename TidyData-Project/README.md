# Tidy Data Project: Federal R&D Budget

***Tidy Data Project: Federal R&D Budget*** applies the principles of tidy data in order to tidy a Federal R&D Budget dataset `fed_rd_year&gdp.csv` and create data visualizations from the tidied data, using a Jupyter Notebook.<br><br>

I've followed these basic principles of tidy data for my project of reshaping the original dataset:
* Each variable is in its own column
* Each observation is in its own row
* Each observational unit forms its own table

The included Jupyter notebook contains a more detailed, step-by-step description of what I did to tidy the data.<br><br>

My end objective is to give viewers of the tidied dataset the capabilities to visualize the data in two meaningful ways:
1. **Comparatively by Department:** See changes over time within a department, or compare departments for each fiscal year.
2. **As a Percentage of GDP:** See changes in the total budgeted for R&D as a percentage of US GDP over time.

## Table of contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Screenshots](#screenshots)

## **Installation**

**Requirements:** <br>
This project uses a Python environment on Jupyter Notebooks.

To install **Tidy Data Project: Federal R&D Budget**, follow these steps:
1. Clone the repository: **`git clone https://github.com/cozette-brown/TidyData-Project.git`**
2. Navigate to the project directory: **`cd TidyData-Project`**
3. Install dependencies: **`pip install -r requirements.txt`**
4. Build the project: **`npm run build`**
5. Start the project: **`npm start`**

## **Usage**

To use **Tidy Data Project: Federal R&D Budget**, follow these steps:

1. **Run the Jupyter Notebook:** Open and run the .ipynb file in your favorite coding environment (will require a Python environment)
2. **Follow the Notebook** Read through the notebook to see an explanation of the data tidying process used before viewing visualizations.
3. **Explore Results:** View some general visualizations and insights at the end of the notebook.

## Data

This project uses the dataset **`fed_rd_year&gdp.csv`**. Provided by David Smiley for the University of Notre Dame's *MDSC 20009: Introduction to Data Science* course. Adapted from `fed_r_d_spending.csv` on GitHub (see it [here](https://github.com/rfordatascience/tidytuesday/blob/main/data/2019/2019-02-12/fed_r_d_spending.csv)).

The tidied dataset has four columns:
| variable      | class     | description                                                       |
| ------------- | --------- | ----------------------------------------------------------------- |
| department    | object    | US agency/department                                              |
| rd_budget     | float64   | Research and Development dollars in inflation-adjusted US dollars |
| year          | object    | Fiscal Year                                                       |
| total_gdp     | float64   | Total US Gross Domestic Product in inflation-adjusted US dollars  |

<br>

For your reference, here are the abbreviations for each US agency/department:
* DOD - Department of Defense
* NASA - National Aeronautics and Space Administration
* DOE - Department of Energy
* HHS - Department of Health and Human Services
* NIH - National Institute of Health
* NSF - National Science Foundation
* USDA - US Department of Agriculture
* Interior - Department of Interior
* DOT - Department of Transportation
* EPA - Environmental Protection Agency
* DOC - Department of Corrections
* DHS - Department of Homeland Security
* VA - Department of Veterans Affairs
* Other - Other research and development spending

## License

This project is part of a portfolio released under the MIT License. See the portfolio license file **[here](https://github.com/cozette-brown/BROWN-Data-Science-Portfolio/blob/d7c128186047d453de9f2491894e4fd0fa3da77d/LICENSE.md)** for details.

## Acknowledgements

**Tidy Data Project: Federal R&D Budget** was created by **[Cozette Brown](https://github.com/cozette-brown)**.

## Screenshots
[![View my data science portfolio](data-science-portfolio-button.jpg)](https://www.github.com/cozette-brown/BROWN-Data-Science-Portfolio)

