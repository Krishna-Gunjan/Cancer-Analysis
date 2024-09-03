---

# Cancer Analysis and Prediction

This project is a Streamlit-based web application that allows users to analyze and predict cancer diagnoses. Users can upload their own dataset, explore the data, visualize different distributions, and make predictions using Linear Regression.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Features

- **Data Upload:** Upload your own CSV dataset for analysis. If no dataset is uploaded, a default cancer dataset is used.
- **Data Cleaning:** Automatically removes columns with missing values and columns that contain a single unique value.
- **Categorical Encoding:** Automatically encodes categorical columns using Label Encoding.
- **Data Visualization:** Provides options to visualize the distribution of the target variable using Line Charts, Bar Charts, Histograms, and Pie Charts.
- **Prediction:** Allows users to input values for selected features and predicts the target variable using Linear Regression. Displays performance metrics like Mean Squared Error (MSE) and R-squared.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Krishna-Gunjan/Cancer-Analysis.git
    cd Cancer-Analysis
    ```

2. **Install dependencies:**

    Make sure you have Python installed (preferably 3.7 or above). Install the required Python packages using:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

## Usage

### Sidebar Navigation

- **Intro:** Provides an introduction to the application and project details.
- **Dataset:** Displays the dataset, allows column selection for prediction, and highlights the selected columns.
- **Analysis:** Visualizes the distribution of the target variable using various chart types.
- **Predict:** Allows the user to input feature values and make predictions using a trained Linear Regression model.

### Data Upload

- You can upload your own CSV file using the sidebar. If no file is uploaded, the app uses a default cancer dataset.

### Visualization

- Select the graph type and view the distribution of the target variable in the Analysis section.

### Prediction

- After selecting the target and prediction columns, input values for each feature, and click "Predict" to get the prediction result and performance metrics.

## Project Structure

- **app.py:** Main application file containing all the Streamlit components and logic.
- **README.md:** Project documentation.
- **requirements.txt:** Python dependencies.

## Acknowledgments

- The default dataset is sourced from the [YBI Foundation's GitHub repository](https://github.com/YBIFoundation/Dataset).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
