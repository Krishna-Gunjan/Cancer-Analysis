import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Default dataset URL
url = 'https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Cancer.csv'

# Define columns used for prediction and the target column globally
prediction_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
target_column = 'diagnosis'

# Set the title of the app
st.title("Cancer Analysis and Prediction")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "Dataset", "Analysis", "Predict"])

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv(url)

# Load the dataset globally so it can be accessed on all pages
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"], help="Default dataset is used if no file is uploaded.")
df = load_data(uploaded_file)

# Function to apply color to text based on column
def color_text(val, column_name):
    if column_name == target_column:
        return 'color: green'
    elif column_name in prediction_columns:
        return 'color: red'
    else:
        return ''

# Function to apply the color to the entire dataframe
def color_columns(df):
    return df.style.applymap(lambda val: color_text(val, df.columns.name), subset=pd.IndexSlice[:, df.columns])

# Intro Page
if page == "Intro":
    st.header("Cancer Analysis and Prediction Project")
    st.write("Built by Krishna Gunjan")
    st.write("This app allows you to upload your own cancer dataset, analyze it, and predict cancer diagnosis using Logistic Regression.")

# Dataset Page
elif page == "Dataset":
    st.header("Dataset")
    st.markdown("**`Green`**: Value to be predicted | **`Red`**: Values used to predict")

    # Apply color to the entire dataframe based on the column
    styled_df = df.style.map(lambda val: 'color: green', subset=[target_column])\
                        .map(lambda val: 'color: red', subset=prediction_columns)
    
    st.dataframe(styled_df)

# Analysis Page
elif page == "Analysis":
    st.header("Analysis")
    st.subheader("Diagnosis Distribution")

    # Plot diagnosis distribution
    fig, ax = plt.subplots()
    df['diagnosis'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Diagnosis Distribution')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Predict Page
elif page == "Predict":
    st.header("Predict Cancer Diagnosis")

    # Input fields for the features
    radius_mean = st.number_input('Enter radius_mean', value=14.5)
    texture_mean = st.number_input('Enter texture_mean', value=20.0)
    perimeter_mean = st.number_input('Enter perimeter_mean', value=92.0)
    area_mean = st.number_input('Enter area_mean', value=654.0)
    smoothness_mean = st.number_input('Enter smoothness_mean', value=0.1)

    # Collect user inputs into a dataframe
    features = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]],
                            columns=prediction_columns)

    # Button for prediction
    if st.button("Predict"):
        # Ensure that df is loaded correctly
        if df.empty:
            st.error("The dataset is not loaded correctly.")
        else:
            # Select the features and target from the dataset
            X = df[prediction_columns]
            y = df[target_column]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Logistic Regression model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Make a prediction based on user input
            prediction = model.predict(features)

            # Display the result
            result = "Positive" if prediction[0] == 'M' else "Negative"
            st.write(f"Prediction: {result}")
