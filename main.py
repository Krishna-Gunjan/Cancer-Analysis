import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import pandasql as psql

# Default dataset URL
url = 'https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Cancer.csv'

# Set the title of the app
st.title("Cancer Analysis and Prediction")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "Dataset", "Analysis", "Predict", "SQL Query"])

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
df = df.dropna(axis=1, how='any')
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_columns)

# Convert categorical columns to numeric
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    # Check if the column contains only strings
    if df[col].apply(lambda x: isinstance(x, str)).all():
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    else:
        st.warning(f"Column '{col}' contains non-string values and will be skipped.")

# Initialize target_column and prediction_columns
target_column = None
prediction_columns = []

# Sidebar elements for selecting columns
if page in ["Dataset", "Predict", "Analysis", "SQL Query"]:
    target_column = st.sidebar.selectbox("Select the target column", df.columns, help="Column to be predicted.")
    prediction_columns = st.sidebar.multiselect("Select columns for prediction", [col for col in df.columns if col != target_column], help="Columns used to predict the target.")

# Intro Page
if page == "Intro":
    st.header("Cancer Analysis and Prediction Project")
    st.write("Built by Krishna Gunjan")
    st.write("This app allows you to upload your own cancer dataset, analyze it, and predict cancer diagnosis using Linear Regression.")

# Dataset Page
elif page == "Dataset":
    st.header("Dataset")
    st.markdown("**`Green`**: Value to be predicted | **`Red`**: Values used for prediction")

    # Create an empty placeholder for the DataFrame
    df_placeholder = st.empty()

    if prediction_columns and target_column:
        # Style the dataset to highlight the columns
        def highlight_columns(s):
            if s.name == target_column:
                return ['color: green'] * len(s)
            elif s.name in prediction_columns:
                return ['color: red'] * len(s)
            else:
                return [''] * len(s)

        # Display the styled DataFrame
        styled_df = df.style.apply(highlight_columns, axis=0)

        # Update the placeholder to show the styled DataFrame
        df_placeholder.dataframe(styled_df, height=400)
    else:
        df_placeholder.empty()
        df_placeholder.dataframe(df, height=400)
        # Show the initial DataFrame if no columns are selected
        st.write("Please select columns for prediction and a target column.")

# Analysis Page
elif page == "Analysis":
    st.header("Analysis")
    st.subheader("Diagnosis Distribution")

    graph_type = st.sidebar.selectbox("Select Graph Type", ["Line Chart", "Bar Chart", "Histogram", "Pie Chart"])

    if target_column:
        data = df[target_column]
        distinct_elements = data.unique()
        
        if len(distinct_elements) > 10:
            min_val = data.min()
            max_val = data.max()
            class_size = (max_val - min_val) / 10
            class_ranges = [(min_val + i * class_size, min_val + (i + 1) * class_size) for i in range(10)]
            
            # Count occurrences in each class
            class_counts = []
            for low, high in class_ranges:
                count = data[(data > low) & (data <= high)].count()
                class_counts.append(count)
            
            # Create labels for the classes
            class_labels = [f"{int(low)}-{int(high)}" for low, high in class_ranges]
        else:
            # If there are 10 or fewer distinct elements, use the original data
            class_counts = data.value_counts().sort_index()
            class_labels = class_counts.index.astype(str)
            class_counts = class_counts.values
        
        fig, ax = plt.subplots(figsize=(12, 8))
        legend_bbox_to_anchor = (1.05, 0.5)  # Position legend outside the graph

        if graph_type == "Histogram":
            if len(distinct_elements) > 10:
                ax.bar(class_labels, class_counts, color='skyblue')
                ax.set_xlim(left=0, right=len(class_labels) - 1)
                ax.set_ylim(bottom=0, top=max(class_counts) * 1.1)
            else:
                data.plot(kind='hist', bins=len(distinct_elements), ax=ax, color='skyblue')
                ax.set_xlim(left=min(data), right=max(data))
                ax.set_ylim(bottom=0, top=data.value_counts().max() * 1.1)

        elif graph_type == "Line Chart":
            if len(distinct_elements) > 10:
                ax.plot(class_labels, class_counts, marker='o', color='skyblue')
                ax.set_xlim(left=0, right=len(class_labels) - 1)
                ax.set_ylim(bottom=0, top=max(class_counts) * 1.1)
            else:
                data.value_counts().sort_index().plot(kind='line', marker='o', ax=ax)
                ax.set_xlim(left=min(data), right=max(data))
                ax.set_ylim(bottom=0, top=data.value_counts().max() * 1.1)

        elif graph_type == "Bar Chart":
            if len(distinct_elements) > 10:
                ax.bar(class_labels, class_counts, color='skyblue')
                ax.set_xlim(left=0, right=len(class_labels) - 1)
                ax.set_ylim(bottom=0, top=max(class_counts) * 1.1)
            else:
                data.value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_xlim(left=min(data), right=max(data))
                ax.set_ylim(bottom=0, top=data.value_counts().max() * 1.1)


        elif graph_type == "Pie Chart":
            # Aggregate data for pie chart if necessary
            if len(distinct_elements) > 10:
                pie_sizes = class_counts
                pie_labels = class_labels
            else:
                pie_sizes = data.value_counts()
                pie_labels = pie_sizes.index

            pie = ax.pie(pie_sizes, colors=plt.get_cmap('Set2').colors, startangle=90)
            # Move legend outside the pie chart
            ax.legend(pie[0], pie_labels, loc='center left', bbox_to_anchor=legend_bbox_to_anchor, title='Categories')

        # Move legend outside for other types
        if graph_type != "Pie Chart":
            ax.legend(loc='center left', bbox_to_anchor=legend_bbox_to_anchor, title='Categories')
        
        ax.set_title(f'{graph_type} of {target_column}')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Count')

        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        st.pyplot(fig)
    else:
        st.write("Please select a target column to analyze.")

# Predict Page
elif page == "Predict":
    st.header("Predict Cancer Diagnosis")

    if prediction_columns and target_column:
        # Input fields for the features
        user_input = {}
        for column in prediction_columns:
            user_input[column] = st.number_input(f'Enter {column}', value=float(df[column].mean()))

        # Display the input fields
        features = pd.DataFrame([user_input])

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
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train a Linear Regression model
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make a prediction based on user input
                prediction = model.predict(features)

                # Display the result
                st.write(f"Prediction: {prediction[0]:.2f}")

                # Calculate performance metrics
                from sklearn.metrics import mean_squared_error, r2_score
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

    else:
        st.write("Please select columns for prediction and a target column.")

elif page == "SQL Query":
    st.header("SQL Query Execution")

    # Show the entire table by default
    st.subheader("df")
    st.dataframe(df, height=300)

    # File uploader for SQL queries
    st.subheader("Upload SQL File")
    sql_file = st.file_uploader("Upload a file with SQL queries (one per line)", type=["txt", "sql"])

    if sql_file:
        sql_queries = sql_file.read().decode('utf-8').splitlines()
        for query in sql_queries:
            if query.strip():  # Only run non-empty lines
                try:
                    query_result = psql.sqldf(query, locals())
                    st.write(f"Query: `{query}`")
                    st.dataframe(query_result)  # Display query result
                except Exception as e:
                    st.error(f"Syntax error in query `{query}`: {str(e)}")
    
    # SQL query input area with placeholder and inline submit button
    query_input = st.text_area(
        "Enter your SQL query...",
        height=100,
        placeholder="Enter your SQL query..."
    )

    # Inline submit button for running the query
    submit_button = st.button("➔", key="run_query", use_container_width=True)

    if submit_button:
        try:
            query_result = psql.sqldf(query_input, locals())

            # Display the query and its result in sequence
            st.markdown("### Query Result")
            st.markdown(f"**Query:**\n```sql\n{query_input}\n```")
            st.markdown("**Result:**")
            st.dataframe(query_result)

        except Exception as e:
            st.error(f"Syntax error in SQL query: {str(e)}")
