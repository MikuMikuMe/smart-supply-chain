Creating a comprehensive Python program for a project like "Smart Supply Chain" requires a combination of several components, such as data loading, predictive analytics, machine learning, and possibly an interactive interface. Below is a simplified version of what such a program may look like, using some Python libraries. This version will focus on data processing, predictive model creation, and basic error handling.

Please note, this example assumes you have a dataset to work with that contains relevant supply chain information. The focus is on predicting delivery times and optimizing logistics costs using a basic linear regression model from `scikit-learn`. You will need to adapt this template to fit your specific dataset and requirements.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import logging

# Initialize logging
logging.basicConfig(filename='supply_chain.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(file_path):
    """
    Load the dataset from a file
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as fnf_error:
        logging.error(f"Error loading file: {fnf_error}")
    except pd.errors.EmptyDataError as ede:
        logging.error(f"Loaded file is empty: {ede}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def preprocess_data(data):
    """
    Preprocess the data, handling missing values, encoding categorical features, etc.
    """
    try:
        # Example: Encoding categorical data
        data = pd.get_dummies(data, drop_first=True)
        # Example: Handling missing values
        data.fillna(data.mean(), inplace=True)
        logging.info("Data preprocessing completed")
        return data
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")

def split_data(data, target_column):
    """
    Splits the data into training and testing sets
    """
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Successfully split data into train and test sets")
        return X_train, X_test, y_train, y_test
    except KeyError as ke:
        logging.error(f"Column not found: {ke}")
    except Exception as e:
        logging.error(f"Failed to split data: {e}")

def build_model(X_train, y_train):
    """
    Build and train a predictive model
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logging.info("Model training completed successfully")
        return model
    except Exception as e:
        logging.error(f"Model building failed: {e}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with testing data
    """
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model evaluation completed, MSE: {mse}")
        return mse
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")

def main():
    # Replace 'supply_chain_data.csv' with the path to your dataset
    data_file = 'supply_chain_data.csv'
    try:
        data = load_data(data_file)
        if data is not None:
            processed_data = preprocess_data(data)
            X_train, X_test, y_train, y_test = split_data(processed_data, 'target_column_name')
            model = build_model(X_train, y_train)
            mse = evaluate_model(model, X_test, y_test)
            print(f"Model Mean Squared Error: {mse}")
    except Exception as e:
        logging.error(f"An error occurred in the main program: {e}")

if __name__ == "__main__":
    main()
```

**Understanding the Code:**

- **Logging and Error Handling:** The program uses the `logging` module to log events during execution. This is crucial for debugging and understanding the program flow. Error handling is employed to catch and log exceptions.

- **Data Handling:** The code attempts to load data from a CSV file. Ensure you have the correct path and permissions for the file.

- **Data Preprocessing:** Data is preprocessed through methods like encoding and filling missing values. Adjust this section based on the characteristics of your dataset.

- **Model Creation and Evaluation:** A simple linear regression model is created and evaluated with mean squared error. This can be replaced with a more sophisticated model depending on your needs.

This is a basic version, and a real-world application would likely require more complex modeling, possibly including machine learning libraries like TensorFlow or PyTorch for deeper analytics, as well as a more robust data handling mechanism. For production systems, consider deploying with a web framework like Django or Flask for an interactive user interface.