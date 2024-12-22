# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import logging
import time

# Step 2: Set up logging
logging.basicConfig(filename='governance_analysis.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to handle retries for file operations (e.g., loading data)
def retry_operation(func, retries=3, delay=2, *args, **kwargs):
    """Retry a function on failure with exponential backoff."""
    attempt = 0
    while attempt < retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            logging.error(f"Attempt {attempt} failed with error: {e}")
            if attempt == retries:
                raise e
            time.sleep(delay * attempt)  # Exponential backoff

# Step 3: Load the governance data (Replace with your actual CSV path)
try:
    data = retry_operation(pd.read_csv, retries=3, delay=2, filepath_or_buffer='governance_data.csv')
    logging.info("Data loaded successfully.")
except FileNotFoundError:
    logging.error("The file 'governance_data.csv' was not found. Please check the file path.")
    raise
except pd.errors.EmptyDataError:
    logging.error("The file 'governance_data.csv' is empty. Please check the content.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred while loading the data: {e}")
    raise

# Display the first few rows of data
logging.info(f"Data Overview: {data.head()}")

# Step 4: Data Analysis (Quick summary)
try:
    logging.info("\nData Summary:")
    logging.info(f"{data.describe()}")
except Exception as e:
    logging.error(f"Error during data analysis: {e}")
    raise

# Step 5: Network Analysis - Create a collaboration network
try:
    G = nx.Graph()

    # Add nodes for each jurisdiction with attributes
    for index, row in data.iterrows():
        G.add_node(row['Jurisdiction'], model=row['Governance Model'], region=row['Region'])

    # Add edges based on collaboration score (threshold example: score > 80 connects jurisdictions)
    for i, row_i in data.iterrows():
        for j, row_j in data.iterrows():
            if i != j and abs(row_i['Collaboration Score'] - row_j['Collaboration Score']) < 20:
                G.add_edge(row_i['Jurisdiction'], row_j['Jurisdiction'], weight=abs(row_i['Collaboration Score'] - row_j['Collaboration Score']))

    # Plot the collaboration network
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G)  # Positioning for better layout
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title('Collaboration Network between Jurisdictions')
    plt.show()

    logging.info("Network analysis completed successfully.")

except Exception as e:
    logging.error(f"Error during network analysis: {e}")
    raise

# Step 6: Geospatial Visualization (Optional - Assuming you have a shapefile for jurisdictions)
try:
    gdf = retry_operation(gpd.read_file, retries=3, delay=2, filename='jurisdiction_boundaries.shp')
    merged = gdf.merge(data, left_on='JurisdictionName', right_on='Jurisdiction')

    # Plot the map with collaboration scores
    merged.plot(column='Collaboration Score', cmap='YlGn', legend=True, figsize=(12, 10))
    plt.title('Geospatial Distribution of Collaboration Scores')
    plt.show()

    logging.info("Geospatial visualization completed successfully.")

except FileNotFoundError:
    logging.error("The shapefile 'jurisdiction_boundaries.shp' was not found. Please check the file path.")
    raise
except gpd.errors.FionaValueError:
    logging.error("Unable to read the shapefile. Check the format or integrity of the shapefile.")
    raise
except KeyError as e:
    logging.error(f"Missing expected column in the shapefile or data: {e}")
    raise
except Exception as e:
    logging.error(f"Error during geospatial analysis: {e}")
    raise

# Step 7: Model Analysis using Linear Regression (Predict collaboration score based on governance model)
try:
    label_encoder = LabelEncoder()
    data['Governance Model Encoded'] = label_encoder.fit_transform(data['Governance Model'])

    # Prepare features (X) and target (y)
    X = data[['Governance Model Encoded']]  # Feature: Encoded Governance Model
    y = data['Collaboration Score']         # Target: Collaboration Score

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions using the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Mean Squared Error (MSE) of the model: {mse}")

except KeyError as e:
    logging.error(f"Missing expected column in the data: {e}")
    raise
except ValueError as e:
    logging.error(f"Issue with the data format or model fitting: {e}")
    raise
except Exception as e:
    logging.error(f"Error during model analysis: {e}")
    raise

# Step 8: Exporting the Results
try:
    # Export the data with predicted collaboration scores to CSV
    predicted_data = X_test.copy()
    predicted_data['Actual Collaboration Score'] = y_test
    predicted_data['Predicted Collaboration Score'] = y_pred
    predicted_data.to_csv('predicted_collaboration_scores.csv', index=False)
    logging.info("Predicted collaboration scores have been exported to 'predicted_collaboration_scores.csv'.")

    # Optionally, export the network graph to a file (GraphML format)
    nx.write_graphml(G, "collaboration_network.graphml")
    logging.info("Collaboration network has been exported to 'collaboration_network.graphml'.")

except Exception as e:
    logging.error(f"Error during data export: {e}")
    raise

# Final Message
logging.info("Analysis complete! Results exported successfully.")
