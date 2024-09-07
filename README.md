# Flight Price Prediction Jupyter Notebook

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Model Performance](#model-performance)
9. [Visualizations](#visualizations)
10. [Future Improvements](#future-improvements)
11. [Contributing](#contributing)
12. [License](#license)

## Overview
This project implements a machine learning solution for predicting flight prices using a Jupyter notebook. By leveraging a Random Forest Regressor and a comprehensive dataset of flight information, we aim to provide accurate price forecasts based on various features such as airline, route details, and booking timeframes.

## Dataset
The analysis is based on the `flights.csv` dataset, which includes the following key features:

| Feature | Description |
|---------|-------------|
| `airline` | Operating airline company |
| `flight` | Flight number |
| `source_city` | Departure city |
| `departure_time` | Time of departure |
| `stops` | Number of stops |
| `arrival_time` | Time of arrival |
| `destination_city` | Arrival city |
| `class` | Travel class (e.g., Economy, Business) |
| `duration` | Flight duration in hours |
| `days_left` | Days remaining until the flight |
| `price` | Target variable: Flight price |

## Project Structure
The project is contained in a single Jupyter notebook named `flight_prices.ipynb`, which includes:

1. **Data Preparation**
   - Data loading and inspection
   - Data cleaning and preprocessing
   - Feature engineering
   - Categorical encoding

2. **Model Development**
   - Random Forest Regressor implementation
   - Model training

3. **Evaluation**
   - Model performance assessment using:
     - R² Score
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

4. **Feature Analysis**
   - Identification of key price determinants
   - Feature importance visualization

5. **Results Visualization**
   - Performance metrics display
   - Predicted vs. Actual price comparison plot

## Installation
To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install Jupyter and required packages:
   ```bash
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open the `flight_prices.ipynb` notebook in your browser.

## Usage
To run the prediction model:

1. Ensure that the `flights.csv` file is in the same directory as the notebook.
2. Open the `flight_prices.ipynb` notebook in Jupyter.
3. Run each cell in the notebook sequentially to perform data analysis, model training, and result visualization.

## Data Preprocessing
The notebook includes several crucial data preprocessing steps to prepare the dataset for machine learning:

1. Dropping Useless Columns: We remove any columns that don't contribute to the prediction task or contain redundant information. This helps to reduce noise in the data and improve model performance.

2. Encoding Categorical Variables:
   a. Binary Encoding: For categorical variables with only two categories, we use binary encoding. This converts the category into a binary (0 or 1) representation.
   
   b. One-Hot Encoding: For categorical variables with more than two categories, we apply one-hot encoding. This creates new binary columns for each category, which helps the model understand categorical data without imposing an ordinal relationship.
   
   c. pd.factorize(): For some categorical variables, we use pandas' factorize() function. This method encodes categorical variables as integer codes and is particularly useful for ordinal categories or when you want to preserve the original structure of the data without expanding the feature space significantly.

These preprocessing steps are crucial because they:
- Reduce data dimensionality by removing unnecessary features
- Convert categorical data into a format that machine learning algorithms can work with
- Preserve important categorical information without introducing false ordinal relationships
- Prepare the data in a way that allows the Random Forest algorithm to make effective splits and predictions

After these preprocessing steps, our data is in a suitable format for training the Random Forest Regressor, ensuring that all features are numeric and properly encoded.

## Model Training
We use a Random Forest Regressor for this prediction task. The model is trained on a subset of the data (typically 80%) and then evaluated on the remaining data. The training process includes:

1. Splitting the data into training and testing sets
2. Initializing the Random Forest Regressor with predefined hyperparameters
3. Fitting the model on the training data
4. Making predictions on the test data

## Model Performance
Current model performance metrics (may vary with different runs):

- R² Score: Approximately 0.95
- MAE: Around 1250
- MSE: Approximately 3,200,000
- RMSE: Around 1,800

Actual performance may vary based on the specific dataset split and random initialization.

## Visualizations
The notebook provides several visualizations to help understand the data and model performance:

1. Predicted vs Actual Prices: A scatter plot comparing the model's predictions against the actual prices.

These visualizations offer insights into which factors most strongly influence flight prices and how well our model is performing across different price ranges.

## Future Improvements
While our current model performs well, there are several areas for potential improvement:

1. Feature Engineering: Create more complex features that might capture additional patterns in the data.
2. Hyperparameter Tuning: Use techniques like Grid Search or Random Search to find optimal hyperparameters for the Random Forest model.
3. Ensemble Methods: Experiment with combining multiple models to potentially improve prediction accuracy.
4. Deep Learning Approaches: Explore whether neural networks could capture more complex patterns in the data.
5. Time Series Analysis: Incorporate time-based features to capture seasonal trends in flight prices.

We encourage contributors to explore these areas and submit their findings and improvements.

## Contributing
We welcome contributions to improve the model's accuracy and efficiency. Please feel free to fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
