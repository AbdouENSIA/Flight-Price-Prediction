# Flight Price Prediction

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Performance](#model-performance)
7. [Contributing](#contributing)
8. [License](#license)

## Overview

This project implements a machine learning solution for predicting flight prices. By leveraging a Random Forest Regressor and a comprehensive dataset of flight information, we aim to provide accurate price forecasts based on various features such as airline, route details, and booking timeframes.

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

1. **Data Preparation**
   - Data cleaning and preprocessing
   - Feature engineering
   - Categorical encoding (e.g., one-hot encoding)

2. **Model Development**
   - Random Forest Regressor implementation
   - Hyperparameter tuning

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
   - Performance metrics plots
   - Predicted vs. Actual price comparisons

## Installation

To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the prediction model:

```bash
python src/main.py
```

For detailed usage instructions, refer to the documentation in the `docs/` directory.

## Model Performance

Current model performance metrics:

- R² Score: 0.95
- MAE: 1254.32
- MSE: 3,245,678.90
- RMSE: 1,801.58

## Contributing

We welcome contributions to improve the model's accuracy and efficiency. Please refer to `CONTRIBUTING.md` for guidelines on how to submit pull requests, report issues, or request features.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
