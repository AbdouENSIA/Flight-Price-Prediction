# ✈️ Flight Price Prediction Jupyter Notebook 🚀

Welcome aboard! Ready to predict flight prices and beat the odds? This project takes you on a journey through data and machine learning, where we’ll use a Random Forest Regressor to crack the code behind flight costs. Buckle up, because it's going to be a smooth ride! 😎🛫

## 📋 Table of Contents
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

## 🌟 Overview
Ever wondered why flight prices can change faster than you can blink? 🤔 This project dives deep into predicting those prices based on features like airline, route details, and booking timeframes. By harnessing the power of machine learning, we aim to demystify the flight pricing puzzle and help you find the best deals (or just understand why you paid so much)! 💸

## 📊 Dataset
The analysis is fueled by the `flights.csv` dataset, packed with key features that affect flight prices:

| Feature          | Description                       |
|------------------|-----------------------------------|
| `airline`        | ✈️ Operating airline company      |
| `flight`         | Flight number                     |
| `source_city`    | 🏙️ Departure city                 |
| `departure_time` | 🕒 Time of departure               |
| `stops`          | 🚏 Number of stops                |
| `arrival_time`   | 🕕 Time of arrival                 |
| `destination_city`| 🛬 Arrival city                   |
| `class`          | 🪑 Travel class (e.g., Economy)   |
| `duration`       | ⏱️ Flight duration in hours       |
| `days_left`      | 📅 Days remaining until the flight |
| `price`          | 🎯 Target variable: Flight price  |

## 🛠️ Project Structure
Everything happens inside the magical `flight_prices.ipynb` notebook! ✨ Here’s what you’ll find inside:

1. **Data Preparation 🧹**
   - Cleaning and preprocessing the data
   - Feature engineering and encoding

2. **Model Development 🧑‍💻**
   - Training a Random Forest Regressor
   - Model evaluation and tuning

3. **Evaluation 📉**
   - How well does our model predict prices? Find out through metrics like R², MAE, MSE, and RMSE.

4. **Feature Analysis 🔍**
   - Discover the key players that influence flight prices.

5. **Results Visualization 📊**
   - Enjoy some eye-catching visuals that compare actual vs. predicted prices!

## ⚙️ Installation
Getting started is as easy as 1-2-3! Follow these steps:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/AbdouENSIA/Flight-Price-Prediction.git
   cd flight-price-prediction
   

2. Install Jupyter and required packages:
   ```bash
   pip install jupyter pandas numpy scikit-learn matplotlib seaborn
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open the `flight_prices.ipynb` notebook in your browser.

## 🚀 Usage
Using the model is as smooth as a first-class flight experience:

1. **Ensure that the `flights.csv` file is in the same directory as the notebook.**
2. **Open the `flight_prices.ipynb` notebook in Jupyter.**
3. **Run each cell in the notebook sequentially to perform data analysis, model training, and result visualization.** Sit back and watch the magic unfold! 🎩✨

## 🔧 Data Preprocessing: Getting the Data Ready for Takeoff! ✈️🚀

Before we let our model soar, we need to work some data magic! 🎩✨ Data preprocessing is like packing your bags for a trip—essential to ensure a smooth journey. Here’s how we prep our data to be machine learning-ready:

1. **Dropping Useless Columns 🗑️:**
   We kick out columns that don’t add value to our prediction game. These redundant or noisy features are like extra baggage—they slow us down! By cleaning up, we make sure our model stays light and focused on what matters. 🎯

2. **Encoding Categorical Variables 🎨:**
   Categorical data is like a secret language our model needs to crack, so let’s decode it like pros:

   - **Binary Encoding 🔄:** For variables with just two categories (like Yes/No, Male/Female), we use binary encoding. It’s a simple 0 or 1 that tells our model exactly what’s up—no need for extra drama. 😉
   
   - **One-Hot Encoding 🌶️:** Got more than two categories? Time for a spicy one-hot encoding! We create new columns for each category, turning text into numbers without making them seem ordered. It’s like giving each option its own seat on the plane! 🪑
   
   - **pd.factorize() 🤖:** For some special categories, we let pandas work its magic with `factorize()`. It gives each unique value a number, keeping the data structure intact and reducing the need for expanding columns all over the place. Perfect for when we want to maintain the original feel of the data. 🧩

### Why These Steps Matter 🚦:
- **Reduce Noise 🌐:** Trim down the fluff by dropping unnecessary columns, making our data leaner and meaner.
- **Speak Machine Language 🤓:** Turn those confusing categorical variables into friendly numerical formats so the model doesn’t get lost in translation.
- **Keep the Context 📚:** Factorize when necessary to preserve the original meaning of categories without overcomplicating things.
- **Ready for Random Forest 🌲:** All these steps ensure our data’s primed and prepped, giving the Random Forest Regressor exactly what it needs to predict like a pro. 🏆

After these magical transformations, our dataset is like a well-packed suitcase—organized, efficient, and ready for the Random Forest adventure ahead! 🎢🌟

## 🧠 Model Training
We train our model using the Random Forest Regressor—a beast at handling complex data:

1. Splitting the data into training and testing sets
2. Initializing the Random Forest Regressor with predefined hyperparameters
3. Fitting the model on the training data
4. Making predictions on the test data

## 📈 Model Performance
Here’s how our model’s flying:

- **R² Score:** 🚀 Approximately 0.95
- **MAE:** 🪙 Around 1250
- **MSE:** 💥 3,200,000
- **RMSE:** ⚡ 1,800

It’s not just about predicting—it’s about predicting with style! ✨

## 📊 Visualizations
Data comes alive through visuals! We’ve got:

- **Predicted vs Actual Prices:** See how well our model guesses the ticket price!

These visualizations offer insights into which factors most strongly influence flight prices and how well our model is performing across different price ranges.

## 🚀 Future Improvements
We’re already soaring high, but there’s always room to go supersonic:

1. Feature Engineering: Create more complex features that might capture additional patterns in the data.
2. Hyperparameter Tuning: Use techniques like Grid Search or Random Search to find optimal hyperparameters for the Random Forest model.
3. Ensemble Methods: Experiment with combining multiple models to potentially improve prediction accuracy.
4. Deep Learning Approaches: Explore whether neural networks could capture more complex patterns in the data.
5. Time Series Analysis: Incorporate time-based features to capture seasonal trends in flight prices.

We encourage contributors to explore these areas and submit their findings and improvements.

## 🤝 Contributing
Got ideas? Improvements? Wanna make this model fly even higher? Fork the repo, make your changes, and send a pull request. Let’s build something amazing together! 🚀

## 📜 License
This project is licensed under the MIT License - feel free to use, modify, and share. Let’s keep the code free and accessible! 🤗
