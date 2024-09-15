# Hybrid-Stock-Market-Analysis-Data-Driven-and-Sentiment-Based


## Overview
This project aims to create a hybrid model that predicts stock market prices using a combination of numerical analysis of historical stock prices and sentiment analysis of news headlines. The primary stock analyzed in this project is the SENSEX (S&P BSE SENSEX), a major stock market index in India. The model leverages both time-series forecasting techniques and natural language processing (NLP) to enhance predictive accuracy.

## Data Sources
- **Historical Stock Prices**: SENSEX data is downloaded from [Yahoo Finance](https://finance.yahoo.com/).
- **News Headlines**: News data is sourced from a large dataset available at [https://bit.ly/36fFPI], containing over 3 million Indian news headlines from 2001 to 2020.

## Methodology
1. **Data Collection**:
   - **Historical Stock Prices**: Downloaded and preprocessed using Python libraries such as Pandas.
   - **News Data**: Cleaned and prepared by removing unwanted characters, normalizing text, and creating sentiment scores for each news headline.

2. **Data Preprocessing**:
   - Converted the historical stock prices into a time-series format.
   - Grouped news headlines by date and calculated sentiment scores (polarity, subjectivity, compound score) using TextBlob and NLTK's VADER sentiment analyzer.
   - Merged historical stock prices with sentiment data to create a unified dataset for modeling.

3. **Feature Engineering**:
   - Created new features using moving averages, rolling statistics, and sentiment scores.
   - Scaled data using MinMaxScaler for compatibility with various machine learning algorithms.

4. **Model Training**:
   - Split the dataset into training and testing sets.
   - Trained multiple regression models, including RandomForestRegressor, AdaBoostRegressor, DecisionTreeRegressor, LightGBM, and XGBoost, to identify the best model based on Mean Squared Error (MSE).

5. **Evaluation**:
   - Compared model performance using MSE. The RandomForestRegressor model showed the best performance with an MSE of 0.0526.

## Results
- The hybrid model successfully integrates numerical and textual data to predict stock prices. The inclusion of sentiment analysis improved the prediction accuracy compared to using only numerical data.
- The model achieved a mean squared error (MSE) of 0.0526 with the RandomForestRegressor, outperforming other models.

## Tools and Libraries
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, NLTK, TextBlob, Statsmodels, Scikit-learn, XGBoost, LightGBM, Pandas DataReader

## How to Use
1. **Data Preparation**: Download the historical stock prices and news headlines data.
2. **Run the Jupyter Notebook**: Follow the steps in the notebook to preprocess the data, train models, and evaluate performance.
3. **Modify Parameters**: Users can select different stocks or datasets for analysis by modifying data paths and parameters in the notebook.

## Future Work
- Extend the model to include more stocks and global news sources.
- Experiment with deep learning models such as LSTM and Transformers for improved performance.
- Incorporate additional features, such as economic indicators, to enhance prediction accuracy.

## License
This project is licensed under the MIT License.

