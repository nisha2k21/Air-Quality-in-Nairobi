# Air Quality Prediction in Nairobi

## Overview

This project focuses on analyzing air quality data from Nairobi and predicting PM 2.5 levels using time series models. Leveraging data from the openAfrica platform, the project aims to assist in environmental monitoring and provide insights for better air quality management. The project utilizes machine learning models to forecast future air quality trends, helping local authorities and environmental agencies make data-driven decisions.

## Project Highlights
- **Objective**: Predict PM 2.5 air quality levels in Nairobi using historical data from the openAfrica platform.
- **Data Source**: Data collected from openAfrica, including air quality readings from Nairobi, Lagos, and Dar es Salaam.
- **Tools Used**: Python, MongoDB, Pandas, scikit-learn, matplotlib, and statsmodels.
- **Techniques**: Time series analysis, autoregressive models (AR), data wrangling, and hyperparameter tuning for model optimization.

## Workflow
1. **Data Collection**: 
   - Extracted air quality data (PM 2.5) from openAfrica, focusing on the cities of Nairobi, Lagos, and Dar es Salaam.
   - Queried MongoDB to obtain the relevant data for time series forecasting.

2. **Data Preprocessing**:
   - Cleaned and processed the data using Pandas, addressing missing values and outliers.
   - Aggregated the data by daily averages and performed exploratory data analysis (EDA) to understand underlying trends and seasonality.
   
3. **Modeling**:
   - Built an **Autoregressive (AR)** model to predict future PM 2.5 levels.
   - Fine-tuned the model using grid search and hyperparameter optimization to improve accuracy.
   - Evaluated model performance using metrics like Root Mean Squared Error (RMSE).

4. **Prediction**:
   - Generated short-term forecasts for PM 2.5 levels in Nairobi.
   - Visualized the predicted air quality levels, highlighting seasonal patterns and possible future risks.

## Results
- The AR model successfully captured trends in PM 2.5 levels, providing short-term predictions with good accuracy.
- The time series model identified seasonal fluctuations in air quality, providing valuable insights for environmental planning and response strategies.

## Key Learnings
- **MongoDB Querying**: Mastered querying a NoSQL database for time series data extraction.
- **Time Series Modeling**: Gained experience in building and optimizing autoregressive models for forecasting.
- **Data Wrangling**: Strengthened data cleaning and preprocessing skills, particularly with real-world, time-based datasets.
  
## Future Improvements
- **Incorporate More Data**: Integrate additional data sources, such as meteorological factors (temperature, humidity, wind speed), to improve prediction accuracy.
- **Advanced Models**: Experiment with advanced forecasting models like SARIMA and LSTM for longer-term predictions.
- **Deployment**: Deploy the model as a web application, providing real-time air quality forecasts to users.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/nisha2k21/air-quality-nairobi.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd air-quality-nairobi
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the model to generate predictions:
   ```bash
   python src/air_quality_forecast.py
   ```

## Project Structure
```
├── data
│   └── air_quality_data.csv       # Raw data file
├── notebooks
│   └── EDA_and_Modeling.ipynb     # Jupyter notebook for analysis and modeling
├── src
│   └── air_quality_forecast.py    # Python script for running the model
├── requirements.txt               # List of required dependencies
└── README.md                      # Project documentation
```

## Technologies Used
- **Programming Language**: Python
- **Database**: MongoDB
- **Libraries**: Pandas, NumPy, scikit-learn, statsmodels
- **Visualization**: matplotlib, seaborn
- **Forecasting Techniques**: Autoregressive (AR) modeling, hyperparameter tuning

## Contact
For any questions, collaboration opportunities, or feedback, feel free to reach out:
- **Email**: nisha2k21@gmail.com
- **LinkedIn**: [Nisha Kumari](https://www.linkedin.com/in/nisha-kumari-041300225/)
