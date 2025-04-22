# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:
To create and implement Holt Winter's Method Model using python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
df = pd.read_csv('ECOMM DATA.csv')

# Convert the 'Order Date' column to datetime format and set it as the index
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)

# Convert 'Profit' column to numeric (removing invalid values)
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')

# Drop rows with missing values in 'Profit' column
clean_data = df.dropna(subset=['Profit'])

# Extract 'Profit' column for time series forecasting
profit_data_clean = clean_data['Profit']

# Optional: Check for frequency consistency (e.g., daily, weekly, etc.)
# profit_data_clean = profit_data_clean.asfreq('B')

# Define model parameters and perform Holt-Winters exponential smoothing
# Adjust seasonal_periods to match the data's seasonality
seasonal_periods = 365  # Example: annual seasonality if data is daily for multiple years

model = ExponentialSmoothing(profit_data_clean, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
fit = model.fit()

# Forecast for the next 30 steps (business days)
n_steps = 30
forecast = fit.forecast(steps=n_steps)

# Create a date range for the forecast (business days)
forecast_index = pd.date_range(start=profit_data_clean.index[-1], periods=n_steps+1, freq='B')[1:]

# Plot the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(profit_data_clean.index, profit_data_clean, label='Original Profit Data', color='blue')
plt.plot(forecast_index, forecast, label='Forecasted Profit', color='orange')
plt.xlabel('Date')
plt.ylabel('Profit')
plt.title('Holt-Winters Forecast for E-Commerce Profit Data')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:


![image](https://github.com/user-attachments/assets/b6f6d0ef-10c4-4a41-be70-d0f698c4a05e)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
