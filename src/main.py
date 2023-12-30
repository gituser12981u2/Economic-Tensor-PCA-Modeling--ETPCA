# src/main.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from tensor_model import process_tensors

# DJIA Stock Price Data
djia_data = {
    'Date': [
        '1929-08-01', '1929-09-03', '1929-10-01', '1929-11-04', '1929-12-02',
        '1930-01-02', '1930-02-01', '1930-03-01', '1930-04-01', '1930-05-01',
        '1930-06-02', '1930-07-01', '1930-08-01', '1930-09-02', '1930-10-01',
        '1930-11-01', '1930-12-01', '1931-01-02', '1931-02-02', '1931-03-02',
        '1931-04-01', '1931-05-01', '1931-06-01', '1931-07-01', '1931-08-01',
        '1931-09-01', '1931-10-01', '1931-11-02', '1931-12-01', '1932-01-02',
        '1932-02-01', '1932-03-01', '1932-04-01', '1932-05-02', '1932-06-01',
        '1932-07-01', '1932-08-01', '1932-09-01', '1932-10-01', '1932-11-01',
        '1932-12-01'
    ],
    'DJIA': [
        350.56, 381.17, 342.57, 257.68, 241.7, 244.2, 268.41, 273.24, 287.11,
        274.59, 274.45, 223.03, 233.57, 240.42, 214.14, 184.89, 185.48,
        169.84, 168.71, 184.38, 170.82, 145.58, 122.77, 152.66, 136.65,
        140.13, 95.66, 104.5, 91.17, 74.62, 79.63, 81.87, 72.18, 55.37,
        44.93, 44.39, 54.94, 73.67, 72.09, 60.22, 58.02
    ]
}

# CPI Data
cpi_data = {
    'Date': [
        '1929-01-01', '1929-02-01', '1929-03-01', '1929-04-01', '1929-05-01',
        '1929-06-01', '1929-07-01', '1929-08-01', '1929-09-01', '1929-10-01',
        '1929-11-01', '1929-12-01', '1930-01-01', '1930-02-01', '1930-03-01',
        '1930-04-01', '1930-05-01', '1930-06-01', '1930-07-01', '1930-08-01',
        '1930-09-01', '1930-10-01', '1930-11-01', '1930-12-01', '1931-01-01',
        '1931-02-01', '1931-03-01', '1931-04-01', '1931-05-01', '1931-06-01',
        '1931-07-01', '1931-08-01', '1931-09-01', '1931-10-01', '1931-11-01',
        '1931-12-01', '1932-01-01', '1932-02-01', '1932-03-01', '1932-04-01',
        '1932-05-01', '1932-06-01', '1932-07-01', '1932-08-01', '1932-09-01',
        '1932-10-01', '1932-11-01', '1932-12-01'
    ],
    'CPI': [
        17.1, 17.1, 17.0, 16.9, 17.0, 17.1, 17.3, 17.3, 17.3, 17.3, 17.3, 17.2,
        17.1, 17.0, 16.9, 17.0, 16.9, 16.8, 16.6, 16.5, 16.6, 16.5, 16.4, 16.1,
        15.9, 15.7, 15.6, 15.5, 15.3, 15.1, 15.1, 15.1, 15.0, 14.9, 14.7, 14.6,
        14.3, 14.1, 14.0, 13.9, 13.7, 13.6, 13.6, 13.5, 13.4, 13.3, 13.2, 13.1
    ]
}

# GNP Yearly Data
gnp_data = {
    'Year': [1929, 1930, 1931, 1932],
    'GNP': [709.6, 657.3, 590.7, 537.3]
}


# Convert to DataFrame
djia_df = pd.DataFrame(djia_data)
djia_df['Date'] = pd.to_datetime(djia_df['Date'])

cpi_df = pd.DataFrame(cpi_data)
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])

gnp_df = pd.DataFrame(gnp_data)

# Interpolating Monthly GNP
date_range = pd.date_range(start='1929-01-01', end='1932-12-01', freq='MS')
monthly_gnp = pd.DataFrame({'Date': date_range})

# Convert 'Date' to numerical values for regression
monthly_gnp['Year_Num'] = monthly_gnp['Date'].dt.year + \
    (monthly_gnp['Date'].dt.month - 1) / 12

# Fit the linear model to yearly GNP data
gnp_df['Year_Num'] = gnp_df['Year']
model = LinearRegression()
model.fit(gnp_df[['Year_Num']], gnp_df['GNP'])

# Predict monthly GNP
monthly_gnp['GNP_Predicted'] = model.predict(monthly_gnp[['Year_Num']])

# # Exclude December from PCA prediction
# exclude_months = [12]  # December
# mask = ~monthly_gnp['Date'].dt.month.isin(exclude_months)
# monthly_gnp_for_pca = monthly_gnp[mask]

# Pass the monthly GNP along with CPI and DJIA to the tensor model
tensor_result = process_tensors(djia_df, cpi_df, monthly_gnp)

# print(tensor_result)

# Fit PCA regression model
pca_regression_model = LinearRegression()
pca_regression_model.fit(tensor_result, monthly_gnp['GNP_Predicted'])

# Predict GNP for all months using the PCA model
# all_tensor_result = process_tensors(djia_df, cpi_df, monthly_gnp)
gnp_predicted_from_pca = pca_regression_model.predict(tensor_result)

# Export to CSV and print table
monthly_gnp.to_csv('predicted_monthly_gnp.csv', index=False)
print(monthly_gnp)

# # Replace December predictions with actual yearly GNP
# for year in gnp_df['Year']:
#     dec_index = monthly_gnp[(monthly_gnp['Date'].dt.year == year) & (
#         monthly_gnp['Date'].dt.month == 12)].index[0]
#     gnp_predicted_from_pca[dec_index] = gnp_df[gnp_df['Year']
#                                                == year]['GNP'].values[0]

# Plotting the linear monthly GNP
plt.figure(figsize=(12, 6))
plt.plot(monthly_gnp['Date'], monthly_gnp['GNP_Predicted'],
         label='Interpolated Monthly GNP')
plt.xlabel('Date')
plt.ylabel('GNP (Billions of Dollars)')
plt.title('Interpolated Monthly GNP (1929-1932)')
plt.legend()
plt.show()

# Plotting the predicted results
plt.figure(figsize=(12, 6))
plt.plot(monthly_gnp['Date'], monthly_gnp['GNP_Predicted'],
         label='Original Interpolated GNP')
plt.plot(monthly_gnp['Date'], gnp_predicted_from_pca,
         label='PCA Predicted GNP', linestyle='--')
plt.xlabel('Date')
plt.ylabel('GNP (Billions of Dollars)')
plt.title('Comparison of Interpolated GNP and PCA Predicted GNP')
plt.legend()
plt.show()

# # Plotting the results
# plt.figure(figsize=(12, 6))
# plt.plot(monthly_gnp['Date'], monthly_gnp['GNP_Predicted'],
#          label='Original Interpolated GNP')
# plt.plot(monthly_gnp['Date'], gnp_predicted_from_pca,
#          label='PCA Predicted GNP', linestyle='--')
# plt.xlabel('Date')
# plt.ylabel('GNP (Billions of Dollars)')
# plt.title('Comparison of Interpolated GNP and PCA Predicted GNP')
# plt.legend()
# plt.show()
