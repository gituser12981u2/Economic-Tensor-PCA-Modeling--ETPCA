# Economic Tensor PCA Modeling (ETPCA)

## Overview
Economic Tensor PCA Modeling (ETPCA) is an analysis tool designed for economic data analysis. It employs a combination of tensor product, Principal Component Analysis (PCA), and linear regression to predict Gross National Product (GNP) values, providing a nuanced understanding of economic trends. 

Note: the current implementation uses CPI and DJIA, but any known metrics of the time can be substituted in at the users will, though the outcome will change and cannot promised to be as correct. Data that represents the "economy" at large will often return best results. 

## Methodology

### Data Collection
- Monthly data for DJIA (Dow Jones Industrial Average) and CPI (Consumer Price Index).
- Yearly GNP values.

### Preprocessing
- Conversion of dates to a numerical format for regression analysis.
- Normalization of data to ensure uniformity in scale.

### Linear Model
- Initial linear interpolation between the beginning and ending GNP values.
- Calculation of the slope for monthly GNP estimation.

### Tensor Product
- Creation of a multidimensional array (tensor) from DJIA and CPI vectors.
- Captures interaction effects between these two economic indicators.

### Principal Component Analysis (PCA)
- Orthogonal transformation of the tensor product.
- Extraction of primary components, reducing dimensionality while retaining key data variations.

### Linear Regression Model
- Mapping PCA data back to a readable GNP scale.
- Training a regression model with PCA components to predict monthly GNP values.

## Economic Interpretation
1. **Tensor Product**: Represents the interaction effect between CPI and DJIA, reflecting the "economy" in relation to GNP.
2. **PCA**: Extracts underlying trends and patterns from the tensor data.
3. **Regression Model**: Translates these trends into predicted monthly GNP values.

## Running the Application
1. Clone the repository:
   ```bash
   git clone https://github.com/gituser12981u2/Economic-Tensor-PCA-Modeling--ETPCA-.git

2. Navigate to the 'src' directory:
   ```
   cd Economic-Tensor-PCA-Modeling--ETPCA-/src

3. Run the main script:
   ```
   python main.py

## Contributing
Contributions to the ETPCA project are welcome. Feel free to fork the repository, make your changes, and submit a pull request.

## License
This project is available under the GPL license
