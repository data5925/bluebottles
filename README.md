# bluebottles
WHAT TO ADD/TRY FOR MODELLING:

* CROSS-VALIDATION (TimeSeriesSplit)
* Train LSTM model on merged_randwick (a model for Maroubra beach only), then test this second model on beach_surveys (which only has Maroubra)
* Try different set of features
* Plot result on map (geographical map)


# Preprocess
Merge by inner
1. Data type
2. Duplicates
3. Outliers

# EDA
**1. Descriptive**
- Summary statistics for numerical and categorical variables
- Distribution of key variables
  
**2. Distribution & Trends**
- Histogram/boxplots... for numerical variables.
- Trends over time (occurrences, wind speeds, wave heights).
- Seasonality in bluebottle occurrences.
  
**3. Correlation**
- Correlation matrix
- Pairwise scatter plots for highly correlated variables.
  
**4. Geospatial**
- Distribution of observations across beaches (latitude & longitude map?).
  
