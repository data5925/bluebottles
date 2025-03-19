# bluebottles
large_file_rename.csv is too large to be added

* Train LSTM model on merged_randwick (a model for all 3 beaches)
* Train LSTM model on merged_randwick (a model for Maroubra beach only)
* Then test this second model on beach_surveys (which only has Maroubra) - potential issue: different feature variable (I think binary vs count?)


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
  
