import numpy as np  
import pandas as pd
df = pd.read_csv('merged_randwick.csv')

df['time'] = pd.to_datetime(df['time'])
bluebottles_mapping = {'Likely': 1, 'Some': 2, 'Many': 3, np.nan: 0}
df['bluebottles'] = df['bluebottles'].map(bluebottles_mapping)

print("Initial Data Types:")
print(df.dtypes)


#check missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]  

#print("Missing values in each column:")
#print(missing_values)

# Check outliers
# Select only numerical columns
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Compute IQR for numerical columns
Q1 = numerical_df.quantile(0.25)
Q3 = numerical_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Here after checkinh the data, outliers in following variables won't be removed. For the rest variables,I directly remove them due to the small amount of outliers(<5%)
excluded_columns = ['presence', 'bluebottles', 'beach_lon', 'length', 'orientation', 'embaymentisation']
columns_to_clean = [col for col in numerical_df.columns if col not in excluded_columns]
outliers_mask = (df[columns_to_clean] < lower_bound[columns_to_clean]) | (df[columns_to_clean] > upper_bound[columns_to_clean])
df = df[~outliers_mask.any(axis=1)].reset_index(drop=True)

# No duplicate_rows,perfect!
duplicate_rows = df[df.duplicated()]
#print(f"Number of duplicate rows: {len(duplicate_rows)}")

