import numpy as np  
import pandas as pd
df = pd.read_csv('merged_randwick.csv')

df['time'] = pd.to_datetime(df['time'])
bluebottles_mapping = {'Likely': 1, 'Some': 2, 'Many': 3, np.nan: 0}
df['bluebottles'] = df['bluebottles'].map(bluebottles_mapping)
beach_mapping = {
    'Maroubra Beach (North)': 0,
    'Coogee Beach': 1,
    'Clovelly Beach': 2
}
df['beach.x'] = df['beach.x'].map(beach_mapping)

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
excluded_columns = ['beach.x','presence', 'bluebottles', 'beach_lon', 'length', 'orientation', 'embaymentisation']
columns_to_clean = [col for col in numerical_df.columns if col not in excluded_columns]
outliers_mask = (df[columns_to_clean] < lower_bound[columns_to_clean]) | (df[columns_to_clean] > upper_bound[columns_to_clean])
df = df[~outliers_mask.any(axis=1)].reset_index(drop=True)

# No duplicate_rows, perfect!
duplicate_rows = df[df.duplicated()]
#print(f"Number of duplicate rows: {len(duplicate_rows)}")

#EDA
#Summary statistics for numerical and categorical variables
numerical_summary = df.describe().T 

categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_summary = pd.DataFrame({
    "Unique Values": df[categorical_cols].nunique(),  
    "Most Frequent": df[categorical_cols].mode().iloc[0],  
    "Frequency": df[categorical_cols].apply(lambda x: x.value_counts().iloc[0])  
})

print(" **Numerical Summary:**")
print(numerical_summary)

print("\n **Categorical Summary:**")
print(categorical_summary)

# Step 1: Descriptive Analysis
# 1. Numerical Variables:
#    - presence: Binary variable (0/1), jellyfish presence only 6.56% of the time
#    - bluebottles: Categories (0=None, 1=Likely, 2=Some, 3=Many), mostly 0
#    - crt_temp: Avg 20.72°C, ranging from 14.93°C to 25.21°C
#    - wave_hs: Avg 1.38m, Min 0.40m, Max 2.56m
#    - wave_cge: Avg 8.11, Max 24.18, highly variable
#    - wnd_sfcWindspeed: Avg 5.42 m/s, Min 1.85 m/s, Max 11.3 m/s
#
# 2. Categorical Variables:
#    - beach: 3 locations (Coogee, Maroubra, Bondi), mostly Coogee Beach (1307)
#    - surf_club: 3 clubs, mainly Coogee SLSC (1307 records)
#    - slsa_branch & state: Only "Sydney Inc." & "NSW"
#    - source: All data from "BeachWatch"

#drop unnecessary col
columns_to_drop = ['time','source','beach', 'beach_key', 'surf_club', 'slsa_branch', 'state', 
                   'beach_lat', 'beach_lon', 'length', 'orientation', 'embaymentisation']
df.drop(columns=columns_to_drop, inplace=True)
