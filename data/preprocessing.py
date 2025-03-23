import numpy as np  
import pandas as pd
import os
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

df['wnd_dir'] = (np.degrees(np.arctan2(df['wnd_uas'], df['wnd_vas'])) + 360) % 360
if 'wave_sin_dir' in df.columns:
    df = df.drop(columns=['wave_sin_dir'])


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
columns_to_drop = ['source','beach', 'beach_key', 'surf_club', 'slsa_branch', 'state', 
                   'beach_lat', 'beach_lon', 'length', 'orientation', 'embaymentisation', 
                   'crt_closest_lat', 'crt_closest_lon', 'wnd_closest_lat', 'wnd_closest_lon', 'wave_closest_lat', 'wave_closest_lon']
df.drop(columns=columns_to_drop, inplace=True)

#Step 2: Distribution and trend
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

num_vars = [
    'presence', 'bluebottles', 
    'crt_temp', 'wave_hs', 
    'wave_cge', 'wnd_sfcWindspeed'
]

#2.1 Histogram/ Boxplots for numerical variables
for var in num_vars:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[var], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()
    
    if var not in ['presence', 'bluebottles']:  
        plt.figure(figsize=(10, 3))
        sns.boxplot(x=df[var], color='lightgreen')
        plt.title(f'Boxplot of {var}', fontsize=14)
        plt.xlabel(var, fontsize=12)
        plt.show()

#2.2 Visualization of categorical variables
#2.2.1 Cross-tabulate beach and bluebottle levels  
cross_tab = pd.crosstab(df['beach.x'], df['bluebottles'])  
# Map labels  
cross_tab.index = ['Maroubra', 'Coogee', 'Clovelly']  
cross_tab.columns = ['None', 'Likely', 'Some', 'Many']  
# Plot stacked bar chart  
plt.figure(figsize=(10, 6))  
cross_tab.plot(kind='bar', stacked=True, colormap='Pastel2')  
plt.title('Bluebottle Levels by Beach', fontsize=14)  
plt.xlabel('Beach', fontsize=12)  
plt.ylabel('Frequency', fontsize=12)  
plt.legend(title='Bluebottle Level', bbox_to_anchor=(1.05, 1))  
plt.xticks(rotation=0)  
plt.show()  

#2.2.2 Heapmap: Show the cross frequency between surf_club and state
contingency = pd.crosstab(df['surf_club'], df['state'])
plt.figure(figsize=(8, 6))
plt.imshow(contingency, cmap='viridis', aspect='auto')
plt.title("Heatmap: surf_club vs state")
plt.xlabel("state")
plt.ylabel("surf_club")
plt.colorbar(label='Count')
plt.xticks(ticks=np.arange(len(contingency.columns)), labels=contingency.columns)
plt.yticks(ticks=np.arange(len(contingency.index)), labels=contingency.index)
for i in range(len(contingency.index)):
    for j in range(len(contingency.columns)):
         plt.text(j, i, contingency.iloc[i, j], ha="center", va="center", color="white")
plt.tight_layout()
plt.show()
