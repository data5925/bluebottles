import pandas as pd

# Load datasets
randwick = pd.read_csv('randwick.csv', parse_dates=['date'], dayfirst=True)
beach_surveys = pd.read_csv('beach_surveys.csv', parse_dates=['date'], dayfirst=True)
wndwavecrt = pd.read_csv('wndwavecrt.csv', parse_dates=['time']) 

# Rename columns in randwick and beach surveys to match wndwavecrt, and change repeated column name 'time' to 'hour' for beach surveys
randwick.rename(columns={'date': 'time', 'lat.x': 'beach_lat', 'lon.x': 'beach_lon'}, inplace=True)
beach_surveys.rename(columns={'date': 'time', 'time': 'hour', 'lat.x': 'beach_lat', 'lon.x': 'beach_lon'}, inplace=True)

# Ensure 'time' column is in datetime format across all datasets
randwick['time'] = pd.to_datetime(randwick['time'])
beach_surveys['time'] = pd.to_datetime(beach_surveys['time'])
wndwavecrt['time'] = pd.to_datetime(wndwavecrt['time'])

# Merge randwick with wndwavecrt using left join
merged_randwick = pd.merge(randwick, wndwavecrt, on=['time', 'beach_lat', 'beach_lon'], how='left')
merged_randwick.to_csv('merged_randwick.csv', index=False)

# Merge beach surveys with wndwavecrt using left join
merged_beach_surveys = pd.merge(beach_surveys, wndwavecrt, on=['time', 'beach_lat', 'beach_lon'], how='left')
merged_beach_surveys.to_csv('merged_beach_surveys.csv', index=False)

# Display the first few rows and check data looks ok
print(merged_randwick.head())
print(merged_beach_surveys.head())