# -*- coding: utf-8 -*-
"""Trend of Seawater & Occurences.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16nu92yicD_6GB-i_lvGfb2yRYvKXNX49
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")
df["time"] = pd.to_datetime(df["time"])
df['month'] = df['time'].dt.month_name()
df['month_num'] = df['time'].dt.month
month_order = ['December', 'January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November']
beach_map = {0: "Maroubra", 1: "Coogee", 2: "Clovelly"}
df["beach_name"] = df["beach.x"].map(beach_map)

# Monthly and overall averages
monthly_avg_by_beach = df.groupby(["beach_name", "month"])["bluebottles"].mean().unstack(level=0).reindex(month_order)
overall_monthly_avg = df.groupby("month")["bluebottles"].mean().reindex(month_order)

# Season define
def map_season(month_num):
    if month_num in [12, 1, 2]:
        return 'Summer'
    elif month_num in [3, 4, 5]:
        return 'Autumn'
    elif month_num in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df["season"] = df["time"].dt.month.map(map_season)
seasonal_sum_by_beach = df.groupby(["season", "beach_name"])["bluebottles"].sum().unstack()
seasonal_sum_by_beach = seasonal_sum_by_beach.fillna(0)
season_order = ["Summer", "Autumn", "Winter", "Spring"]
seasonal_sum_by_beach = seasonal_sum_by_beach.loc[season_order]
season_bar_positions = {'Summer': 1.5, 'Autumn': 4.5, 'Winter': 7.5, 'Spring': 10.5}
seasons = {
    'Summer': (0, 3), 'Autumn': (3, 6), 'Winter': (6, 9), 'Spring': (9, 11.5),
}
season_colors = {
    'Summer': '#E6FFE6', 'Autumn': '#FFF3E6', 'Winter': '#E6F2FF', 'Spring': '#DBE0FF',
}
beach_colors = {
    'Maroubra': 'orange',
    'Coogee': 'royalblue',
    'Clovelly': 'olivedrab',
}

# Plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Line plot
ax1.plot(overall_monthly_avg.index, overall_monthly_avg.values,
         marker='o', linewidth=2, color='darkblue', label="Average Bluebottle Level")

# Seasonal background
ax1.axvspan(-0.5, 0, color=season_colors['Spring'], alpha=0.3)
for season, (start, end) in seasons.items():
    ax1.axvspan(start, end, color=season_colors[season], alpha=0.3)
    mid = (start + end) / 2
    ax1.text(mid, ax1.get_ylim()[1] * 0.85, season, fontsize=12, color='dimgray',
             ha='center', va='top', weight='bold', alpha=0.8)

ax1.set_ylabel("Avg Occurences", fontsize=12)
ax1.set_xticks(range(12))
ax1.set_xticklabels(month_order, rotation=45)
ax1.set_xlim(-0.5, 11.5)
ax1.grid(color='lightgray', linestyle='--', linewidth=0.7)
ax1.set_title("Trend and Distribution of Bluebottle Occurrence", fontsize=15)

# Right axis for stacked bars
ax2 = ax1.twinx()
bar_width = 0.8
bottom = [0] * len(season_bar_positions)

# Stacked bars
for beach in ['Maroubra', 'Coogee', 'Clovelly']:
    values = [seasonal_sum_by_beach.loc[season, beach] for season in season_order]
    ax2.bar(season_bar_positions.values(), values, bar_width,
            label=f"{beach} (Total)", color=beach_colors[beach],
            alpha=0.5, bottom=bottom)
    bottom = [i + j for i, j in zip(bottom, values)]

ax2.set_ylabel("Total Occurences", fontsize=12)

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
           loc="upper right", bbox_to_anchor=(1, 0.75), fontsize=9)

plt.tight_layout()
plt.savefig("Trend_Distri_Seasonality_Occurence.png")
plt.show()

# Monthly average of sea water temperature & salinity
sea_water_vars = ['crt_temp', 'crt_salt']
monthly_sea_water = df.groupby('month')[sea_water_vars].mean().reindex(month_order)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot temperature & salinity
ax.plot(monthly_sea_water.index, monthly_sea_water['crt_temp'], marker='o', label='Sea Water Temperature (°C)', linewidth=2, color='firebrick')
ax.set_ylabel('Temperature (°C)', color='firebrick', fontsize=12)
ax.tick_params(axis='y', labelcolor='firebrick')

# Second y-axis for salinity
ax2 = ax.twinx()
ax2.plot(monthly_sea_water.index, monthly_sea_water['crt_salt'], marker='s', label='Sea Water Salinity (PSU)', linewidth=2, color='darkgreen')
ax2.set_ylabel('Salinity (PSU)', color='darkgreen', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkgreen')

# Looks
ax.set_title("Trend of Sea Water Temperature & Salinity", fontsize=15)
ax.set_xticks(range(12))
ax.set_xticklabels(month_order, rotation=45)
ax.grid(color='lightgray', linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.savefig("Trend_SeaWater.png")
plt.show()

