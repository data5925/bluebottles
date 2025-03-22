import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_data.csv", parse_dates=['time'])

# Get correlation matrix and plot heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cleaned Bluebottle Dataset")
plt.show()

# Identify highly correlated variable pairs (absolute correlation > 0.7)
high_corr_pairs = (
    corr_matrix.where(lambda x: abs(x) > 0.7)
    .stack()
    .reset_index()
    .rename(columns={0: "correlation"})
)

# Remove mirrored and self-correlating pairs
high_corr_pairs = high_corr_pairs[high_corr_pairs["level_0"] != high_corr_pairs["level_1"]]
high_corr_pairs = high_corr_pairs.drop_duplicates(subset=["correlation"])
high_corr_pairs

"""
         level_0       level_1  correlation
3       presence   bluebottles     0.873358
8       crt_temp      wnd_temp     0.976827
10         crt_u         crt_v     0.980680
19       wave_hs      wave_cge     0.945487
21      wave_t01       wave_fp    -0.782807
27  wave_sin_dir  wave_cos_dir     0.835671
28  wave_sin_dir      wave_dir    -0.901167
31  wave_cos_dir      wave_dir    -0.988970
"""

# Create scatter plots for each highly correlated pair
top_correlated_pairs = list(zip(high_corr_pairs["level_0"], high_corr_pairs["level_1"]))
fig, axes = plt.subplots(4, 2, figsize=(10, 8))
axes = axes.flatten()

for i, (var1, var2) in enumerate(top_correlated_pairs):
    sns.scatterplot(data=df, x=var1, y=var2, alpha=0.5, ax=axes[i])
    axes[i].set_title(f"{var1} vs {var2} (r={high_corr_pairs.iloc[i]['correlation']:.2f})")

plt.tight_layout()
plt.show()

"""
Observations:
- 
- 
- 
- 
- 
- 
- 
- 
"""