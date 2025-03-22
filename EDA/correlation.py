import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_data.csv", parse_dates=['time'])

# Get correlation matrix and plot heatmap
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Cleaned Bluebottle Dataset")
plt.savefig('heatmap.png', bbox_inches='tight')
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
plt.savefig('high_corr_scatterplots.png', bbox_inches='tight')
plt.show()

"""
Observations:
- None or likely bluebottle sightings correspond to an absence, while some or many sightings correspond to a presence
- Strong positive correlation between surface air temperature (Kelvin) and potential water temperature (Celsius) is expected
- Strong positive correlation between sea water x velocity (crt_u) and y velocity (crt_v) means dominant current directions are NE and SW
- Wave energy is proportional to the square of wave height: bigger waves (hs) result in exponentially more energy being carried by the waves (cge) (strong positive exponential correlation)
- Relatively strong negative correlation between t01 (avg time between waves) and fp (frequency of wave peaks) make sense (higher t01 = slower waves, higher fp = faster waves)
- The sin of dir and cosine of dir make parts of sin and cosine waves respectively (last 2 scatterplots), and form a part of the unit circle together (3rd row, right)
- The above 3 graphs being incomplete implies that the directions of the waves in our data doesn't cover the full 0 to 360 range, which makes sense

"""