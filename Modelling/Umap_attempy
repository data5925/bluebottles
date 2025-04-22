# 3D UMAP
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("final_data.csv")
drop_cols = ['time', 'presence', 'bluebottles']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['presence']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

reducer_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, random_state=42)
X_umap_3d = reducer_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['#5DADE2', '#E74C3C']
for cls in [0, 1]:
    idx = y == cls
    ax.scatter(
        X_umap_3d[idx, 0], X_umap_3d[idx, 1], X_umap_3d[idx, 2],
        label=f"Presence = {cls}",
        alpha=0.5,
        s=20,
        color=colors[cls]
    )
ax.set_title("3D UMAP Projection of Environmental Features", fontsize=14)
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")
ax.legend()
plt.tight_layout()
plt.show()

#corr
# !pip install umap-learn

import pandas as pd
import numpy as np
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("final_data.csv")
df = df.drop(columns=['time', 'bluebottles']) 
target = df['presence']
features = df.drop(columns=['presence'])

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

reducer = umap.UMAP(n_components=3, random_state=42)
umap_embedding = reducer.fit_transform(scaled_features)

umap_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
umap_df["presence"] = target.values

merged_df = pd.concat([umap_df[["UMAP1", "UMAP2", "UMAP3"]], features.reset_index(drop=True)], axis=1)
correlation_matrix = merged_df.corr().loc[["UMAP1", "UMAP2", "UMAP3"]]

plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between UMAP Components and Environmental Features")
plt.tight_layout()
plt.show()

umap_with_features = pd.concat([umap_df[['presence']], pd.DataFrame(scaled_features, columns=features.columns)], axis=1)
corr = umap_with_features.corr()['presence'].drop('presence').sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
plt.title('Correlation between Original Features and Presence (via UMAP)')
plt.xlabel('Correlation with Presence')
plt.tight_layout()
plt.show()
