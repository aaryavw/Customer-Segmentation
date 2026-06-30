import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==============================================================================
# 1. LOAD AND INSPECT THE DATA
# ==============================================================================
print("Loading banking customer dataset...")
# Make sure the filename matches your downloaded Kaggle file
df = pd.read_csv('CC GENERAL.csv')

print(f"Dataset Shape: {df.shape}")
print("\nColumns available for clustering:")
print(df.columns.tolist())

# ==============================================================================
# 2. DATA CLEANING & PREPROCESSING
# ==============================================================================
# Drop the Customer ID column because it's just text and has no numerical meaning
X = df.drop('CUST_ID', axis=1)

# Handle missing values (e.g., MINIMUM_PAYMENTS or CREDIT_LIMIT might have nulls)
# We fill missing values with the median of their respective columns
X = X.fillna(X.median())

# Scale the data. In K-Means, features with massive numbers (like CREDIT_LIMIT) 
# will naturally dominate smaller scales (like PURCHASES_FREQUENCY) if not normalized.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================================================================
# 3. THE ELBOW METHOD (Finding the Optimal Number of Clusters)
# ==============================================================================
print("\nRunning the Elbow Method to find optimal cluster counts...")
wcss = []  # Within-Cluster Sum of Square (Inertia)

# Test clustering options from 1 group up to 10 groups
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method Grid')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
print("Displaying Elbow plot... Look for the 'bend' or elbow point.")
plt.show()

# ==============================================================================
# 4. TRAINING THE FINAL K-MEANS MODEL
# ==============================================================================
# Based on the typical configuration of this dataset, 4 clusters is usually optimal
optimal_k = 4
print(f"\nTraining final K-Means model with K={optimal_k} clusters...")

kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Add the cluster labels back onto our original dataframe for banking analysis
df['Cluster'] = cluster_labels

# ==============================================================================
# 5. PROFILE AND UNDERSTAND THE CUSTOMER SEGMENTS
# ==============================================================================
print("\n================ CUSTOMER SEGMENT PROFILES ================")
# Calculate the average behavior profile of each group
cluster_profiles = df.drop('CUST_ID', axis=1).groupby('Cluster').mean()
print(cluster_profiles[['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']])

# ==============================================================================
# 6. VISUALIZE THE CLUSTERS (Using PCA)
# ==============================================================================
# Since we can't look at a 17-dimensional graph, we compress the data into 
# 2 primary dimensions using PCA (Principal Component Analysis) for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='Set1', alpha=0.7)
plt.title('Banking Customer Segments Visualized via PCA Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Customer Cluster')
plt.show()
