# -*- coding: utf-8 -*-
"""Diet Plan Recommendation Model (Simplified & Clean Version)"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Load Data ---
df = pd.read_csv('/Users/dhrumilpatel/DietPlanRecommendationSystem/cleaned_dataset.csv')

# Remove unwanted unnamed columns (if any)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# --- 2. Handle Missing Values ---
# Fill missing numeric values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop rows where food name is missing
df.dropna(subset=['food'], inplace=True)

# --- 3. Create Veg/Non-Veg Feature ---
non_veg_keywords = [
    'chicken', 'beef', 'pork', 'lamb', 'fish', 'salmon', 'tuna',
    'shrimp', 'crab', 'lobster', 'egg', 'turkey', 'bacon',
    'sausage', 'ham', 'venison', 'duck', 'goose'
]

def classify_food(food_name):
    """Classify food as veg or non-veg based on keywords."""
    name = str(food_name).lower()
    if 'cheese' in name:
        return 'veg'
    for keyword in non_veg_keywords:
        if keyword in name:
            return 'non-veg'
    return 'veg'

df['food_type'] = df['food'].apply(classify_food)

# --- 4. Feature Selection ---
features = ['Caloric Value', 'Protein', 'Fat', 'Carbohydrates']
X = df[features]

# --- 5. Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. K-Means Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# --- 7. Label Clusters ---
# Inverse transform centers to get values in original scale
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)

# Sort clusters by calorie level
calorie_sorted = cluster_centers.sort_values('Caloric Value').index

cluster_labels = {
    calorie_sorted[0]: 'Weight Loss',
    calorie_sorted[1]: 'Healthy',
    calorie_sorted[2]: 'Weight Gain'
}

df['cluster_label'] = df['cluster'].map(cluster_labels)

# --- 8. Save Model, Scaler, and Data ---
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
df.to_csv('clustered_food_data.csv', index=False)

# --- 9. Show Results ---
print("\n--- Cluster Centers (Nutritional Averages) ---")
print(cluster_centers)

print("\n--- Cluster Label Mapping ---")
print(cluster_labels)

print("\n--- Sample of Clustered Data ---")
print(df[['food', 'Caloric Value', 'Protein', 'Fat', 'Carbohydrates',
          'food_type', 'cluster_label']].head())

print("\n✔ Model, scaler, and clustered data saved successfully.")
