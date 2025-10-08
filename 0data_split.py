import pandas as pd # data analysis
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

path = "Crop_recommendation.csv"
crop = pd.read_csv(path)
print(crop.head(5))

print(crop.info())
print(crop['label'].unique())
print(crop.describe())

sns.jointplot(x="rainfall",y="humidity",data=crop[(crop['temperature']<40) & 
                                                  (crop['rainfall']>40)],height=10,hue="label")
plt.show()

# Group by 'label' and calculate the mean for each feature
label_summary = crop.groupby('label').mean()

# standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(label_summary)

# pairwise distances between crop labels based on their feature means
distance_matrix = squareform(pdist(features_scaled, metric='euclidean'))

# Create a df to better analyze the distances
distance_df = pd.DataFrame(distance_matrix, index=label_summary.index, columns=label_summary.index)

# Replace diagonal values (self-comparison) with NaN to exclude them
np.fill_diagonal(distance_matrix, np.nan)

# Find the most similar pairs (smallest distances)
similar_pairs = []
threshold = 5.0  # threshold for similarity

for label1 in distance_df.index:
    for label2 in distance_df.columns:
        if label1 != label2 and not np.isnan(distance_df[label1][label2]) and distance_df[label1][label2] < threshold:
            similar_pairs.append((label1, label2, distance_df[label1][label2]))

# sort the similar pairs by distance
similar_pairs = sorted(similar_pairs, key=lambda x: x[2])

# the most similar labels
if similar_pairs:
    print("Most similar labels:")
    for pair in similar_pairs:
        print(f"Labels: {pair[0]} and {pair[1]} with distance: {pair[2]:.2f}")
else:
    print("No similar labels found within the threshold.")

# heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(distance_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Pairwise distance between crop labels")
plt.show()

# the most similar pairs
# apple and grapes
# balckgram and mothbeans: mothbeans
#coffee and jute
#cotton and watermelon
#jute and rice
#lentil and mothbeans and mungbean:lentil and mothbeans
# mothbeans and blackgram, lentil, and mungbean: mothbeans and blackgram, lentil
# muskmelon and watermelon: muskmelon 
# orange and pomegranate: pomegranate


# mothbeans, blackgram, lentil, muskmelon, pomegranate
labels_to_remove = ['mothbeans', 'lentil', 'blackgram', 'muskmelon', 'pomegranate']  # Verify these names match exactly

# Filter the dataset to remove unwanted labels
filtered_crop = crop[~crop['label'].isin(labels_to_remove)]
filtered_crop.to_csv("filtered_crops_cleaned.csv", index=False)
print("Remaining labels after removal:", filtered_crop['label'].unique())

