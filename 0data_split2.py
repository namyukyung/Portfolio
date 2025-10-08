import pandas as pd # data analysis
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

path = "filtered_crops_cleaned.csv"
crop = pd.read_csv(path)
print(crop.head(5))

sns.jointplot(x="rainfall",y="humidity",data=crop[(crop['temperature']<40) & 
                                                  (crop['rainfall']>40)],height=10,hue="label")
plt.show()

# Define the labels to remove based on analysis
labels_to_remove = ['mungbean', 'pigeonpeas', 'papaya', 'apple', 'mango', 'cotton', 'jute']

# Filter the dataset to keep only distinct crops
filtered_crop = crop[~crop['label'].isin(labels_to_remove)]

# Save the filtered dataset to a new CSV file
filtered_crop.to_csv("distinct_crops.csv", index=False)

# Verify the remaining labels
print(f"Remaining distinct labels: {filtered_crop['label'].unique()}")
