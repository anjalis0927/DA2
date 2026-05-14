import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample customer data
data = {
    'Age': [22, 25, 47, 52, 46, 56, 23, 27, 30, 40],
    'Income': [25000, 27000, 65000, 70000, 62000, 80000, 29000, 32000, 40000, 50000],
    'SpendingScore': [80, 76, 20, 18, 25, 15, 85, 70, 60, 50]
}

# Create dataframe
df = pd.DataFrame(data)

print("\nCustomer Data:\n")
print(df)

# Visualize data
sns.pairplot(df)
plt.show()

# Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = kmeans.fit_predict(scaled_data)

print("\nClustered Data:\n")
print(df)

# Scatter plot
plt.figure(figsize=(8,6))

plt.scatter(
    df['Income'],
    df['SpendingScore'],
    c=df['Cluster'],
    cmap='viridis',
    s=100
)

plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')

plt.show()