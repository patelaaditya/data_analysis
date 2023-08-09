import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('sales_data.csv')

# Data cleaning
data_cleaned = data.dropna()  # Handle missing values

# Data analysis
total_revenue = data_cleaned['revenue'].sum()
best_selling_products = data_cleaned.groupby('product')['quantity'].sum().sort_values(ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=best_selling_products.index, y=best_selling_products.values)
plt.xticks(rotation=45)
plt.xlabel('Product')
plt.ylabel('Quantity Sold')
plt.title('Best Selling Products')
plt.show()

# Customer segmentation
X = data_cleaned[['quantity', 'price']]
kmeans = KMeans(n_clusters=3)
data_cleaned['cluster'] = kmeans.fit_predict(X)

# Recommendations
cluster_summary = data_cleaned.groupby('cluster').mean()