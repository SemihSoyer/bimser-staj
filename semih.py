import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Örnek veri oluştur
np.random.seed(0)
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

# K-Means modelini oluştur
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Merkezleri ve etiketleri al
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Veriyi ve merkezleri görselleştir
colors = ["g.", "r."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()
