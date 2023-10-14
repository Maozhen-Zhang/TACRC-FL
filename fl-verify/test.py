from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import numpy as np

# 生成随机高维数据，假设有5个簇，每个簇的维度为10
num_samples = 100
num_features = 10
num_clusters = 5

data = np.random.rand(num_samples, num_features)

# 假设data是你的数据
# data = np.array([[1, 2],
#                  [1, 3],
#                  [1, 4],
#                  [2, 4],
#                  [2, 4],
#                  [2, 4],
#                  [2, 4]])

# 创建KMeans对象并指定簇的数量
k = 3  # 设定簇的数量
kmeans = KMeans(n_clusters=k, n_init=10)

# 对数据进行聚类
kmeans.fit(data)

# 获取每个样本所属的簇
labels = kmeans.labels_

# 获取每个簇的中心点
cluster_centers = kmeans.cluster_centers_

# 打印每个样本的聚类标签和中心点
# for i in range(len(data)):
#     print("样本:", data[i], " 聚类标签:", labels[i])
#

# print("簇中心点:", cluster_centers)


# 假设data是你的数据，labels是对应的聚类标签
#总轮廓系数
silhouette_avg = silhouette_score(data, labels)
print("平均轮廓系数:", silhouette_avg)
#每个样本的轮廓系数
sample_silhouette_values = silhouette_samples(data, labels)
# 打印每个样本的轮廓系数值
for i, silhouette in enumerate(sample_silhouette_values):
    print(f"样本 {i}: 轮廓系数值 = {silhouette}")


unique_labels = np.unique(labels)
print(unique_labels)
print(labels)
for label in unique_labels:
    cluster_samples = data[labels == label]
    cluster_silhouette_values = silhouette_samples(cluster_samples, labels[labels == label])
    cluster_avg_silhouette = np.mean(cluster_silhouette_values)
    print(f"簇 {label} 的平均轮廓系数: {cluster_avg_silhouette}")