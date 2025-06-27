import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv('nigerian-songs.csv')

# 2. 选择音频特征列
audio_features = ['danceability', 'acousticness', 'energy',
                  'instrumentalness', 'liveness', 'loudness',
                  'speechiness', 'tempo']

# 3. 数据预处理
X = df[audio_features].copy()

# 处理缺失值（如果有）
X.fillna(X.mean(), inplace=True)

# 4. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 使用肘部法则确定最佳K值
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

    # 计算轮廓系数（数据量大时可能较慢）
    if len(X_scaled) > 10000:  # 如果数据量太大可跳过
        silhouette_scores.append(None)
    else:
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

# 可视化肘部法则
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# 可视化轮廓系数
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'go-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.tight_layout()
plt.show()

# 6. 选择最佳K值（这里示例选择4，实际应根据图表选择）
best_k = 3

# 7. 使用最佳K值进行聚类
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 8. 将聚类结果添加到原始数据
df['cluster'] = kmeans.labels_

# 9. 分析聚类结果
# 查看每个聚类的歌曲数量
print("Cluster distribution:")
print(df['cluster'].value_counts())

# 查看每个聚类的特征均值
cluster_means = df.groupby('cluster')[audio_features].mean()
print("\nCluster characteristics:")
print(cluster_means)


# 10. 可视化聚类特征（雷达图）
def plot_radar_chart(cluster_means, features):
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for idx, row in cluster_means.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title('Audio Features by Cluster', size=20, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


plot_radar_chart(cluster_means, audio_features)
