from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_clustering(df, output_dir="outputs"):
    X_cluster = df[['Area', 'Total_Price_EUR']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal eps using k-distance graph
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(X_scaled)
    distances, indices = neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, 9], axis=0)
    
    # Plot k-distance graph to visualize elbow
    plt.figure(figsize=(8, 4))
    plt.plot(distances)
    plt.ylabel("10-NN Distance")
    plt.xlabel("Data Points (sorted)")
    plt.title("K-distance Graph (Elbow = Optimal eps)")
    plt.savefig(f"{output_dir}/k_distance_graph.png")
    plt.close()
    
    # Use elbow point (around 90th percentile of distances)
    eps_optimal = np.percentile(distances, 95)
    
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN with optimal eps
    dbscan = DBSCAN(eps=eps_optimal, min_samples=10)
    df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
    
    # Print diagnostics
    n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
    n_noise = list(df['DBSCAN_Cluster']).count(-1)
    print(f"\nDBSCAN Diagnostics (eps={eps_optimal:.4f}):")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise} ({100*n_noise/len(df):.1f}%)")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.scatterplot(x='Area', y='Total_Price_EUR', hue='KMeans_Cluster', data=df, palette='viridis', ax=axes[0])
    axes[0].set_title('K-Means Clustering (k=3)')
    
    sns.scatterplot(x='Area', y='Total_Price_EUR', hue='DBSCAN_Cluster', data=df, palette='Set1', ax=axes[1])
    axes[1].set_title(f'DBSCAN Clustering (eps={eps_optimal:.4f}, -1 = Noise)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/clustering_results.png")
    plt.close()
    
    return df