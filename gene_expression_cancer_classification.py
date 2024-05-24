import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

def kmeans_auto(data, max_clusters=10):
    
    silhouette_scores = []
    
    # Avalia o K-means para diferentes números de clusters
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
    # Encontrar o número ideal de clusters usando o método silhouette
    optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Adicionamos 2 devido à diferença
    print("Número ideal de clusters K-Means:", optimal_num_clusters)
    
    # Aplicar K-means com o número ideal de clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(data)
    
    return clusters

def agglomerative_auto(data, max_clusters=10):

    silhouette_scores = []
    
    # Avalia o Agglomerative Clustering para diferentes números de clusters
    for n_clusters in range(2, max_clusters+1):
        agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_cluster.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
    # Encontrar o número ideal de clusters usando o método silhouette
    optimal_num_clusters = np.argmax(silhouette_scores) + 2  # Adicionamos 2 devido à diferença
    print("Número ideal de clusters Agglomerative Clustering:", optimal_num_clusters)
    
    # Aplicar Agglomerative Clustering com o número ideal de clusters
    agg_cluster = AgglomerativeClustering(n_clusters=optimal_num_clusters)
    clusters = agg_cluster.fit_predict(data)
    
    return clusters

def dbscan_auto(data, eps_values, min_samples=5):

    silhouette_scores = []
    
    # Avalia o DBSCAN para diferentes valores de eps
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data)
        
        # Calcula o escore silhouette
        if len(set(cluster_labels)) > 1:  # Precisa de mais de um cluster para calcular o silhouette
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(-1)  # Se houver apenas um cluster, o silhouette é indefinido
        
    # Encontra o valor de 'eps' que maximiza o silhouette
    optimal_eps_index = np.argmax(silhouette_scores)
    optimal_eps = eps_values[optimal_eps_index]
    print("Valor ótimo de eps DBSCAN:", optimal_eps)
    
    # Aplica o DBSCAN com o valor ótimo de eps
    dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    
    return clusters


def main():
    
    # Ler o dataset desejado e remover linhas nulas/vazias 
    dados = pd.read_csv(fr"data.csv")
    dados = dados.dropna()
    
    # Remover colunas não numéricas
    dados = dados.select_dtypes(include=['float64', 'int64'])

    # Lendo o arquivo referente aos labels
    labels = pd.read_csv(fr"labels.csv")
    labels = labels.drop(labels.columns[0], axis=1)
    
    # Inicializar o normalizador StandardScaler
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)
    
    #Aplicar o PCA
    pca = PCA()  # Especificando que queremos reduzir a dimensionalidade para 2
    dados_pca = pca.fit_transform(dados_normalizados)
    
    # Criar um DataFrame com os resultados do PCA
    pca_df = pd.DataFrame(data=dados_pca)
    
   # eps_values = [0.1, 0.5, 1.0, 1.5, 2.0]  # Lista de valores de eps para serem avaliados
    eps_values = np.linspace(0.1, 2.0, num=20)
    
    result_kmeans = kmeans_auto(pca_df)
    result_agglomerative = agglomerative_auto(pca_df)
    result_DBSCAN = dbscan_auto(pca_df, eps_values)
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=result_kmeans, cmap='viridis')
    plt.title("K-Means")
    
    plt.subplot(1, 3, 2)
    plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=result_agglomerative, cmap='viridis')
    plt.title("Agglomerative Clustering")

    plt.subplot(1, 3, 3)
    plt.scatter(dados_pca[:, 0], dados_pca[:, 1], c=result_DBSCAN, cmap='viridis')
    plt.title("DBSCAN")

    plt.show()

if __name__ == "__main__":
    main()