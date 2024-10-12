import numpy as np
import csv
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# Cargar y limpiar embeddings y labels
def cargarIncrustaciones(file_path):
    incrustaciones = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            vectores = line.strip().split()
            if len(vectores) > 2:
                label = vectores[0]
                if "Dataset" in label:
                    nuevo_label = label.split("Dataset", 1)[-1].lstrip('-_')
                    embedding = np.array([float(x) for x in vectores[2:]])
                    incrustaciones.append(embedding)
                    labels.append(nuevo_label)
    return np.array(incrustaciones), labels

# Cargar archivo CSV de canciones originales y plagios
def cargarCasos(file_path):
    casos = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Saltar la cabecera
        for row in reader:
            cancion_original, cancion_plagio = row
            if cancion_plagio != "Vacio":  # Filtrar los casos donde el plagio es "Vacio"
                casos.append((cancion_original, cancion_plagio))
    return casos

# Normalizar datos
def preprocesarData(incrustaciones):
    scaler = StandardScaler()
    datos_normalizados = scaler.fit_transform(incrustaciones)
    return datos_normalizados

# Visualizar clusters con labels
def visualizarClusters(incrustaciones, modelo_labels, labels, title="Modelo"):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(incrustaciones)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=modelo_labels, cmap='viridis')

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8, alpha=0.75)

    plt.title(f"Clusters - {title}")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar(scatter, label="Cluster")
    plt.show()

# Aplicar K-means
def aplicarKMeans(incrustaciones, n_clusters, random_state=491):
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=random_state, 
                    init='k-means++',
                    algorithm="elkan",
                    max_iter=800)
    kmeans.fit(incrustaciones)
    return kmeans

# Aplicar Mean Shift
def aplicarMeanShift(incrustaciones):
    bandwidth = estimate_bandwidth(incrustaciones, quantile=0.2, n_samples=44)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(incrustaciones)
    return mean_shift

# Aplicar DBSCAN
def aplicarDBSCAN(incrustaciones, eps=0.2, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(incrustaciones)
    return dbscan

# Calcular accuracy
def calcularAccuracy(casos, labels, modelo_labels):
    total_casos = 0
    casos_correctos = 0
    
    # Convertir labels en un diccionario para facilitar la búsqueda
    label_to_cluster = {label: modelo_labels[i] for i, label in enumerate(labels)}
    
    for original, plagio in casos:
        if original in label_to_cluster and plagio in label_to_cluster:
            total_casos += 1
            # Si ambos están en el mismo cluster
            if label_to_cluster[original] == label_to_cluster[plagio]:
                casos_correctos += 1

    accuracy = casos_correctos / total_casos if total_casos > 0 else 0
    return accuracy

def calcular_distancias_clusters(modelo):
    # Obtener los centros de los clusters
    centros = modelo.cluster_centers_
    # Calcular distancias entre cada par de centros
    distancias = euclidean_distances(centros, centros)
    return distancias

# Funciones para guardar clusters
def guardarClustersTxt(modelo_labels, labels, nombre_archivo, distancias_clusters=None):
    clusters = {}
    for i, label in enumerate(labels):
        cluster = modelo_labels[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(label)

    with open(nombre_archivo, 'w') as file:
        for cluster, items in clusters.items():
            file.write(f"Cluster {cluster}:\n")
            for item in items:
                file.write(f"- {item}\n")
            file.write("----\n")
        
        if distancias_clusters is not None:
            file.write("Distancias entre clusters:\n")
            for i, distancias in enumerate(distancias_clusters):
                file.write(f"Cluster {i}: {distancias.tolist()}\n")


def guardarClustersDiccionario(modelo_labels, labels, nombre_archivo, distancias_clusters=None):
    clusters = {}
    for i, label in enumerate(labels):
        cluster = modelo_labels[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(label)

    with open(nombre_archivo, 'w') as file:
        file.write("cluster_dict = {\n")
        for cluster, items in clusters.items():
            file.write(f"    {cluster}: {items},\n")
        file.write("}\n")
        
        if distancias_clusters is not None:
            file.write("clusters_cercanos = {\n")
            for i, distancias in enumerate(distancias_clusters):
                file.write(f"    {i}: {np.argsort(distancias).tolist()},\n")
            file.write("}\n")



#--------------------------------------- EJECUCION------------------------------------------------
file_path_incrustaciones = '../midi2vec/edgelist2vec/embeddings.bin'
file_path_casos = 'CasosRealesInformacion.csv'

# Cargar incrustaciones y labels
incrustaciones, labels = cargarIncrustaciones(file_path_incrustaciones)

# Preprocesar los datos
print("Preprocesar los datos")
datos_normalizados = preprocesarData(incrustaciones)

# Cargar casos reales
print("Cargar casos reales")
casos = cargarCasos(file_path_casos)

# Aplicar K-means
print("\nK-means")
n_clusters = 4
random_state =32
kmeans_model = aplicarKMeans(datos_normalizados, n_clusters)
# Calcular distancias entre clusters
distancias_kmeans = calcular_distancias_clusters(kmeans_model)
accuracy_kmeans = calcularAccuracy(casos, labels, kmeans_model.labels_)
print(f"Accuracy K-means: {accuracy_kmeans:.2f}")
guardarClustersTxt(kmeans_model.labels_, labels, "clusters_kmeans.txt", distancias_clusters=distancias_kmeans)
guardarClustersDiccionario(kmeans_model.labels_, labels, "clusters_kmeans.py", distancias_clusters=distancias_kmeans)


# Aplicar Mean Shift
print("\nMean Shift")
mean_shift_model = aplicarMeanShift(datos_normalizados)
distancias_meanshift = calcular_distancias_clusters(mean_shift_model)
accuracy_meanshift = calcularAccuracy(casos, labels, mean_shift_model.labels_)
print(f"Accuracy Mean Shift: {accuracy_meanshift:.2f}")
guardarClustersTxt(mean_shift_model.labels_, labels, "clusters_meanshift.txt", distancias_clusters=distancias_meanshift)
guardarClustersDiccionario(mean_shift_model.labels_, labels, "clusters_meanshift.py", distancias_clusters=distancias_meanshift)

# Aplicar DBSCAN
print("\nDBSCAN")
dbscan_model = aplicarDBSCAN(datos_normalizados, eps=0.05, min_samples=5)
# DBSCAN no tiene centros de clusters, así que no se calculan distancias
accuracy_dbscan = calcularAccuracy(casos, labels, dbscan_model.labels_)
print(f"Accuracy DBSCAN: {accuracy_dbscan:.2f}")
guardarClustersTxt(dbscan_model.labels_, labels, "clusters_dbscan.txt")
guardarClustersDiccionario(dbscan_model.labels_, labels, "clusters_dbscan.py")
