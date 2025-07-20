from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

datos = pd.read_csv('./assets/clientes.csv')

X = datos[['edad', 'compra_promedio', 'compras_anuales']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

datos['segmento'] = kmeans.labels_

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans.labels_)
plt.title('Segmentación de Clientes')
plt.show()

# Análisis Exploratorio de Datos (EDA)
import seaborn as sns

# Correlación
sns.pairplot(datos[['edad', 'compra_promedio', 'compras_anuales']])
plt.suptitle("Distribución de Variables", y=1.02)
plt.show()

# Heatmap de correlación
sns.heatmap(datos[['edad', 'compra_promedio', 'compras_anuales']].corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()
# 2. Determinar el Número Óptimo de Clusters (Elbow Method + Silhouette)
from sklearn.metrics import silhouette_score

# Elbow Method
inertias = []
silhouettes = []
K = range(2, 10)
for k in K:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, model.labels_))

# Gráfica de codo
plt.plot(K, inertias, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia (Within-Cluster Sum of Squares)')
plt.title('Método del Codo')
plt.show()

# Silhouette score
plt.plot(K, silhouettes, marker='x', color='green')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.title('Score de Silueta')
plt.show()
