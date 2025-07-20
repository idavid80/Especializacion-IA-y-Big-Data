import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# 1. Cargar los datos
datos = pd.read_csv('./assets/clientes.csv')

# Selección de variables relevantes
X = datos[['edad', 'compra_promedio', 'compras_anuales']]

# 2. Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Análisis Exploratorio

# Pairplot para distribución y relación entre variables
sns.pairplot(datos[['edad', 'compra_promedio', 'compras_anuales']])
plt.suptitle("Distribución de Variables", y=1.02)
plt.show()

# Heatmap de correlación
sns.heatmap(datos[['edad', 'compra_promedio', 'compras_anuales']].corr(), annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

# 4. Método del Codo y Silhouette
inertias = []
silhouettes = []
K = range(2, 10)

for k in K:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, model.labels_))

# Gráfico del codo
plt.plot(K, inertias, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# Gráfico de silhouette
plt.plot(K, silhouettes, marker='x', color='green')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.title('Score de Silueta')
plt.show()

# 5. KMeans final (k=3 como ejemplo)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Asignar segmento a cada cliente
datos['segmento'] = kmeans.labels_

# 6. Interpretación de Clusters
resumen_segmentos = datos.groupby('segmento').agg({
    'edad': 'mean',
    'compra_promedio': 'mean',
    'compras_anuales': 'mean',
    'cliente_id': 'count'
}).rename(columns={'cliente_id': 'n_clientes'}).round(2)

print("\nResumen por segmento:\n")
print(resumen_segmentos)

# Descripción detallada
for i, row in resumen_segmentos.iterrows():
    print(f"Segmento {i}: Edad promedio {row['edad']} años, "
          f"Compra promedio ${row['compra_promedio']}, "
          f"{row['compras_anuales']} compras/año, "
          f"{int(row['n_clientes'])} clientes")

# 7. Visualización 2D de Clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=datos['segmento'], cmap='viridis', s=50)
plt.xlabel('Edad (escalada)')
plt.ylabel('Compra Promedio (escalada)')
plt.title('Segmentación de Clientes (2D)')
plt.show()

# 8. Visualización 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=datos['segmento'], cmap='viridis', s=60)
ax.set_xlabel('Edad')
ax.set_ylabel('Compra Promedio')
ax.set_zlabel('Compras Anuales')
plt.title('Segmentación de Clientes (3D)')
plt.show()

# 9. Guardar el resultado
datos.to_csv('./soluciones/ia/lab4/clientes_segmentados.csv', index=False)
print("\nArchivo 'clientes_segmentados.csv' generado con éxito.")
