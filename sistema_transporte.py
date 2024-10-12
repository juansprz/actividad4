# Importamos las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar el archivo CSV
data = pd.read_csv('/Users/juanperez/Desktop/IBERO/INTELIGENCIA ARTIFICIAL/ACTIVIDAD 4/datos_transporte_masivo.csv')

# Seleccionamos las columnas relevantes para el modelo (Pasajeros, NivelCongestion, TiempoEspera, DistanciaProxima)
X = data[['Pasajeros', 'NivelCongestion', 'TiempoEspera', 'DistanciaProxima']]

# Creamos el modelo KMeans con 3 clusters (puedes ajustar el número según lo que quieras analizar)
kmeans = KMeans(n_clusters=3)

# Ajustamos el modelo a los datos
kmeans.fit(X)

# Añadimos una nueva columna con el cluster asignado para cada estación
data['Cluster'] = kmeans.labels_

# Guardamos los resultados en un nuevo archivo CSV
data.to_csv('resultado_agrupamiento.csv', index=False)

# Visualización de los resultados (opcional, ajusta según el análisis)
plt.scatter(data['Pasajeros'], data['NivelCongestion'], c=data['Cluster'])
plt.xlabel('Pasajeros')
plt.ylabel('Nivel de Congestion')
plt.title('Agrupamiento de estaciones de transporte')
plt.show()