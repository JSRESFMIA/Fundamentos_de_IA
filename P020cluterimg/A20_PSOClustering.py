#================================================================
# Codigo numero 20, Agrupamiento usando conjuntos de particulas
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================

#=================================
#  Módulos necesarios
#=================================
import pandas as pd     # Para manejo de datos
import numpy as np      # Para operaciones numéricas
from pso_clustering import PSOClusteringSwarm  # Algoritmo PSO para clustering (debes tener este archivo o módulo disponible)

# Activar o desactivar gráficos
plot = True

#=================================
#  Leer datos (desde archivo .txt)
#=================================
data_points = pd.read_csv('iris.txt', sep=',', header=None)

#=================================
#  Obtener las clases reales (columna 4, recordando que se cuenta desde 0)
#=================================
clusters = data_points[4].values  # Convierte la columna de etiquetas a un arreglo de NumPy

#=================================
#  Eliminar la columna 4 del conjunto de datos
#=================================
data_points = data_points.drop([4], axis=1)  # axis=1 indica columna

#=================================
#  Usar solo las columnas 0 y 1 para graficar (x, y)
#=================================
if plot:
    data_points = data_points[[0, 1]]

#=================================
#  Convertir los datos a arreglo NumPy 2D
#=================================
data_points = data_points.values

#=================================
#  Ejecutar el algoritmo de agrupamiento con PSO
#=================================
pso = PSOClusteringSwarm(n_clusters=3, n_particles=10, data=data_points, hybrid=True)
pso.start(iteration=1000, plot=plot)  # Ejecutar durante 1000 iteraciones

#=================================
#  Mapeo de clases reales a números
#=================================
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])



