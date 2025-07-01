#================================================================
# Programa 20, PSO CLustering
#================================================================
# Serrano Ramirez Javier
# Fundamentos de Inteligencia Artificial
# IPN ESFM
# Junio 2025
#================================================================
#================================================================
#  Clase de optimización usando enjambres de partículas
#================================================================

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle  # Asegúrate de tener este archivo correctamente 

class PSOClusteringSwarm:
    def __init__(self, n_clusters: int, n_particles: int, data: np.ndarray, hybrid=True, w=0.72, c1=1.49, c2=1.49):
        """
        Inicializa el enjambre.
        :param n_clusters: número de agrupamientos
        :param n_particles: número de partículas
        :param data: arreglo de datos (número_de_puntos x dimensiones)
        :param hybrid: si se debe usar k-means como inicialización
        :param w, c1, c2: parámetros del algoritmo PSO
        """
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.data = data

        self.particles = []
        self.gb_pos = None  # mejor posición global (centroides)
        self.gb_val = np.inf  # mejor valor de función de aptitud
        self.gb_clustering = None  # mejor agrupamiento (clustering) encontrado

        self._generate_particles(hybrid, w, c1, c2)

    def _print_initial(self, iteration, plot):
        print('*** Inicializando enjambre con', self.n_particles, 'PARTÍCULAS, ', self.n_clusters, 'AGRUPAMIENTOS,',
              iteration, 'ITERACIONES MÁXIMAS y con PLOT =', plot, '***')
        print('Datos =', self.data.shape[0], 'puntos en', self.data.shape[1], 'dimensiones')

    def _generate_particles(self, hybrid: bool, w: float, c1: float, c2: float):
        """
        Genera partículas con k agrupamientos y puntos en t dimensiones.
        """
        for i in range(self.n_particles):
            particle = Particle(n_clusters=self.n_clusters, data=self.data, use_kmeans=hybrid, w=w, c1=c1, c2=c2)
            self.particles.append(particle)

    def update_gb(self, particle):
        """
        Actualiza la mejor solución global si esta partícula tiene un mejor valor.
        """
        if particle.pb_val < self.gb_val:
            self.gb_val = particle.pb_val
            self.gb_pos = particle.pb_pos.copy()
            self.gb_clustering = particle.pb_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el proceso de optimización.
        :param iteration: número máximo de iteraciones
        :param plot: si se debe graficar el progreso
        :return: mejor agrupamiento y mejor valor de aptitud
        
        """
        self._print_initial(iteration, plot)
        progress = []

        for i in range(iteration):
            if i % 200 == 0:
                print('Iteración', i, 'Mejor valor (GB) =', self.gb_val)
                print('Mejor agrupamiento hasta ahora =', self.gb_clustering)

                if plot:
                    centroids = self.gb_pos
                    if self.gb_clustering is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.gb_clustering, cmap='viridis')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_pb(data=self.data)
                self.update_gb(particle=particle)

            for particle in self.particles:
                particle.move_centroids(gb_pos=self.gb_pos)

            progress.append([self.gb_pos, self.gb_clustering, self.gb_val])

        print('¡Finalizado!')
        return self.gb_clustering, self.gb_val
