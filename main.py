
import osmnx as ox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from shapely.geometry import Point
import geopandas as gpd
import seaborn as sns 

# Configuración inicial de OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True
#Samuel Isaí Ramos Díaz
# DEFINICIÓN DE LA ZONA DE ESTUDIO.
# Despues cambiar a 'Guadalajara, Jalisco, Mexico'. Tarda mucho en cargar
municipios_zmg = [
    "Guadalajara, Jalisco, Mexico",
    "Zapopan, Jalisco, Mexico",
     "San Pedro Tlaquepaque, Jalisco, Mexico",
    "Tonalá, Jalisco, Mexico",
     "Tlajomulco de Zúñiga, Jalisco, Mexico" #
]

print(f"Configuración lista. Zona seleccionada: {municipios_zmg}")

print(f"1. Descargando mapa base de: {municipios_zmg}...")
G = ox.graph_from_place(municipios_zmg, network_type='drive')

print("2. Descargando Puntos de Interés")

tags_zmg = {
    'building': ['apartments', 'commercial', 'office', 'retail', 'residential'],
    'amenity': ['school', 'university', 'hospital'],
    'office': True, # Todas las oficinas
    'leisure': ['park', 'plaza']
}

print("Descargando Puntos de Interés de la ZMG...")
pois = ox.features_from_place(municipios_zmg, tags=tags_zmg)

# Necesitamos convertirlos a puntos (x, y) para el K-Means.
pois_centroids = pois.geometry.centroid

# Extraemos coordenadas (Longitud, Latitud)
X_coords = np.array([[point.x, point.y] for point in pois_centroids])

print(f"¡Listo! Se encontraron {len(X_coords)} puntos de alta actividad real.")
print("Estos puntos representan dónde la gente vive, trabaja o estudia.")

print("3. Calculando Paradas Óptimas sobre la densidad real...")

# Definimos cuántas paradas queremos (puedes ajustar este número)
NUM_PARADAS = 20
# NUM_PARADAS = 50

# K-Means ahora se entrena con DATOS REALES de ubicación de edificios/comercios
kmeans = KMeans(n_clusters=NUM_PARADAS, random_state=42, n_init=10)
kmeans.fit(X_coords)

paradas_optimas = kmeans.cluster_centers_


fig, ax = ox.plot_graph(G, show=False, close=False,
                        edge_color='#555555', edge_linewidth=0.5,
                        node_size=0, bgcolor='black', figsize=(12,12))

# 1. Dibujamos el MAPA DE CALOR (KDE Plot)
sns.kdeplot(x=X_coords[:,0], y=X_coords[:,1],
            ax=ax, cmap="magma", fill=True, alpha=0.45,
            levels=100, thresh=0.05)

# 2. Dibujamos los puntos reales (Edificios/Escuelas) - Pequeños puntos blancos
ax.scatter(X_coords[:, 0], X_coords[:, 1], c='white', s=2, alpha=0.3, label='Puntos de Demanda (Edificios)')

# 3. Dibujamos las PARADAS OPTIMIZADAS - X Cian grandes
ax.scatter(paradas_optimas[:, 0], paradas_optimas[:, 1], c='cyan',
           s=200, marker='X', edgecolors='white', linewidth=2,
           label='Paradas Optimizadas (Centroides)')

plt.title(f"Mapa de Calor de Demanda y Paradas Sugeridas\n{municipios_zmg}", color='white', fontsize=15)
plt.legend(loc='upper right')
plt.show()

import networkx as nx
import random

print("1. Preparando el entorno para las hormigas...")


nodos_paradas = ox.nearest_nodes(G, paradas_optimas[:, 0], paradas_optimas[:, 1])

print(f"Paradas ajustadas a la red vial: {len(nodos_paradas)} nodos.")

# 2. Crear la MATRIZ DE DISTANCIAS (El tablero de juego)

num_paradas = len(nodos_paradas)
dist_matrix = np.zeros((num_paradas, num_paradas))

print("Calculando matriz de distancias reales (Dijkstra)...")
for i in range(num_paradas):
    for j in range(num_paradas):
        if i != j:
            # Calculamos la distancia más corta en la red vial entre la parada i y j
            try:
                length = nx.shortest_path_length(G, nodos_paradas[i], nodos_paradas[j], weight='length')
                dist_matrix[i][j] = length
            except nx.NetworkXNoPath:
                dist_matrix[i][j] = 999999 # Penalización si no hay camino (calle cerrada/sentido contrario)

print("Matriz calculada")

class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):

        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=all_time_shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path

            # Opcional: Imprimir progreso cada 10 iteraciones
            if i % 10 == 0:
                print(f"Iteración {i}: Mejor costo actual = {int(all_time_shortest_path[1])} metros")

        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        # Evaporación
        self.pheromone * (1 - self.decay)
        # Depósito de nueva feromona solo por las mejores hormigas
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0) # Todas empiezan en la parada 0 (arbitrario, es un ciclo)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # Volver al inicio (Ruta Circular)
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0 # No volver a visitar lo ya visitado


        row = pheromone ** self.alpha * (( 1.0 / (dist + 0.0001)) ** self.beta)

        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

print("2. Liberando a las hormigas...")

# Configuración del ACO
# - 20 Hormigas
# - 50 Iteraciones
# - Evaporación rápida (0.95) para fomentar exploración
ant_colony = AntColonyOptimizer(dist_matrix, n_ants=20, n_best=5, n_iterations=50, decay=0.95, alpha=1, beta=2)

ruta_optima_indices, distancia_total = ant_colony.run()

print(f"¡Optimización terminada! Distancia total estimada: {distancia_total:.2f} metros.")
print("Secuencia de paradas (IDs):", ruta_optima_indices)


print("3. Reconstruyendo la ruta calle por calle...")

ruta_completa_nodos = []

# Iteramos sobre los pares de la solución (origen -> destino)
for u, v in ruta_optima_indices:
    nodo_origen = nodos_paradas[u]
    nodo_destino = nodos_paradas[v]

    # Calculamos el camino real calle por calle entre estas dos paradas
    camino_parcial = nx.shortest_path(G, nodo_origen, nodo_destino, weight='length')

    # Agregamos al recorrido total (quitamos el último para no duplicar nodos al unir)
    ruta_completa_nodos.extend(camino_parcial[:-1])

# Agregamos el nodo final para cerrar el ciclo
ruta_completa_nodos.append(nodos_paradas[ruta_optima_indices[-1][1]])

print("¡Ruta reconstruida lista para visualizar!")

# Graficamos la ruta óptima sobre el mapa
fig, ax = ox.plot_graph_route(G, ruta_completa_nodos,
                              route_color='r', route_linewidth=4, route_alpha=0.8,
                              node_size=0, show=False, close=False, figsize=(20,20))

# Dibujamos también las paradas encima para ver que sí pasa por ellas
x_paradas = [G.nodes[n]['x'] for n in nodos_paradas]
y_paradas = [G.nodes[n]['y'] for n in nodos_paradas]

ax.scatter(x_paradas, y_paradas, c='cyan', s=100, marker='o', zorder=5, label='Paradas')

plt.title("Ruta de Transporte Público Optimizada (ACO + K-Means)", color='white', fontsize=18) # Slightly larger title font
plt.legend()
plt.show()