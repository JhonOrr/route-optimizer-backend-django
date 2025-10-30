import numpy as np
import math
from typing import List, Tuple, Dict

def haversine_distance(lat1, lon1, lat2, lon2):
    # Radio de la Tierra en km
    R = 6371.0
    
    # Convertir grados a radianes
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Diferencias
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Fórmula de Haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c  # Distancia en km


class ACOVRPPD_MultiVehicle_Dynamic:
    def __init__(self, num_ants, iterations, evaporation_rate, alpha, beta, vehicles, nodes):
        """
        Inicializa el algoritmo ACO para VRP con recogida y entrega en entorno dinámico
        
        Args:
            num_ants: Número de hormigas
            iterations: Número de iteraciones
            evaporation_rate: Tasa de evaporación de feromonas
            alpha: Peso de las feromonas
            beta: Peso de la visibilidad
            vehicles: Lista de vehículos con estado inicial
                    [(capacidad_actual, capacidad_maxima, distancia_recorrida, distancia_maxima, lat_inicial, lon_inicial), ...]
            nodes: Lista de nodos [('tipo', lat, lon, cambio_capacidad), ...]
        """
        self.num_ants = num_ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.vehicles = vehicles
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.vehicle_start_positions = self._create_vehicle_start_positions()
        self.pd_pairs = self._create_pd_pairs()
        self.distances = self._compute_distances()

        total_points = self.num_nodes + len(self.vehicle_start_positions)
        self.pheromone = np.ones((total_points, total_points)) * 0.1

    def _create_vehicle_start_positions(self):
        """Crea posiciones iniciales para cada vehículo como nodos virtuales"""
        start_positions = []
        for i, vehicle in enumerate(self.vehicles):
            # Cada vehículo tiene su propia posición inicial (nodo virtual)
            start_positions.append((
                f'vehicle_{i}',
                vehicle[4],  # lat_inicial
                vehicle[5],  # lon_inicial
                0  # Sin cambio de capacidad
            ))
        return start_positions

    def _create_pd_pairs(self):
        """Crea pares pickup-delivery, considerando que algunos deliveries no tienen pickup"""
        pd_pairs = {}
        pickup_nodes = {}
        
        # Primero identificamos todos los pickups
        for i, node in enumerate(self.nodes):
            if node[0] == 'pickup':
                # Asumimos que el pickup tiene un ID que coincide con su delivery
                # En implementación real, esto debería venir de los datos
                order_id = f"order_{i}"  # Esto debería ser más robusto en producción
                pickup_nodes[order_id] = i
        
        # Luego emparejamos con deliveries
        for i, node in enumerate(self.nodes):
            if node[0] == 'delivery':
                # Buscar el pickup correspondiente
                # En implementación real, usar un ID de orden compartido
                order_id = f"order_{i-1}"  # Asumiendo que el pickup precede al delivery
                if order_id in pickup_nodes:
                    pd_pairs[pickup_nodes[order_id]] = i
        
        return pd_pairs

    def _compute_distances(self):
        """Calcula matriz de distancias incluyendo posiciones iniciales de vehículos"""
        # Incluir posiciones de vehículos en la matriz de distancias
        all_points = self.nodes + self.vehicle_start_positions
        num_all_points = len(all_points)
        
        distances = np.zeros((num_all_points, num_all_points))
        for i in range(num_all_points):
            for j in range(num_all_points):
                if i != j:
                    lat1, lon1 = all_points[i][1], all_points[i][2]
                    lat2, lon2 = all_points[j][1], all_points[j][2]
                    distances[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
        return distances

    def _get_node_index_offset(self, vehicle_index):
        """Calcula el offset para índices de nodos basado en posiciones de vehículos"""
        return len(self.nodes) + vehicle_index

    def _validate_move(self, vehicle_routes, current_vehicle, next_node, current_load, current_distance, vehicle_index):
        """
        Valida si un movimiento es permitido considerando estado actual del vehículo
        """
        # Validar capacidad
        node_capacity_change = self.nodes[next_node][3]
        new_load = current_load + node_capacity_change
        
        if new_load < 0 or new_load > self.vehicles[current_vehicle][1]:  # capacidad_maxima
            return False
        
        # Validar precedencia pickup-delivery
        if self.nodes[next_node][0] == 'delivery':
            # Buscar si este delivery tiene un pickup asociado
            pickup_node = None
            for pickup, delivery in self.pd_pairs.items():
                if delivery == next_node:
                    pickup_node = pickup
                    break
            
            if pickup_node is not None and pickup_node not in vehicle_routes[current_vehicle]:
                return False
        
        # Validar distancia máxima
        last_node = vehicle_routes[current_vehicle][-1]
        
        # Si el último nodo es una posición inicial de vehículo, usar índice offset
        if last_node >= len(self.nodes):
            actual_last_node = last_node - len(self.nodes)
            dist = self.distances[actual_last_node][next_node]
        else:
            dist = self.distances[last_node][next_node]
        
        estimated_total = current_distance + dist
        
        # Añadir distancia de regreso al depósito para estimación
        return_to_depot = self.distances[next_node][0]  # índice 0 es el depósito
        estimated_total += return_to_depot
        
        if estimated_total > self.vehicles[current_vehicle][3]:  # distancia_maxima
            return False
        
        return True

    def _select_next_vehicle_and_node(self, vehicle_routes, unvisited, vehicle_distances, vehicle_loads):
        """Selecciona el próximo vehículo y nodo usando reglas ACO"""
        valid_moves = []
        
        for vehicle_idx in range(len(self.vehicles)):
            if not vehicle_routes[vehicle_idx]:  # Vehículo sin ruta
                continue
                
            current_route = vehicle_routes[vehicle_idx]
            current_load = vehicle_loads[vehicle_idx]
            current_dist = vehicle_distances[vehicle_idx]
            last_node = current_route[-1]
            
            for node in unvisited:
                if self._validate_move(vehicle_routes, vehicle_idx, node, current_load, current_dist, vehicle_idx):
                    # Calcular probabilidad
                    pheromone = self.pheromone[last_node][node]
                    visibility = 1 / (self.distances[last_node][node] + 1e-6)
                    prob = (pheromone ** self.alpha) * (visibility ** self.beta)
                    valid_moves.append((vehicle_idx, node, prob))
        
        if not valid_moves:
            return None, None
        
        # Selección probabilística
        total_prob = sum(p for _, _, p in valid_moves)
        probabilities = [p / total_prob for _, _, p in valid_moves]
        idx = np.random.choice(len(valid_moves), p=probabilities)
        return valid_moves[idx][0], valid_moves[idx][1]

    def run(self):
        """Ejecuta el algoritmo ACO"""
        best_routes = None
        best_distance = float('inf')
        
        for iteration in range(self.iterations):
            for ant in range(self.num_ants):
                # Inicializar rutas: cada vehículo empieza en su posición inicial
                vehicle_routes = [[] for _ in range(len(self.vehicles))]
                vehicle_distances = [0.0] * len(self.vehicles)
                vehicle_loads = [vehicle[0] for vehicle in self.vehicles]  # capacidad_actual
                
                # Asignar posición inicial a cada vehículo
                for i in range(len(self.vehicles)):
                    start_node_index = len(self.nodes) + i  # Índice del nodo de inicio del vehículo
                    vehicle_routes[i].append(start_node_index)
                
                unvisited = set(range(len(self.nodes)))  # Solo nodos reales, no posiciones de vehículos
                
                # Construir rutas
                while unvisited:
                    vehicle, node = self._select_next_vehicle_and_node(
                        vehicle_routes, unvisited, vehicle_distances, vehicle_loads
                    )
                    
                    if vehicle is None:  # No hay movimientos válidos
                        break
                    
                    # Actualizar ruta y métricas
                    last_node = vehicle_routes[vehicle][-1]
                    new_dist = self.distances[last_node][node]
                    
                    vehicle_routes[vehicle].append(node)
                    vehicle_distances[vehicle] += new_dist
                    vehicle_loads[vehicle] += self.nodes[node][3]  # Actualizar carga
                    unvisited.remove(node)
                
                # Cerrar rutas: regresar al depósito
                total_distance = 0
                for i in range(len(vehicle_routes)):
                    if len(vehicle_routes[i]) > 1:  # Si el vehículo visitó algún nodo
                        last_node = vehicle_routes[i][-1]
                        return_dist = self.distances[last_node][0]  # Al depósito
                        vehicle_routes[i].append(0)  # Depósito
                        vehicle_distances[i] += return_dist
                        total_distance += vehicle_distances[i]
                
                # Validar y actualizar mejor solución
                if total_distance < best_distance and self._validate_routes(vehicle_routes, vehicle_loads):
                    best_distance = total_distance
                    best_routes = [route.copy() for route in vehicle_routes]
            
            # Actualizar feromonas
            if best_routes is not None:
                self._update_pheromones(best_routes, best_distance)
        
        return best_routes, best_distance

    def _validate_routes(self, vehicle_routes, final_loads):
        """Valida que las rutas cumplan todas las restricciones"""
        # Validar capacidad final
        for i, load in enumerate(final_loads):
            if load < 0 or load > self.vehicles[i][1]:
                return False
        
        # Validar pares pickup-delivery
        for pickup, delivery in self.pd_pairs.items():
            pickup_found = False
            delivery_found = False
            same_vehicle = False
            
            for vehicle_idx, route in enumerate(vehicle_routes):
                if pickup in route:
                    pickup_found = True
                    pickup_vehicle = vehicle_idx
                    pickup_position = route.index(pickup)
                
                if delivery in route:
                    delivery_found = True
                    delivery_vehicle = vehicle_idx
                    delivery_position = route.index(delivery)
            
            if not pickup_found or not delivery_found:
                return False
            
            if pickup_vehicle != delivery_vehicle:
                return False
            
            if pickup_position >= delivery_position:
                return False
        
        return True

    def _update_pheromones(self, best_routes, best_distance):
        """Actualiza la matriz de feromonas"""
        # Evaporación
        self.pheromone *= (1 - self.evaporation_rate)
        
        # Depositar feromonas en las mejores rutas
        for route in best_routes:
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                
                # Verificar que los índices sean válidos
                if from_node < self.pheromone.shape[0] and to_node < self.pheromone.shape[1]:
                    self.pheromone[from_node][to_node] += 1 / best_distance

# Ejemplo de uso
if __name__ == "__main__":
    # Definir nodos (incluyendo depot)
    nodes = [
        ('depot', 40.7128, -74.0060, 0),      # Depósito (índice 0)
        ('pickup', 40.7138, -74.0060, 10),    # Recojo 1
        ('delivery', 40.7148, -74.0060, -10), # Entrega 1  
        ('delivery', 40.7200, -74.0100, -5),  # Entrega sin pickup (ya recogido)
        ('pickup', 40.7533, -73.9829, 15),    # Recojo 2
        ('delivery', 40.7504, -73.9897, -15), # Entrega 2
    ]

    # Definir vehículos: (capacidad_actual, capacidad_maxima, distancia_recorrida, distancia_maxima, lat, lon)
    vehicles = [
        (25, 50, 20, 200, 40.7100, -74.0050),  # Vehículo 1
        (10, 50, 15, 180, 40.7150, -74.0080),  # Vehículo 2
    ]

    # Crear y ejecutar algoritmo
    aco = ACOVRPPD_MultiVehicle_Dynamic(
        num_ants=10,
        iterations=100,
        evaporation_rate=0.1,
        alpha=1.0,
        beta=2.0,
        vehicles=vehicles,
        nodes=nodes
    )

    routes, distance = aco.run()
    print("Mejores rutas:", routes)
    print("Distancia total:", distance)