import numpy as np
import math

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


class ACOVRPPD_MultiVehicle:
    def __init__(self, num_ants, iterations, evaporation_rate, alpha, beta, vehicles, nodes):
        self.num_ants = num_ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.vehicles = vehicles  # Lista de tuplas: [(capacidad, distancia_max), ...]
        self.nodes = nodes  # Formato: [('depot', 0, 0, 0), ('pickup', 2, 3, 1), ('delivery', 5, 4, -1), ...]
        self.num_nodes = len(nodes)
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * 0.1
        self.distances = self._compute_distances()
        self.pd_pairs = self._create_pd_pairs()  # Diccionario {pickup_id: delivery_id}

    def _create_pd_pairs(self):
        pd_pairs = {}
        for i, node in enumerate(self.nodes):
            if node[0] == 'pickup':
                delivery_id = i + 1
                if delivery_id < self.num_nodes and self.nodes[delivery_id][0] == 'delivery':
                    pd_pairs[i] = delivery_id
        return pd_pairs

    def _compute_distances(self):
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # Obtener coordenadas geográficas
                    lat1, lon1 = self.nodes[i][1], self.nodes[i][2]
                    lat2, lon2 = self.nodes[j][1], self.nodes[j][2]
                    
                    distances[i][j] = haversine_distance(lat1, lon1, lat2, lon2)
        return distances

    def _validate_move(self, vehicle_routes, current_vehicle, next_node, current_load, current_distance):
        # Validar capacidad
        new_load = current_load + self.nodes[next_node][3]
        if new_load < 0 or new_load > self.vehicles[current_vehicle][0]:
            return False
        
        # Validar precedencia (si es delivery, debe tener su pickup en la ruta)
        if self.nodes[next_node][0] == 'delivery':
            pickup_node = self.pd_pairs_inv.get(next_node)
            if pickup_node is None or pickup_node not in vehicle_routes[current_vehicle]:
                return False
        
        # Validar distancia: distancia actual + al nuevo nodo + regreso al depósito
        last_node = vehicle_routes[current_vehicle][-1]
        new_segment = self.distances[last_node][next_node]
        return_to_depot = self.distances[next_node][0]
        estimated_total = current_distance + new_segment + return_to_depot
        if estimated_total > self.vehicles[current_vehicle][1]:
            return False
        
        return True

    def _select_next_vehicle_and_node(self, vehicle_routes, unvisited, vehicle_distances):
        valid_moves = []
        for vehicle in range(len(self.vehicles)):
            current_route = vehicle_routes[vehicle]
            current_load = sum(self.nodes[node][3] for node in current_route)
            current_node = current_route[-1]
            current_dist = vehicle_distances[vehicle]
            
            for node in unvisited:
                if self._validate_move(vehicle_routes, vehicle, node, current_load, current_dist):
                    pheromone = self.pheromone[current_node][node]
                    visibility = 1 / (self.distances[current_node][node] + 1e-6)
                    prob = (pheromone ** self.alpha) * (visibility ** self.beta)
                    valid_moves.append((vehicle, node, prob))
        
        if not valid_moves:
            return None, None
        
        total = sum(p for _, _, p in valid_moves)
        probabilities = [p / total for _, _, p in valid_moves]
        idx = np.random.choice(len(valid_moves), p=probabilities)
        return valid_moves[idx][0], valid_moves[idx][1]

    def run(self):
        best_routes = None
        best_distance = float('inf')
        self.pd_pairs_inv = {v: k for k, v in self.pd_pairs.items()}  # Inversa para validación

        for _ in range(self.iterations):
            for ant in range(self.num_ants):
                # Inicializar rutas: cada vehículo empieza en el depósito
                vehicle_routes = [[0] for _ in range(len(self.vehicles))]
                vehicle_distances = [0] * len(self.vehicles)  # Distancia acumulada por vehículo
                unvisited = set(range(1, self.num_nodes))  # Todos excepto depósito
                
                while unvisited:
                    vehicle, node = self._select_next_vehicle_and_node(vehicle_routes, unvisited, vehicle_distances)
                    if vehicle is None:  # No hay movimientos válidos
                        break
                    
                    # Actualizar ruta y distancia
                    last_node = vehicle_routes[vehicle][-1]
                    new_dist = self.distances[last_node][node]
                    vehicle_routes[vehicle].append(node)
                    vehicle_distances[vehicle] += new_dist
                    unvisited.remove(node)
                
                # Cerrar rutas: regresar al depósito si el vehículo se usó
                for i in range(len(vehicle_routes)):
                    if len(vehicle_routes[i]) > 1:  # Si tiene nodos además del depósito
                        last_node = vehicle_routes[i][-1]
                        return_dist = self.distances[last_node][0]
                        vehicle_routes[i].append(0)
                        vehicle_distances[i] += return_dist
                
                # Calcular distancia total y validar solución
                total_distance = sum(vehicle_distances)
                if total_distance < best_distance and self._validate_routes(vehicle_routes, vehicle_distances):
                    best_distance = total_distance
                    best_routes = vehicle_routes
                
                # Reiniciar para la siguiente hormiga
                vehicle_routes = None
                vehicle_distances = None
            
            # Actualización global de feromonas
            if best_routes is not None:
                self._update_pheromones(best_routes, best_distance)
                
        return best_routes, best_distance

    def _validate_routes(self, vehicle_routes, vehicle_distances):
        # Validar pares pickup-delivery
        for pickup, delivery in self.pd_pairs.items():
            found = False
            for route in vehicle_routes:
                if pickup in route and delivery in route:
                    if route.index(pickup) > route.index(delivery):
                        return False  # Delivery antes que pickup
                    found = True
            if not found:
                return False  # Par no encontrado en la misma ruta
        
        # Validar distancias máximas
        for i, dist in enumerate(vehicle_distances):
            if dist > self.vehicles[i][1]:
                return False
        
        return True

    def _update_pheromones(self, best_routes, best_distance):
        # Evaporación
        self.pheromone *= (1 - self.evaporation_rate)
        # Depositar feromonas en la mejor ruta
        for route in best_routes:
            for i in range(len(route) - 1):
                from_node, to_node = route[i], route[i+1]
                self.pheromone[from_node][to_node] += 1 / best_distance
    