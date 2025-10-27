import numpy as np
import math
from typing import List, Dict, Tuple, Optional

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia haversine entre dos puntos geográficos"""
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


class VRPEnvironment:
    """
    Ambiente de Reinforcement Learning para el problema de reasignación dinámica de rutas.
    """
    
    def __init__(self, depot: Dict, vehicles: List[Dict], current_routes: List[Dict]):
        """
        Args:
            depot: Diccionario con lat/lng del depósito
            vehicles: Lista de vehículos disponibles con capacidad y max_distance
            current_routes: Rutas actuales en formato JSON (del algoritmo ACO)
        """
        self.depot = depot
        self.vehicles = vehicles
        self.current_routes = current_routes
        
        # Estado interno
        self.vehicle_loads = {}  # {vehicle_id: carga_actual}
        self.vehicle_distances = {}  # {vehicle_id: distancia_recorrida}
        self.pending_orders = []  # Órdenes pendientes de asignación
        self.completed_orders = []  # Órdenes completadas
        
        # Inicializar estado
        self._initialize_state()
    
    def _initialize_state(self):
        """Inicializa el estado del ambiente basado en las rutas actuales"""
        for route in self.current_routes:
            vehicle_id = route['vehicle_id']
            self.vehicle_loads[vehicle_id] = 0
            self.vehicle_distances[vehicle_id] = 0
            
            # Calcular carga y distancia actual
            for stop in route['stops']:
                if stop['type'] in ['pickup', 'delivery']:
                    self.vehicle_loads[vehicle_id] += stop.get('demand', 0)
            
            self.vehicle_distances[vehicle_id] = route['total_distance']
    
    def get_state(self) -> np.ndarray:
        """
        Retorna el estado actual del ambiente como vector de características.
        
        Estado incluye:
        - Capacidad disponible de cada vehículo (normalizada)
        - Distancia restante de cada vehículo (normalizada)
        - Número de órdenes pendientes (normalizada)
        - Características de la orden actual a asignar
        """
        state_vector = []
        
        # Características de vehículos
        for vehicle in self.vehicles:
            vehicle_id = vehicle['id']
            capacity = vehicle['capacity']
            max_distance = vehicle['max_distance']
            
            current_load = self.vehicle_loads.get(vehicle_id, 0)
            current_distance = self.vehicle_distances.get(vehicle_id, 0)
            
            # Normalizar valores [0, 1]
            available_capacity = max(0, (capacity - current_load) / capacity)
            available_distance = max(0, (max_distance - current_distance) / max_distance)
            
            state_vector.extend([available_capacity, available_distance])
        
        # Características globales
        num_pending = len(self.pending_orders) / 100.0  # Normalizar
        state_vector.append(min(1.0, num_pending))
        
        # Si hay orden pendiente, agregar sus características
        if self.pending_orders:
            order = self.pending_orders[0]
            weight_normalized = order.get('weight', 0) / 1000.0  # Normalizar peso
            state_vector.append(min(1.0, weight_normalized))
        else:
            state_vector.append(0.0)
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """Retorna el tamaño del vector de estado"""
        # 2 características por vehículo + 2 características globales
        return len(self.vehicles) * 2 + 2
    
    def get_action_size(self) -> int:
        """Retorna el número de acciones posibles (número de vehículos + 1 para no asignar)"""
        return len(self.vehicles) + 1
    
    def step(self, action: int, order: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción en el ambiente.
        
        Args:
            action: Índice del vehículo al que asignar la orden (o len(vehicles) para no asignar)
            order: Orden a asignar
            
        Returns:
            Tuple con (nuevo_estado, recompensa, done, info)
        """
        reward = 0
        info = {'assigned': False, 'feasible': False, 'vehicle_id': None}
        
        # Acción: no asignar (dejar pendiente)
        if action == len(self.vehicles):
            self.pending_orders.append(order)
            reward = -1.0  # Penalización leve por no asignar
            info['assigned'] = False
        else:
            # Intentar asignar a vehículo
            vehicle = self.vehicles[action]
            vehicle_id = vehicle['id']
            
            # Verificar factibilidad
            feasible, insertion_cost = self._check_feasibility(vehicle_id, order)
            
            if feasible:
                # Asignar orden al vehículo
                self._assign_order_to_vehicle(vehicle_id, order)
                
                # Recompensa basada en eficiencia
                reward = self._calculate_reward(insertion_cost, order)
                info['assigned'] = True
                info['feasible'] = True
                info['vehicle_id'] = vehicle_id
                self.completed_orders.append(order)
            else:
                # Orden no factible para este vehículo
                self.pending_orders.append(order)
                reward = -5.0  # Penalización mayor por asignación inválida
                info['assigned'] = False
                info['feasible'] = False
        
        # Obtener nuevo estado
        next_state = self.get_state()
        
        # Episodio termina cuando no hay más órdenes pendientes
        done = len(self.pending_orders) == 0
        
        return next_state, reward, done, info
    
    def _check_feasibility(self, vehicle_id: int, order: Dict) -> Tuple[bool, float]:
        """
        Verifica si una orden puede ser asignada a un vehículo.
        
        Returns:
            Tuple (es_factible, costo_insercion)
        """
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if not vehicle:
            return False, float('inf')
        
        # Verificar capacidad
        order_weight = order.get('weight', 0)
        current_load = self.vehicle_loads.get(vehicle_id, 0)
        
        if current_load + order_weight > vehicle['capacity']:
            return False, float('inf')
        
        # Calcular costo de inserción (distancia adicional)
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if not route:
            return False, float('inf')
        
        pickup_location = order['pickupAddress']
        delivery_location = order['deliveryAddress']
        
        # Calcular mejor posición de inserción
        best_cost = float('inf')
        stops = route['stops']
        
        # Intentar insertar pickup y delivery en todas las posiciones válidas
        for i in range(1, len(stops)):
            for j in range(i + 1, len(stops) + 1):
                # Insertar pickup en posición i, delivery en posición j
                insertion_cost = self._calculate_insertion_cost(
                    stops, i, j, pickup_location, delivery_location
                )
                
                if insertion_cost < best_cost:
                    best_cost = insertion_cost
        
        # Verificar restricción de distancia máxima
        current_distance = self.vehicle_distances.get(vehicle_id, 0)
        new_distance = current_distance + best_cost
        
        if new_distance > vehicle['max_distance']:
            return False, best_cost
        
        return True, best_cost
    
    def _calculate_insertion_cost(self, stops: List[Dict], pickup_pos: int, 
                                   delivery_pos: int, pickup_loc: Dict, 
                                   delivery_loc: Dict) -> float:
        """Calcula el costo de insertar pickup y delivery en posiciones específicas"""
        # Obtener coordenadas de paradas adyacentes
        prev_pickup = stops[pickup_pos - 1]['location']
        next_pickup = stops[pickup_pos]['location'] if pickup_pos < len(stops) else self.depot
        
        prev_delivery = stops[delivery_pos - 1]['location']
        next_delivery = stops[delivery_pos]['location'] if delivery_pos < len(stops) else self.depot
        
        # Costo de insertar pickup
        cost_pickup_remove = haversine_distance(
            prev_pickup['lat'], prev_pickup['lng'],
            next_pickup['lat'], next_pickup['lng']
        )
        
        cost_pickup_add = (
            haversine_distance(
                prev_pickup['lat'], prev_pickup['lng'],
                pickup_loc['latitude'], pickup_loc['longitude']
            ) +
            haversine_distance(
                pickup_loc['latitude'], pickup_loc['longitude'],
                next_pickup['lat'], next_pickup['lng']
            )
        )
        
        # Costo de insertar delivery
        cost_delivery_remove = haversine_distance(
            prev_delivery['lat'], prev_delivery['lng'],
            next_delivery['lat'], next_delivery['lng']
        )
        
        cost_delivery_add = (
            haversine_distance(
                prev_delivery['lat'], prev_delivery['lng'],
                delivery_loc['latitude'], delivery_loc['longitude']
            ) +
            haversine_distance(
                delivery_loc['latitude'], delivery_loc['longitude'],
                next_delivery['lat'], next_delivery['lng']
            )
        )
        
        return (cost_pickup_add - cost_pickup_remove) + (cost_delivery_add - cost_delivery_remove)
    
    def _assign_order_to_vehicle(self, vehicle_id: int, order: Dict):
        """Asigna una orden a un vehículo y actualiza el estado"""
        # Actualizar carga del vehículo
        order_weight = order.get('weight', 0)
        self.vehicle_loads[vehicle_id] = self.vehicle_loads.get(vehicle_id, 0) + order_weight
        
        # Actualizar ruta del vehículo (simplificado, en producción necesitaría re-optimización local)
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if route:
            # Encontrar mejor posición de inserción y actualizar
            pickup_stop = {
                'type': 'pickup',
                'location': {
                    'lat': order['pickupAddress']['latitude'],
                    'lng': order['pickupAddress']['longitude']
                },
                'demand': order_weight,
                'order_id': order['id'],
                'customer': order.get('customer')
            }
            
            delivery_stop = {
                'type': 'delivery',
                'location': {
                    'lat': order['deliveryAddress']['latitude'],
                    'lng': order['deliveryAddress']['longitude']
                },
                'demand': -order_weight,
                'order_id': order['id'],
                'customer': order.get('customer')
            }
            
            # Insertar en las mejores posiciones (simplificado)
            route['stops'].insert(-1, pickup_stop)
            route['stops'].insert(-1, delivery_stop)
            
            # Recalcular distancia total
            total_distance = 0
            for i in range(len(route['stops']) - 1):
                stop1 = route['stops'][i]['location']
                stop2 = route['stops'][i + 1]['location']
                total_distance += haversine_distance(
                    stop1['lat'], stop1['lng'],
                    stop2['lat'], stop2['lng']
                )
            
            route['total_distance'] = total_distance
            self.vehicle_distances[vehicle_id] = total_distance
    
    def _calculate_reward(self, insertion_cost: float, order: Dict) -> float:
        """
        Calcula la recompensa por asignar una orden.
        
        Factores considerados:
        - Costo de inserción (menor es mejor)
        - Utilización de capacidad del vehículo
        - Urgencia de la orden (si existe)
        """
        # Recompensa base por asignación exitosa
        reward = 10.0
        
        # Penalización por costo de inserción (normalizado)
        insertion_penalty = min(5.0, insertion_cost / 10.0)
        reward -= insertion_penalty
        
        # Bonus por eficiencia en utilización de capacidad
        # (Esto incentiva a llenar vehículos antes de usar nuevos)
        order_weight = order.get('weight', 0)
        efficiency_bonus = (order_weight / 1000.0) * 2.0
        reward += efficiency_bonus
        
        return reward
    
    def reset(self, new_routes: List[Dict] = None):
        """Reinicia el ambiente con nuevas rutas"""
        if new_routes:
            self.current_routes = new_routes
        
        self.vehicle_loads = {}
        self.vehicle_distances = {}
        self.pending_orders = []
        self.completed_orders = []
        
        self._initialize_state()
        
        return self.get_state()
    
    def add_new_order(self, order: Dict):
        """Agrega una nueva orden al sistema"""
        self.pending_orders.append(order)
    
    def cancel_order(self, order_id: int):
        """Cancela una orden existente"""
        # Remover de órdenes pendientes
        self.pending_orders = [o for o in self.pending_orders if o.get('id') != order_id]
        
        # Remover de rutas asignadas
        for route in self.current_routes:
            original_stops = route['stops']
            route['stops'] = [s for s in original_stops if s.get('order_id') != order_id]
            
            # Recalcular distancia si se removieron paradas
            if len(route['stops']) != len(original_stops):
                vehicle_id = route['vehicle_id']
                total_distance = 0
                
                for i in range(len(route['stops']) - 1):
                    stop1 = route['stops'][i]['location']
                    stop2 = route['stops'][i + 1]['location']
                    total_distance += haversine_distance(
                        stop1['lat'], stop1['lng'],
                        stop2['lat'], stop2['lng']
                    )
                
                route['total_distance'] = total_distance
                self.vehicle_distances[vehicle_id] = total_distance
                
                # Recalcular carga
                new_load = sum(s.get('demand', 0) for s in route['stops'] if s['type'] == 'pickup')
                self.vehicle_loads[vehicle_id] = new_load
    
    def remove_vehicle(self, vehicle_id: int) -> List[Dict]:
        """
        Remueve un vehículo y retorna las órdenes que deben ser reasignadas.
        
        Returns:
            Lista de órdenes que estaban asignadas al vehículo removido
        """
        orders_to_reassign = []
        
        # Encontrar ruta del vehículo
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        
        if route:
            # Extraer órdenes únicas de la ruta
            order_ids_seen = set()
            for stop in route['stops']:
                if stop.get('order_id') and stop['order_id'] not in order_ids_seen:
                    order_ids_seen.add(stop['order_id'])
                    # Reconstruir orden desde las paradas
                    order = {
                        'id': stop['order_id'],
                        'weight': abs(stop.get('demand', 0)),
                        'customer': stop.get('customer'),
                    }
                    
                    # Encontrar pickup y delivery para esta orden
                    pickup_stop = next((s for s in route['stops'] 
                                       if s.get('order_id') == stop['order_id'] 
                                       and s['type'] == 'pickup'), None)
                    delivery_stop = next((s for s in route['stops'] 
                                         if s.get('order_id') == stop['order_id'] 
                                         and s['type'] == 'delivery'), None)
                    
                    if pickup_stop and delivery_stop:
                        order['pickupAddress'] = {
                            'latitude': pickup_stop['location']['lat'],
                            'longitude': pickup_stop['location']['lng']
                        }
                        order['deliveryAddress'] = {
                            'latitude': delivery_stop['location']['lat'],
                            'longitude': delivery_stop['location']['lng']
                        }
                        
                        orders_to_reassign.append(order)
            
            # Remover ruta de current_routes
            self.current_routes = [r for r in self.current_routes if r['vehicle_id'] != vehicle_id]
            
            # Remover vehículo de la lista de vehículos
            self.vehicles = [v for v in self.vehicles if v['id'] != vehicle_id]
            
            # Limpiar estado del vehículo
            if vehicle_id in self.vehicle_loads:
                del self.vehicle_loads[vehicle_id]
            if vehicle_id in self.vehicle_distances:
                del self.vehicle_distances[vehicle_id]
        
        return orders_to_reassign