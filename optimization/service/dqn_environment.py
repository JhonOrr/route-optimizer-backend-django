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
    Ambiente de Reinforcement Learning para reasignación dinámica de rutas.
    Considera estados de órdenes y posiciones actuales de vehículos.
    """
    
    def __init__(self, depot: Dict, vehicles: List[Dict], current_routes: List[Dict]):
        """
        Args:
            depot: Diccionario con lat/lng del depósito
            vehicles: Lista de vehículos con capacidad, max_distance y posición actual
            current_routes: Rutas actuales en formato JSON (del algoritmo ACO)
        """
        self.depot = depot
        self.vehicles = vehicles
        self.current_routes = current_routes
        
        # Estado interno
        self.vehicle_loads = {}
        self.vehicle_distances = {}
        self.vehicle_positions = {}  # Nueva: posición actual de cada vehículo
        self.pending_orders = []
        self.completed_orders = []
        
        self._initialize_state()
    
    def _initialize_state(self):
        """Inicializa el estado del ambiente incluyendo posiciones actuales"""
        for route in self.current_routes:
            vehicle_id = route['vehicle_id']
            self.vehicle_loads[vehicle_id] = 0
            self.vehicle_distances[vehicle_id] = 0
            
            # Calcular carga actual
            for stop in route['stops']:
                if stop['type'] in ['pickup', 'delivery']:
                    self.vehicle_loads[vehicle_id] += stop.get('demand', 0)
            
            self.vehicle_distances[vehicle_id] = route['total_distance']
            
            # Determinar posición actual del vehículo (última parada no-depot)
            last_position = self.depot.copy()
            for stop in route['stops']:
                if stop['type'] != 'depot':
                    last_position = stop['location']
            
            self.vehicle_positions[vehicle_id] = last_position
    
    def get_pending_stops_for_order(self, order: Dict) -> List[Dict]:
        """
        Retorna las paradas pendientes según el estado de la orden.
        
        Estados desde Spring Boot:
        - 'pendiente' o '1': Incluir pickup y delivery
        - 'recogido' o '2': Solo delivery
        - 'completo' o '3': Ninguna
        - 'cancelada' o '4': Ninguna
        - 'pospuesta' o '5': Ninguna (no se reasigna)
        """
        status = str(order.get('status', 'pendiente')).lower()
        stops = []
        
        if status in ['pendiente', '1']:
            # Orden pendiente: necesita pickup y delivery
            stops.append({
                'type': 'pickup',
                'location': order['pickupAddress'],
                'demand': order.get('weight', 0)
            })
            stops.append({
                'type': 'delivery',
                'location': order['deliveryAddress'],
                'demand': -order.get('weight', 0)
            })
        elif status in ['recogido', '2']:
            # Orden recogida: solo falta delivery
            stops.append({
                'type': 'delivery',
                'location': order['deliveryAddress'],
                'demand': -order.get('weight', 0)
            })
        # Para 'completo' (3), 'cancelada' (4) o 'pospuesta' (5), no se agregan paradas
        
        return stops
    
    def get_state(self) -> np.ndarray:
        """
        Retorna el estado actual del ambiente como vector de características.
        
        Estado incluye:
        - Capacidad disponible de cada vehículo (normalizada)
        - Distancia restante de cada vehículo (normalizada)
        - Distancia desde posición actual al depósito (normalizada)
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
            current_pos = self.vehicle_positions.get(vehicle_id, self.depot)
            
            # Normalizar valores [0, 1]
            available_capacity = max(0, (capacity - current_load) / capacity)
            available_distance = max(0, (max_distance - current_distance) / max_distance)
            
            # Distancia desde posición actual al depósito
            distance_to_depot = haversine_distance(
                current_pos['lat'], current_pos['lng'],
                self.depot['lat'], self.depot['lng']
            )
            normalized_distance_to_depot = min(1.0, distance_to_depot / max_distance)
            
            state_vector.extend([
                available_capacity, 
                available_distance,
                normalized_distance_to_depot
            ])
        
        # Características globales
        num_pending = len(self.pending_orders) / 100.0
        state_vector.append(min(1.0, num_pending))
        
        # Si hay orden pendiente, agregar sus características
        if self.pending_orders:
            order = self.pending_orders[0]
            weight_normalized = order.get('weight', 0) / 1000.0
            state_vector.append(min(1.0, weight_normalized))
            
            # Estado de la orden (one-hot encoding simplificado)
            status = str(order.get('status', 'pendiente')).lower()
            state_vector.append(1.0 if status in ['pendiente', '1'] else 0.0)
            state_vector.append(1.0 if status in ['recogido', '2'] else 0.0)
        else:
            state_vector.extend([0.0, 0.0, 0.0])
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_state_size(self) -> int:
        """Retorna el tamaño del vector de estado"""
        # 3 características por vehículo + 1 global + 3 de orden
        return len(self.vehicles) * 3 + 4
    
    def get_action_size(self) -> int:
        """Retorna el número de acciones posibles"""
        return len(self.vehicles) + 1
    
    def step(self, action: int, order: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Ejecuta una acción en el ambiente.
        
        Args:
            action: Índice del vehículo al que asignar la orden
            order: Orden a asignar
            
        Returns:
            Tuple con (nuevo_estado, recompensa, done, info)
        """
        reward = 0
        info = {'assigned': False, 'feasible': False, 'vehicle_id': None}
        
        # Filtrar paradas según estado de la orden
        pending_stops = self.get_pending_stops_for_order(order)
        
        if not pending_stops:
            # Orden completada o cancelada, no se asigna
            reward = -0.1
            info['assigned'] = False
            info['reason'] = 'order_not_actionable'
        elif action == len(self.vehicles):
            # No asignar (dejar pendiente)
            self.pending_orders.append(order)
            reward = -1.0
            info['assigned'] = False
        else:
            # Intentar asignar a vehículo
            vehicle = self.vehicles[action]
            vehicle_id = vehicle['id']
            
            # Verificar factibilidad
            feasible, insertion_cost = self._check_feasibility(vehicle_id, order, pending_stops)
            
            if feasible:
                # Asignar orden al vehículo
                self._assign_order_to_vehicle(vehicle_id, order, pending_stops)
                
                # Recompensa basada en eficiencia
                reward = self._calculate_reward(insertion_cost, order)
                info['assigned'] = True
                info['feasible'] = True
                info['vehicle_id'] = vehicle_id
                self.completed_orders.append(order)
            else:
                # Orden no factible
                self.pending_orders.append(order)
                reward = -5.0
                info['assigned'] = False
                info['feasible'] = False
        
        next_state = self.get_state()
        done = len(self.pending_orders) == 0
        
        return next_state, reward, done, info
    
    def _check_feasibility(self, vehicle_id: int, order: Dict, 
                          pending_stops: List[Dict]) -> Tuple[bool, float]:
        """
        Verifica si una orden puede ser asignada a un vehículo.
        Considera la posición actual del vehículo.
        """
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if not vehicle:
            return False, float('inf')
        
        # Verificar capacidad
        order_weight = sum(abs(stop['demand']) for stop in pending_stops if stop['type'] == 'pickup')
        current_load = self.vehicle_loads.get(vehicle_id, 0)
        
        if current_load + order_weight > vehicle['capacity']:
            return False, float('inf')
        
        # Obtener ruta actual
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if not route:
            return False, float('inf')
        
        # Obtener posición actual del vehículo
        current_position = self.vehicle_positions.get(vehicle_id, self.depot)
        
        # Calcular mejor posición de inserción desde la posición actual
        best_cost = float('inf')
        stops = route['stops']
        
        # Encontrar el índice de la posición actual en la ruta
        current_stop_idx = self._find_current_stop_index(route, current_position)
        
        # Solo considerar inserciones después de la posición actual
        for i in range(current_stop_idx, len(stops)):
            insertion_cost = self._calculate_insertion_cost_from_current(
                stops, i, pending_stops, current_position, current_stop_idx
            )
            
            if insertion_cost < best_cost:
                best_cost = insertion_cost
        
        # Verificar restricción de distancia máxima
        current_distance = self.vehicle_distances.get(vehicle_id, 0)
        new_distance = current_distance + best_cost
        
        if new_distance > vehicle['max_distance']:
            return False, best_cost
        
        return True, best_cost
    
    def _find_current_stop_index(self, route: Dict, current_position: Dict) -> int:
        """Encuentra el índice de la parada actual del vehículo"""
        stops = route['stops']
        for idx, stop in enumerate(stops):
            if (abs(stop['location']['lat'] - current_position['lat']) < 0.0001 and
                abs(stop['location']['lng'] - current_position['lng']) < 0.0001):
                return idx
        # Si no se encuentra, asumir que está en el primer stop
        return 0
    
    def _calculate_insertion_cost_from_current(self, stops: List[Dict], 
                                              insertion_idx: int,
                                              pending_stops: List[Dict],
                                              current_position: Dict,
                                              current_stop_idx: int) -> float:
        """
        Calcula el costo de insertar paradas pendientes considerando la posición actual.
        """
        total_cost = 0
        
        # Si estamos insertando después de la posición actual
        if insertion_idx <= current_stop_idx:
            return float('inf')  # No podemos insertar en el pasado
        
        prev_location = stops[insertion_idx - 1]['location'] if insertion_idx > 0 else current_position
        next_location = stops[insertion_idx]['location'] if insertion_idx < len(stops) else self.depot
        
        # Calcular costo de inserción para cada parada pendiente
        for pending_stop in pending_stops:
            stop_location = pending_stop['location']
            
            # Costo original (de prev a next directamente)
            original_cost = haversine_distance(
                prev_location['lat'], prev_location['lng'],
                next_location['lat'], next_location['lng']
            )
            
            # Nuevo costo (de prev a nueva parada, y de nueva parada a next)
            new_cost = (
                haversine_distance(
                    prev_location['lat'], prev_location['lng'],
                    stop_location['latitude'], stop_location['longitude']
                ) +
                haversine_distance(
                    stop_location['latitude'], stop_location['longitude'],
                    next_location['lat'], next_location['lng']
                )
            )
            
            total_cost += (new_cost - original_cost)
            
            # Actualizar prev_location para la siguiente parada
            prev_location = {
                'lat': stop_location['latitude'],
                'lng': stop_location['longitude']
            }
        
        return total_cost
    
    def _assign_order_to_vehicle(self, vehicle_id: int, order: Dict, 
                                 pending_stops: List[Dict]):
        """
        Asigna una orden a un vehículo y actualiza el estado.
        Inserta las paradas pendientes después de la posición actual.
        """
        # Actualizar carga del vehículo
        order_weight = sum(abs(stop['demand']) for stop in pending_stops if stop['type'] == 'pickup')
        self.vehicle_loads[vehicle_id] = self.vehicle_loads.get(vehicle_id, 0) + order_weight
        
        # Obtener ruta actual
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if not route:
            return
        
        # Obtener posición actual
        current_position = self.vehicle_positions.get(vehicle_id, self.depot)
        current_stop_idx = self._find_current_stop_index(route, current_position)
        
        # Crear paradas completas
        new_stops = []
        for pending_stop in pending_stops:
            stop_dict = {
                'type': pending_stop['type'],
                'location': {
                    'lat': pending_stop['location']['latitude'],
                    'lng': pending_stop['location']['longitude']
                },
                'demand': pending_stop['demand'],
                'order_id': order['id'],
                'customer': order.get('customer')
            }
            new_stops.append(stop_dict)
        
        # Insertar paradas después de la posición actual (antes del último depot)
        insertion_point = max(current_stop_idx + 1, len(route['stops']) - 1)
        for stop in reversed(new_stops):
            route['stops'].insert(insertion_point, stop)
        
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
        
        # Actualizar posición actual (ahora es la última parada agregada)
        if new_stops:
            last_new_stop = new_stops[-1]
            self.vehicle_positions[vehicle_id] = last_new_stop['location']
    
    def _calculate_reward(self, insertion_cost: float, order: Dict) -> float:
        """
        Calcula la recompensa por asignar una orden.
        """
        reward = 10.0
        
        # Penalización por costo de inserción
        insertion_penalty = min(5.0, insertion_cost / 10.0)
        reward -= insertion_penalty
        
        # Bonus por eficiencia
        order_weight = order.get('weight', 0)
        efficiency_bonus = (order_weight / 1000.0) * 2.0
        reward += efficiency_bonus
        
        # Bonus adicional por órdenes ya recogidas (priorizar entregas)
        status = str(order.get('status', 'pendiente')).lower()
        if status in ['recogido', '2']:
            reward += 3.0  # Priorizar completar entregas
        
        return reward
    
    def update_order_status(self, order_id: int, new_status: str) -> Dict:
        """
        Actualiza el estado de una orden existente.
        
        Returns:
            Dict con información de la actualización
        """
        updated = False
        
        # Buscar en órdenes pendientes
        for order in self.pending_orders:
            if order.get('id') == order_id:
                order['status'] = new_status
                updated = True
                break
        
        # Si cambió a cancelada, completo o pospuesta, remover de rutas
        if new_status in ['cancelada', '4', 'completo', '3', 'pospuesta', '5']:
            self.cancel_order(order_id)
        
        return {
            'order_id': order_id,
            'new_status': new_status,
            'updated': updated
        }
    
    def reset(self, new_routes: List[Dict] = None):
        """Reinicia el ambiente con nuevas rutas"""
        if new_routes:
            self.current_routes = new_routes
        
        self.vehicle_loads = {}
        self.vehicle_distances = {}
        self.vehicle_positions = {}
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
            vehicle_id = route['vehicle_id']
            
            # Filtrar paradas de esta orden
            route['stops'] = [s for s in original_stops if s.get('order_id') != order_id]
            
            # Recalcular distancia y posición si se removieron paradas
            if len(route['stops']) != len(original_stops):
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
                new_load = sum(
                    s.get('demand', 0) 
                    for s in route['stops'] 
                    if s['type'] == 'pickup'
                )
                self.vehicle_loads[vehicle_id] = new_load
                
                # Actualizar posición (última parada no-depot)
                last_position = self.depot.copy()
                for stop in route['stops']:
                    if stop['type'] != 'depot':
                        last_position = stop['location']
                self.vehicle_positions[vehicle_id] = last_position
    
    def remove_vehicle(self, vehicle_id: int) -> List[Dict]:
        """
        Remueve un vehículo y retorna las órdenes que deben ser reasignadas.
        
        IMPORTANTE: Solo reasigna órdenes en estado "pendiente" (1).
        Las órdenes "recogido" (2) Spring Boot las cambia a "pospuesta" (5) 
        y NO deben ser reasignadas.
        
        Returns:
            Lista de órdenes pendientes para reasignar
        """
        orders_to_reassign = []
        
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        
        if route:
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
                    
                    # Determinar estado de la orden
                    if pickup_stop and delivery_stop:
                        # Tiene ambas paradas → está pendiente
                        order['status'] = 'pendiente'
                        order['pickupAddress'] = {
                            'latitude': pickup_stop['location']['lat'],
                            'longitude': pickup_stop['location']['lng']
                        }
                        order['deliveryAddress'] = {
                            'latitude': delivery_stop['location']['lat'],
                            'longitude': delivery_stop['location']['lng']
                        }
                        # Solo agregar a reasignación si está pendiente
                        orders_to_reassign.append(order)
                        
                    elif delivery_stop and not pickup_stop:
                        # Solo tiene delivery → estaba recogido
                        # Spring Boot la cambió a "pospuesta", NO reasignar
                        order['status'] = 'pospuesta'
                        order['deliveryAddress'] = {
                            'latitude': delivery_stop['location']['lat'],
                            'longitude': delivery_stop['location']['lng']
                        }
                        # NO se agrega a orders_to_reassign
                        
                    elif pickup_stop and not delivery_stop:
                        # Solo tiene pickup (caso raro) → pendiente
                        order['status'] = 'pendiente'
                        order['pickupAddress'] = {
                            'latitude': pickup_stop['location']['lat'],
                            'longitude': pickup_stop['location']['lng']
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
            if vehicle_id in self.vehicle_positions:
                del self.vehicle_positions[vehicle_id]
        
        return orders_to_reassign