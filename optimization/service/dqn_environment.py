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
    
    IMPORTANTE: La posición actual es solo un punto de referencia (última parada completada).
    Las nuevas órdenes se insertan en las paradas FUTURAS, no en la posición actual.
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
        self.vehicle_distances = {}  # Distancia FUTURA (desde posición actual)
        self.vehicle_positions = {}  # Posición actual (última parada completada)
        self.pending_orders = []
        self.completed_orders = []
        
        self._initialize_state()
    
    def _initialize_state(self):
        """
        Inicializa el estado del ambiente.
        
        IMPORTANTE: Después de ACO, route['stops'] contiene:
        [depot_inicial, pickup_1, delivery_1, ..., depot_final]
        
        Necesitamos filtrar los depots y considerar solo las paradas reales.
        """
        for route in self.current_routes:
            vehicle_id = route['vehicle_id']
            
            # Filtrar stops para obtener solo las paradas reales (sin depots)
            real_stops = [s for s in route['stops'] if s['type'] != 'depot']
            
            # Calcular carga máxima que alcanzará el vehículo
            # Esto es importante porque la capacidad se refiere a la carga EN CUALQUIER MOMENTO
            max_load = 0
            current_load = 0
            
            for stop in real_stops:
                current_load += stop.get('demand', 0)
                max_load = max(max_load, abs(current_load))
            
            self.vehicle_loads[vehicle_id] = max_load
            
            # La posición actual depende del contexto:
            # 1. Si viene de vehicles con current_position → usar esa
            # 2. Si es primera vez (ACO inicial) → depot
            vehicle_info = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
            
            if vehicle_info and vehicle_info.get('current_position'):
                # Usar posición proporcionada
                current_position = vehicle_info['current_position']
            else:
                # Primera vez después de ACO → está en depot
                current_position = self.depot.copy()
            
            self.vehicle_positions[vehicle_id] = current_position
            
            # Calcular distancia FUTURA
            # Si está en depot, la distancia es desde depot hasta completar todas las paradas
            # Si está en otra posición, es desde ahí
            future_distance = self._calculate_future_distance(
                vehicle_id, real_stops, current_position
            )
            self.vehicle_distances[vehicle_id] = future_distance
    
    def _calculate_future_distance(self, vehicle_id: int, 
                                   real_stops: List[Dict], 
                                   current_position: Dict) -> float:
        """
        Calcula la distancia desde la posición actual hasta completar todas las paradas.
        
        Args:
            real_stops: Lista de paradas reales (sin depots)
            current_position: Posición actual del vehículo
        """
        if not real_stops:
            return 0.0
        
        total_distance = 0.0
        
        # Distancia desde posición actual hasta primera parada
        first_stop = real_stops[0]['location']
        total_distance += haversine_distance(
            current_position['lat'], current_position['lng'],
            first_stop['lat'], first_stop['lng']
        )
        
        # Distancias entre paradas consecutivas
        for i in range(len(real_stops) - 1):
            stop1 = real_stops[i]['location']
            stop2 = real_stops[i + 1]['location']
            total_distance += haversine_distance(
                stop1['lat'], stop1['lng'],
                stop2['lat'], stop2['lng']
            )
        
        # Distancia desde última parada de vuelta al depot
        last_stop = real_stops[-1]['location']
        total_distance += haversine_distance(
            last_stop['lat'], last_stop['lng'],
            self.depot['lat'], self.depot['lng']
        )
        
        return total_distance
    
    def get_pending_stops_for_order(self, order: Dict) -> List[Dict]:
        """
        Retorna las paradas pendientes según el estado de la orden.
        """
        status = str(order.get('status', 'pendiente')).lower()
        stops = []
        
        if status in ['pendiente', '1']:
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
            stops.append({
                'type': 'delivery',
                'location': order['deliveryAddress'],
                'demand': -order.get('weight', 0)
            })
        
        return stops
    
    def get_state(self) -> np.ndarray:
        """Retorna el estado actual del ambiente como vector de características."""
        state_vector = []
        
        for vehicle in self.vehicles:
            vehicle_id = vehicle['id']
            capacity = vehicle['capacity']
            max_distance = vehicle['max_distance']
            
            current_load = self.vehicle_loads.get(vehicle_id, 0)
            future_distance = self.vehicle_distances.get(vehicle_id, 0)
            current_pos = self.vehicle_positions.get(vehicle_id, self.depot)
            
            # Normalizar valores
            available_capacity = max(0, (capacity - current_load) / capacity) if capacity > 0 else 0
            available_distance = max(0, (max_distance - future_distance) / max_distance) if max_distance > 0 else 0
            
            distance_to_depot = haversine_distance(
                current_pos['lat'], current_pos['lng'],
                self.depot['lat'], self.depot['lng']
            )
            normalized_distance_to_depot = min(1.0, distance_to_depot / max_distance) if max_distance > 0 else 0
            
            state_vector.extend([
                available_capacity, 
                available_distance,
                normalized_distance_to_depot
            ])
        
        num_pending = len(self.pending_orders) / 100.0
        state_vector.append(min(1.0, num_pending))
        
        if self.pending_orders:
            order = self.pending_orders[0]
            weight_normalized = order.get('weight', 0) / 1000.0
            state_vector.append(min(1.0, weight_normalized))
            
            status = str(order.get('status', 'pendiente')).lower()
            state_vector.append(1.0 if status in ['pendiente', '1'] else 0.0)
            state_vector.append(1.0 if status in ['recogido', '2'] else 0.0)
        else:
            state_vector.extend([0.0, 0.0, 0.0])
        
        return np.array(state_vector, dtype=np.float32)
    
    def get_state_size(self) -> int:
        return len(self.vehicles) * 3 + 4
    
    def get_action_size(self) -> int:
        return len(self.vehicles) + 1
    
    def step(self, action: int, order: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """Ejecuta una acción en el ambiente."""
        reward = 0
        info = {'assigned': False, 'feasible': False, 'vehicle_id': None}
        
        pending_stops = self.get_pending_stops_for_order(order)
        
        if not pending_stops:
            reward = -0.1
            info['assigned'] = False
            info['reason'] = 'order_not_actionable'
        elif action == len(self.vehicles):
            self.pending_orders.append(order)
            reward = -1.0
            info['assigned'] = False
            info['reason'] = 'no_assignment_chosen'
        else:
            vehicle = self.vehicles[action]
            vehicle_id = vehicle['id']
            
            feasible, insertion_cost, best_position = self._check_feasibility(
                vehicle_id, order, pending_stops
            )
            
            if feasible:
                self._assign_order_to_vehicle(
                    vehicle_id, order, pending_stops, best_position
                )
                
                reward = self._calculate_reward(insertion_cost, order)
                info['assigned'] = True
                info['feasible'] = True
                info['vehicle_id'] = vehicle_id
                info['insertion_cost'] = insertion_cost
                info['insertion_position'] = best_position
                self.completed_orders.append(order)
            else:
                self.pending_orders.append(order)
                reward = -5.0
                info['assigned'] = False
                info['feasible'] = False
                info['insertion_cost'] = insertion_cost
                info['reason'] = 'infeasible_insertion'
        
        next_state = self.get_state()
        done = len(self.pending_orders) == 0
        
        return next_state, reward, done, info
    
    def _check_feasibility(self, vehicle_id: int, order: Dict, 
                          pending_stops: List[Dict]) -> Tuple[bool, float, int]:
        """
        Verifica factibilidad considerando que la posición actual NO es una parada.
        
        Returns:
            (feasible, insertion_cost, best_insertion_position)
        """
        vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
        if not vehicle:
            return False, float('inf'), -1
        
        # Verificar capacidad
        order_weight = sum(abs(stop['demand']) for stop in pending_stops if stop['type'] == 'pickup')
        current_load = self.vehicle_loads.get(vehicle_id, 0)
        
        if current_load + order_weight > vehicle['capacity']:
            return False, float('inf'), -1
        
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if not route:
            return False, float('inf'), -1
        
        current_position = self.vehicle_positions.get(vehicle_id, self.depot)
        
        # Filtrar depots de las paradas
        real_stops = [s for s in route['stops'] if s['type'] != 'depot']
        
        # Evaluar todas las posiciones posibles de inserción
        best_cost = float('inf')
        best_position = -1
        
        for insert_idx in range(len(real_stops) + 1):
            cost = self._calculate_insertion_cost(
                real_stops, insert_idx, pending_stops, current_position
            )
            
            if cost < best_cost:
                best_cost = cost
                best_position = insert_idx
        
        # Verificar restricción de distancia
        current_future_distance = self.vehicle_distances.get(vehicle_id, 0)
        new_future_distance = current_future_distance + best_cost
        
        if new_future_distance > vehicle['max_distance']:
            return False, best_cost, best_position
        
        return True, best_cost, best_position
    
    def _calculate_insertion_cost(self, real_stops: List[Dict], 
                                  insertion_idx: int,
                                  pending_stops: List[Dict],
                                  current_position: Dict) -> float:
        """
        Calcula el costo de insertar paradas en una posición específica.
        
        Args:
            real_stops: Lista de paradas reales (sin depots)
            insertion_idx: Índice donde insertar (0 = al inicio, len = al final)
            pending_stops: Paradas a insertar
            current_position: Posición actual del vehículo
        """
        
        # Determinar puntos antes y después de la inserción
        if insertion_idx == 0:
            # Insertar al inicio: desde posición actual
            prev_location = current_position
            next_location = real_stops[0]['location'] if real_stops else self.depot
        elif insertion_idx == len(real_stops):
            # Insertar al final: después de última parada, antes de depot
            prev_location = real_stops[-1]['location'] if real_stops else current_position
            next_location = self.depot
        else:
            # Insertar en medio
            prev_location = real_stops[insertion_idx - 1]['location']
            next_location = real_stops[insertion_idx]['location']
        
        # Costo original (de prev a next directamente)
        original_cost = haversine_distance(
            prev_location['lat'], prev_location['lng'],
            next_location['lat'], next_location['lng']
        )
        
        # Calcular nuevo costo con las paradas insertadas
        new_cost = 0
        current_loc = prev_location
        
        for pending_stop in pending_stops:
            stop_location = pending_stop['location']
            stop_lat = stop_location.get('latitude', stop_location.get('lat'))
            stop_lng = stop_location.get('longitude', stop_location.get('lng'))
            
            new_cost += haversine_distance(
                current_loc['lat'], current_loc['lng'],
                stop_lat, stop_lng
            )
            
            current_loc = {'lat': stop_lat, 'lng': stop_lng}
        
        # Distancia desde última parada insertada hasta next
        new_cost += haversine_distance(
            current_loc['lat'], current_loc['lng'],
            next_location['lat'], next_location['lng']
        )
        
        # El incremento de distancia
        return new_cost - original_cost
    
    def _assign_order_to_vehicle(self, vehicle_id: int, order: Dict, 
                                 pending_stops: List[Dict], insertion_idx: int):
        """
        Asigna una orden insertando las paradas en la mejor posición.
        
        IMPORTANTE: Reconstruye route['stops'] con depots al inicio y final.
        """
        # Actualizar carga máxima
        # Necesitamos recalcular porque la nueva orden puede cambiar la carga máxima
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        if not route:
            return
        
        # Filtrar stops actuales (sin depots)
        real_stops = [s for s in route['stops'] if s['type'] != 'depot']
        
        # Crear paradas completas para la nueva orden
        new_stops = []
        for pending_stop in pending_stops:
            stop_location = pending_stop['location']
            stop_lat = stop_location.get('latitude', stop_location.get('lat'))
            stop_lng = stop_location.get('longitude', stop_location.get('lng'))
            
            stop_dict = {
                'type': pending_stop['type'],
                'location': {
                    'lat': stop_lat,
                    'lng': stop_lng
                },
                'demand': pending_stop['demand'],
                'order_id': order['id'],
                'customer': order.get('customer')
            }
            new_stops.append(stop_dict)
        
        # Insertar en la posición calculada
        for idx, stop in enumerate(new_stops):
            real_stops.insert(insertion_idx + idx, stop)
        
        # Recalcular carga máxima
        max_load = 0
        current_load = 0
        for stop in real_stops:
            current_load += stop.get('demand', 0)
            max_load = max(max_load, abs(current_load))
        
        self.vehicle_loads[vehicle_id] = max_load
        
        # Recalcular distancia futura
        current_position = self.vehicle_positions.get(vehicle_id, self.depot)
        new_future_distance = self._calculate_future_distance(
            vehicle_id, real_stops, current_position
        )
        
        # Reconstruir route['stops'] con depots
        route['stops'] = [
            {'type': 'depot', 'location': self.depot}
        ] + real_stops + [
            {'type': 'depot', 'location': self.depot}
        ]
        
        route['total_distance'] = new_future_distance
        self.vehicle_distances[vehicle_id] = new_future_distance
    
    def _calculate_reward(self, insertion_cost: float, order: Dict) -> float:
        """Calcula la recompensa por asignar una orden."""
        reward = 10.0
        
        # Penalización por costo de inserción
        insertion_penalty = min(5.0, insertion_cost / 10.0)
        reward -= insertion_penalty
        
        # Bonus por eficiencia
        order_weight = order.get('weight', 0)
        efficiency_bonus = (order_weight / 1000.0) * 2.0
        reward += efficiency_bonus
        
        # Bonus por órdenes ya recogidas
        status = str(order.get('status', 'pendiente')).lower()
        if status in ['recogido', '2']:
            reward += 3.0
        
        return reward
    
    def update_order_status(self, order_id: int, new_status: str) -> Dict:
        """Actualiza el estado de una orden existente."""
        updated = False
        
        for order in self.pending_orders:
            if order.get('id') == order_id:
                order['status'] = new_status
                updated = True
                break
        
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
        self.pending_orders = [o for o in self.pending_orders if o.get('id') != order_id]
        
        for route in self.current_routes:
            vehicle_id = route['vehicle_id']
            
            # Filtrar stops sin depots
            real_stops = [s for s in route['stops'] if s['type'] != 'depot']
            original_count = len(real_stops)
            
            # Remover paradas de esta orden
            real_stops = [s for s in real_stops if s.get('order_id') != order_id]
            
            if len(real_stops) != original_count:
                # Hubo cambios, recalcular todo
                current_position = self.vehicle_positions.get(vehicle_id, self.depot)
                
                # Recalcular carga máxima
                max_load = 0
                current_load = 0
                for stop in real_stops:
                    current_load += stop.get('demand', 0)
                    max_load = max(max_load, abs(current_load))
                
                self.vehicle_loads[vehicle_id] = max_load
                
                # Recalcular distancia futura
                new_future_distance = self._calculate_future_distance(
                    vehicle_id, real_stops, current_position
                )
                
                # Reconstruir route['stops'] con depots
                route['stops'] = [
                    {'type': 'depot', 'location': self.depot}
                ] + real_stops + [
                    {'type': 'depot', 'location': self.depot}
                ]
                
                route['total_distance'] = new_future_distance
                self.vehicle_distances[vehicle_id] = new_future_distance
    
    def remove_vehicle(self, vehicle_id: int) -> List[Dict]:
        """
        Remueve un vehículo y retorna las órdenes pendientes para reasignar.
        Solo reasigna órdenes en estado "pendiente".
        """
        orders_to_reassign = []
        
        route = next((r for r in self.current_routes if r['vehicle_id'] == vehicle_id), None)
        
        if route:
            # Filtrar stops sin depots
            real_stops = [s for s in route['stops'] if s['type'] != 'depot']
            
            order_ids_seen = set()
            for stop in real_stops:
                if stop.get('order_id') and stop['order_id'] not in order_ids_seen:
                    order_ids_seen.add(stop['order_id'])
                    
                    order = {
                        'id': stop['order_id'],
                        'weight': abs(stop.get('demand', 0)),
                        'customer': stop.get('customer'),
                    }
                    
                    pickup_stop = next((s for s in real_stops 
                                       if s.get('order_id') == stop['order_id'] 
                                       and s['type'] == 'pickup'), None)
                    delivery_stop = next((s for s in real_stops 
                                         if s.get('order_id') == stop['order_id'] 
                                         and s['type'] == 'delivery'), None)
                    
                    if pickup_stop and delivery_stop:
                        order['status'] = 'pendiente'
                        order['pickupAddress'] = {
                            'latitude': pickup_stop['location']['lat'],
                            'longitude': pickup_stop['location']['lng']
                        }
                        order['deliveryAddress'] = {
                            'latitude': delivery_stop['location']['lat'],
                            'longitude': delivery_stop['location']['lng']
                        }
                        orders_to_reassign.append(order)
                        
                    elif delivery_stop and not pickup_stop:
                        order['status'] = 'pospuesta'
                        order['deliveryAddress'] = {
                            'latitude': delivery_stop['location']['lat'],
                            'longitude': delivery_stop['location']['lng']
                        }
                        
                    elif pickup_stop and not delivery_stop:
                        order['status'] = 'pendiente'
                        order['pickupAddress'] = {
                            'latitude': pickup_stop['location']['lat'],
                            'longitude': pickup_stop['location']['lng']
                        }
                        orders_to_reassign.append(order)
            
            self.current_routes = [r for r in self.current_routes if r['vehicle_id'] != vehicle_id]
            self.vehicles = [v for v in self.vehicles if v['id'] != vehicle_id]
            
            if vehicle_id in self.vehicle_loads:
                del self.vehicle_loads[vehicle_id]
            if vehicle_id in self.vehicle_distances:
                del self.vehicle_distances[vehicle_id]
            if vehicle_id in self.vehicle_positions:
                del self.vehicle_positions[vehicle_id]
        
        return orders_to_reassign