"""
Servicio para sincronizar datos desde Spring Boot y detectar cambios.
Maneja estados de órdenes y posiciones actuales de vehículos.
"""

from typing import Dict, List, Tuple, Optional
from optimization.models import (
    OptimizationExecution, 
    OrderSnapshot, 
    VehicleSnapshot,
    RouteAssignment,
    DQNState
)
from optimization.service.api_service import obtener_ordenes, obtener_vehiculos
from django.utils import timezone


class SyncService:
    """
    Servicio para sincronizar y detectar cambios en órdenes y vehículos.
    """
    
    def __init__(self):
        self.last_execution = None
        self.current_orders = []
        self.current_vehicles = []
        self.previous_orders = []
        self.previous_vehicles = []
    
    def get_last_execution(self) -> Optional[OptimizationExecution]:
        """Obtiene la última ejecución completada"""
        return OptimizationExecution.objects.filter(
            status='completed'
        ).order_by('-executed_at').first()
    
    def fetch_current_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Obtiene los datos actuales desde Spring Boot.
        
        Returns:
            Tuple (orders, vehicles)
        """
        try:
            orders = obtener_ordenes()
            vehicles = obtener_vehiculos()
            
            self.current_orders = orders
            self.current_vehicles = vehicles
            
            return orders, vehicles
        
        except Exception as e:
            raise Exception(f"Error al obtener datos desde Spring Boot: {str(e)}")
    
    def load_previous_data(self, execution: OptimizationExecution) -> Tuple[List[Dict], List[Dict]]:
        """
        Carga los datos de la última ejecución desde SQLite.
        
        Returns:
            Tuple (orders, vehicles)
        """
        # Cargar órdenes
        order_snapshots = OrderSnapshot.objects.filter(execution=execution)
        orders = []
        
        for snapshot in order_snapshots:
            order = {
                'id': snapshot.order_id,
                'weight': snapshot.weight,
                'status': snapshot.status,
                'pickupAddress': {
                    'latitude': snapshot.pickup_lat,
                    'longitude': snapshot.pickup_lng
                },
                'deliveryAddress': {
                    'latitude': snapshot.delivery_lat,
                    'longitude': snapshot.delivery_lng
                },
                'customer': snapshot.customer_data
            }
            orders.append(order)
        
        # Cargar vehículos
        vehicle_snapshots = VehicleSnapshot.objects.filter(execution=execution)
        vehicles = []
        
        for snapshot in vehicle_snapshots:
            vehicle = {
                'id': snapshot.vehicle_id,
                'capacity': snapshot.capacity,
                'max_distance': snapshot.max_distance,
                'status': snapshot.status,
                'current_position': {
                    'lat': snapshot.current_lat,
                    'lng': snapshot.current_lng
                } if snapshot.current_lat and snapshot.current_lng else None
            }
            vehicles.append(vehicle)
        
        self.previous_orders = orders
        self.previous_vehicles = vehicles
        
        return orders, vehicles
    
    def detect_changes(self) -> Dict:
        """
        Detecta cambios entre la última ejecución y el estado actual.
        
        Returns:
            Dict con información de cambios:
            {
                'new_orders': [...],
                'cancelled_orders': [...],
                'removed_vehicles': [...],
                'status_changed_orders': [...],
                'has_changes': bool
            }
        """
        changes = {
            'new_orders': [],
            'cancelled_orders': [],
            'removed_vehicles': [],
            'status_changed_orders': [],
            'has_changes': False
        }
        
        # Crear conjuntos para comparación rápida
        prev_order_ids = {o['id'] for o in self.previous_orders}
        curr_order_ids = {o['id'] for o in self.current_orders}
        
        prev_vehicle_ids = {v['id'] for v in self.previous_vehicles if v.get('status') == 1}
        curr_vehicle_ids = {v['id'] for v in self.current_vehicles if v.get('status') == 1}
        
        # Detectar nuevas órdenes (solo las activas: pendiente o recogido)
        new_order_ids = curr_order_ids - prev_order_ids
        changes['new_orders'] = [
            o for o in self.current_orders 
            if o['id'] in new_order_ids and self._is_order_active(o.get('status'))
        ]
        
        # Detectar órdenes canceladas o completadas
        for curr_order in self.current_orders:
            prev_order = next(
                (o for o in self.previous_orders if o['id'] == curr_order['id']), 
                None
            )
            
            # Si la orden cambió a cancelada o completada
            if prev_order:
                prev_status = self._normalize_status(prev_order.get('status'))
                curr_status = self._normalize_status(curr_order.get('status'))
                
                if prev_status != curr_status:
                    changes['status_changed_orders'].append({
                        'order_id': curr_order['id'],
                        'old_status': prev_status,
                        'new_status': curr_status
                    })
                    
                    # Si cambió a cancelada, agregar a lista de canceladas
                    if curr_status == 'cancelada':
                        changes['cancelled_orders'].append(curr_order['id'])
        
        # Detectar órdenes que ya no existen en el sistema
        deleted_order_ids = prev_order_ids - curr_order_ids
        changes['cancelled_orders'].extend(list(deleted_order_ids))
        
        # Detectar vehículos removidos
        removed_vehicle_ids = prev_vehicle_ids - curr_vehicle_ids
        changes['removed_vehicles'] = list(removed_vehicle_ids)
        
        # Determinar si hay cambios
        changes['has_changes'] = (
            len(changes['new_orders']) > 0 or
            len(changes['cancelled_orders']) > 0 or
            len(changes['removed_vehicles']) > 0 or
            len(changes['status_changed_orders']) > 0
        )
        
        return changes
    
    def _normalize_status(self, status) -> str:
        """
        Normaliza el estado de la orden a un formato estándar.
        
        Mapeo desde Spring Boot:
        - '1' o 'pendiente' -> 'pendiente'
        - '2' o 'recogido' -> 'recogido'
        - '3' o 'completo' -> 'completo'
        - '4' o 'cancelada' -> 'cancelada'
        - '5' o 'pospuesta' -> 'pospuesta'
        """
        status_str = str(status).lower().strip()
        
        if status_str in ['1', 'pendiente']:
            return 'pendiente'
        elif status_str in ['2', 'recogido']:
            return 'recogido'
        elif status_str in ['3', 'completo', 'completada']:
            return 'completo'
        elif status_str in ['4', 'cancelada']:
            return 'cancelada'
        elif status_str in ['5', 'pospuesta']:
            return 'pospuesta'
        else:
            return 'pendiente'  # Por defecto
    
    def _is_order_active(self, status) -> bool:
        """
        Determina si una orden está activa (pendiente o recogido).
        
        Returns:
            True si la orden está pendiente o recogida
            False si está completa, cancelada o pospuesta
        """
        normalized = self._normalize_status(status)
        return normalized in ['pendiente', 'recogido']
    
    def save_snapshot(self, execution: OptimizationExecution, 
                     orders: List[Dict], vehicles: List[Dict]):
        """
        Guarda un snapshot del estado actual en SQLite.
        Incluye posiciones actuales de vehículos.
        """
        # Guardar órdenes con su estado normalizado
        for order in orders:
            normalized_status = self._normalize_status(order.get('status', 'pendiente'))
            
            OrderSnapshot.objects.create(
                execution=execution,
                order_id=order['id'],
                weight=order.get('weight', 0),
                status=normalized_status,
                pickup_lat=order['pickupAddress']['latitude'],
                pickup_lng=order['pickupAddress']['longitude'],
                delivery_lat=order['deliveryAddress']['latitude'],
                delivery_lng=order['deliveryAddress']['longitude'],
                customer_data=order.get('customer')
            )
        
        # Guardar vehículos con posición actual
        for vehicle in vehicles:
            current_pos = vehicle.get('current_position')
            
            VehicleSnapshot.objects.create(
                execution=execution,
                vehicle_id=vehicle['id'],
                capacity=vehicle.get('capacity', 1000),
                max_distance=vehicle.get('max_distance', 100),
                status=vehicle.get('status', 1),
                current_lat=current_pos['lat'] if current_pos else None,
                current_lng=current_pos['lng'] if current_pos else None
            )
    
    def save_route_assignments(self, execution: OptimizationExecution, routes: List[Dict]):
        """
        Guarda las asignaciones de rutas en SQLite.
        """
        for route in routes:
            vehicle_id = route['vehicle_id']
            cumulative_distance = 0
            cumulative_load = 0
            
            for idx, stop in enumerate(route['stops']):
                if stop['type'] in ['pickup', 'delivery']:
                    order_id = stop.get('order_id')
                    demand = stop.get('demand', 0)
                    
                    if stop['type'] == 'pickup':
                        cumulative_load += demand
                    
                    # Calcular distancia al siguiente punto
                    distance_to_next = None
                    if idx < len(route['stops']) - 1:
                        stop1 = stop['location']
                        stop2 = route['stops'][idx + 1]['location']
                        
                        from optimization.service.dqn_environment import haversine_distance
                        distance_to_next = haversine_distance(
                            stop1['lat'], stop1['lng'],
                            stop2['lat'], stop2['lng']
                        )
                        cumulative_distance += distance_to_next
                    
                    RouteAssignment.objects.create(
                        execution=execution,
                        order_id=order_id,
                        vehicle_id=vehicle_id,
                        stop_sequence=idx,
                        stop_type=stop['type'],
                        distance_to_next=distance_to_next,
                        cumulative_distance=cumulative_distance,
                        cumulative_load=cumulative_load
                    )
    
    def sync_and_detect(self) -> Dict:
        """
        Sincroniza datos y detecta cambios.
        
        Returns:
            Dict con información completa de sincronización y cambios
        """
        # Obtener última ejecución
        last_execution = self.get_last_execution()
        
        # Obtener datos actuales desde Spring Boot
        current_orders, current_vehicles = self.fetch_current_data()
        
        # Si hay una ejecución previa, cargar datos y detectar cambios
        if last_execution:
            previous_orders, previous_vehicles = self.load_previous_data(last_execution)
            changes = self.detect_changes()
        else:
            # Primera ejecución, todas las órdenes activas son nuevas
            changes = {
                'new_orders': [o for o in current_orders if self._is_order_active(o.get('status'))],
                'cancelled_orders': [],
                'removed_vehicles': [],
                'status_changed_orders': [],
                'has_changes': True
            }
        
        return {
            'last_execution_id': last_execution.id if last_execution else None,
            'last_execution_date': last_execution.executed_at if last_execution else None,
            'current_orders_count': len(current_orders),
            'current_vehicles_count': len(current_vehicles),
            'active_orders_count': len([o for o in current_orders if self._is_order_active(o.get('status'))]),
            'active_vehicles_count': len([v for v in current_vehicles if v.get('status') == 1]),
            'changes': changes
        }
    
    def get_active_orders(self) -> List[Dict]:
        """Retorna solo las órdenes activas (pendiente o recogido)"""
        return [o for o in self.current_orders if self._is_order_active(o.get('status'))]
    
    def get_active_vehicles(self) -> List[Dict]:
        """Retorna solo los vehículos activos (status = 1)"""
        return [v for v in self.current_vehicles if v.get('status') == 1]
    
    def extract_vehicle_positions_from_routes(self, routes: List[Dict], depot: Dict) -> Dict:
        """
        Extrae las posiciones actuales de los vehículos desde las rutas.
        La posición actual es la última parada no-depot de cada ruta.
        
        Returns:
            Dict {vehicle_id: {'lat': ..., 'lng': ...}}
        """
        vehicle_positions = {}
        
        for route in routes:
            vehicle_id = route['vehicle_id']
            last_position = depot.copy()
            
            # Encontrar última parada no-depot
            for stop in route['stops']:
                if stop['type'] != 'depot':
                    last_position = stop['location']
            
            vehicle_positions[vehicle_id] = last_position
        
        return vehicle_positions