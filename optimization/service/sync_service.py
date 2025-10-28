"""
Servicio para sincronizar datos desde Spring Boot y detectar cambios.
Archivo: optimization/service/sync_service.py
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
                'status': snapshot.status
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
        
        # Detectar nuevas órdenes
        new_order_ids = curr_order_ids - prev_order_ids
        changes['new_orders'] = [
            o for o in self.current_orders 
            if o['id'] in new_order_ids and o.get('status') == '1'
        ]
        
        # Detectar órdenes canceladas
        cancelled_order_ids = prev_order_ids - curr_order_ids
        changes['cancelled_orders'] = list(cancelled_order_ids)
        
        # Detectar cambios de estado en órdenes
        for curr_order in self.current_orders:
            prev_order = next(
                (o for o in self.previous_orders if o['id'] == curr_order['id']), 
                None
            )
            if prev_order and prev_order.get('status') != curr_order.get('status'):
                changes['status_changed_orders'].append({
                    'order_id': curr_order['id'],
                    'old_status': prev_order.get('status'),
                    'new_status': curr_order.get('status')
                })
        
        # Detectar vehículos removidos (status cambió de 1 a 0 o fue eliminado)
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
    
    def save_snapshot(self, execution: OptimizationExecution, 
                     orders: List[Dict], vehicles: List[Dict]):
        """
        Guarda un snapshot del estado actual en SQLite.
        """
        # Guardar órdenes
        for order in orders:
            OrderSnapshot.objects.create(
                execution=execution,
                order_id=order['id'],
                weight=order.get('weight', 0),
                status=order.get('status', '0'),
                pickup_lat=order['pickupAddress']['latitude'],
                pickup_lng=order['pickupAddress']['longitude'],
                delivery_lat=order['deliveryAddress']['latitude'],
                delivery_lng=order['deliveryAddress']['longitude'],
                customer_data=order.get('customer')
            )
        
        # Guardar vehículos
        for vehicle in vehicles:
            VehicleSnapshot.objects.create(
                execution=execution,
                vehicle_id=vehicle['id'],
                capacity=vehicle.get('capacity', 1000),
                max_distance=vehicle.get('max_distance', 100),
                status=vehicle.get('status', 1)
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
                        # Esta distancia debería calcularse, pero por simplicidad usamos None
                        distance_to_next = None
                    
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
            # Primera ejecución, todas las órdenes son nuevas
            changes = {
                'new_orders': [o for o in current_orders if o.get('status') == '1'],
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
            'active_orders_count': len([o for o in current_orders if o.get('status') == '1']),
            'active_vehicles_count': len([v for v in current_vehicles if v.get('status') == 1]),
            'changes': changes
        }
    
    def get_active_orders(self) -> List[Dict]:
        """Retorna solo las órdenes activas (status = '1')"""
        return [o for o in self.current_orders if o.get('status') == '1']
    
    def get_active_vehicles(self) -> List[Dict]:
        """Retorna solo los vehículos activos (status = 1)"""
        return [v for v in self.current_vehicles if v.get('status') == 1]