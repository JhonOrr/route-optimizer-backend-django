import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from .dqn_environment import VRPEnvironment
from .dqn_agent import DQNAgent


class RouteManager:
    """
    Gestor central para reasignación dinámica de rutas usando DQN.
    Maneja estados de órdenes y posiciones actuales de vehículos.
    
    Coordina:
    - Adición de nuevas órdenes
    - Cancelación de órdenes
    - Cambios de estado de órdenes
    - Remoción de vehículos
    - Entrenamiento del agente DQN
    """
    
    def __init__(self, depot: Dict, vehicles: List[Dict], 
                 initial_routes: List[Dict],
                 model_path: str = 'models/dqn_vrp_model.pth'):
        """
        Args:
            depot: Coordenadas del depósito
            vehicles: Lista de vehículos disponibles
            initial_routes: Rutas iniciales del algoritmo ACO
            model_path: Ruta para guardar/cargar el modelo DQN
        """
        self.depot = depot
        self.vehicles = vehicles
        self.current_routes = initial_routes
        self.model_path = model_path
        
        # Inicializar ambiente
        self.env = VRPEnvironment(depot, vehicles, initial_routes)
        
        # Inicializar agente DQN
        state_size = self.env.get_state_size()
        action_size = self.env.get_action_size()
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=32,
            target_update_freq=10
        )
        
        # Cargar modelo si existe
        if os.path.exists(model_path):
            try:
                self.agent.load_model(model_path)
                print(f"Modelo DQN cargado desde: {model_path}")
            except Exception as e:
                print(f"No se pudo cargar el modelo: {e}. Iniciando con modelo nuevo.")
        
        # Estadísticas
        self.operations_log = []
    
    def add_order(self, order: Dict, training: bool = True) -> Dict:
        """
        Añade una nueva orden al sistema y la asigna usando DQN.
        Considera el estado de la orden para determinar las paradas pendientes.
        
        Args:
            order: Nueva orden a asignar
            training: Si es True, entrena el agente durante la asignación
            
        Returns:
            Dict con resultado de la operación
        """
        # Verificar si la orden es accionable
        status = self._normalize_status(order.get('status', 'pendiente'))
        
        if status in ['cancelada', 'completo', 'pospuesta']:
            return {
                'type': 'add_order',
                'order_id': order.get('id'),
                'assigned': False,
                'vehicle_id': None,
                'reward': 0,
                'loss': None,
                'epsilon': self.agent.epsilon,
                'reason': f'order_status_{status}'
            }
        
        # Agregar orden al ambiente
        self.env.add_new_order(order)
        
        # Obtener estado actual
        state = self.env.get_state()
        
        # Seleccionar acción (asignar a vehículo o dejar pendiente)
        action = self.agent.act(state, training=training)
        
        # Ejecutar acción
        next_state, reward, done, info = self.env.step(action, order)
        
        # Si está en modo entrenamiento, almacenar experiencia y entrenar
        if training:
            self.agent.remember(state, action, reward, next_state, done)
            loss = self.agent.replay()
            self.agent.decay_epsilon()
        else:
            loss = None
        
        # Actualizar rutas actuales
        self.current_routes = self.env.current_routes
        
        # Registrar operación
        operation_result = {
            'type': 'add_order',
            'order_id': order.get('id'),
            'order_status': status,
            'assigned': info['assigned'],
            'vehicle_id': info.get('vehicle_id'),
            'reward': reward,
            'loss': loss,
            'epsilon': self.agent.epsilon,
            'pending_stops': len(self.env.get_pending_stops_for_order(order))
        }
        
        self.operations_log.append(operation_result)
        
        return operation_result
    
    def update_order_status(self, order_id: int, new_status: str, training: bool = True) -> Dict:
        """
        Actualiza el estado de una orden existente.
        
        Estados:
        - 'pendiente' (1): Requiere pickup y delivery
        - 'recogido' (2): Solo requiere delivery
        - 'completo' (3): Ya fue entregada, se remueve de rutas
        - 'cancelada' (4): Se remueve de rutas
        - 'pospuesta' (5): Se remueve de rutas (no se reasigna)
        
        Args:
            order_id: ID de la orden
            new_status: Nuevo estado
            training: Si es True, entrena durante la operación
            
        Returns:
            Dict con resultado de la operación
        """
        normalized_status = self._normalize_status(new_status)
        
        # Actualizar estado en el ambiente
        result = self.env.update_order_status(order_id, normalized_status)
        
        # Si cambió a cancelada, completo o pospuesta, cancelar
        if normalized_status in ['cancelada', 'completo', 'pospuesta']:
            cancel_result = self.cancel_order(order_id)
            result.update(cancel_result)
        
        # Actualizar rutas actuales
        self.current_routes = self.env.current_routes
        
        operation_result = {
            'type': 'status_change',
            'order_id': order_id,
            'new_status': normalized_status,
            'success': result.get('updated', False),
            'removed_from_routes': normalized_status in ['cancelada', 'completo', 'pospuesta']
        }
        
        self.operations_log.append(operation_result)
        
        return operation_result
    
    def cancel_order(self, order_id: int) -> Dict:
        """
        Cancela una orden existente.
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            Dict con resultado de la operación
        """
        # Cancelar orden en el ambiente
        self.env.cancel_order(order_id)
        
        # Actualizar rutas actuales
        self.current_routes = self.env.current_routes
        
        operation_result = {
            'type': 'cancel_order',
            'order_id': order_id,
            'success': True
        }
        
        self.operations_log.append(operation_result)
        
        return operation_result
    
    def remove_vehicle(self, vehicle_id: int, training: bool = True) -> Dict:
        """
        Remueve un vehículo y reasigna sus órdenes a otros vehículos.
        
        IMPORTANTE: Solo reasigna órdenes en estado "pendiente" (1).
        Las órdenes en estado "recogido" (2) Spring Boot las cambia a "pospuesta" (5)
        y NO deben ser reasignadas por DQN.
        
        Args:
            vehicle_id: ID del vehículo a remover
            training: Si es True, entrena durante la reasignación
            
        Returns:
            Dict con resultado de la operación
        """
        # Obtener órdenes que deben ser reasignadas (solo pendientes)
        orders_to_reassign = self.env.remove_vehicle(vehicle_id)
        
        reassignment_results = []
        successful_reassignments = 0
        pending_orders = []
        postponed_count = 0
        
        # Intentar reasignar cada orden pendiente
        for order in orders_to_reassign:
            result = self.add_order(order, training=training)
            reassignment_results.append(result)
            
            if result['assigned']:
                successful_reassignments += 1
            else:
                pending_orders.append(order)
        
        # Actualizar rutas actuales
        self.current_routes = self.env.current_routes
        
        operation_result = {
            'type': 'remove_vehicle',
            'vehicle_id': vehicle_id,
            'orders_to_reassign': len(orders_to_reassign),
            'successful_reassignments': successful_reassignments,
            'pending_orders': len(pending_orders),
            'pending_order_ids': [o.get('id') for o in pending_orders],
            'note': 'Orders with status "recogido" were changed to "pospuesta" by Spring Boot and are NOT reassigned',
            'reassignment_details': reassignment_results
        }
        
        self.operations_log.append(operation_result)
        
        return operation_result
    
    def batch_add_orders(self, orders: List[Dict], training: bool = True) -> Dict:
        """
        Añade múltiples órdenes en batch.
        
        Args:
            orders: Lista de órdenes a añadir
            training: Si es True, entrena durante la asignación
            
        Returns:
            Dict con resumen de resultados
        """
        results = []
        successful_assignments = 0
        pending_orders = []
        skipped_orders = []
        
        for order in orders:
            status = self._normalize_status(order.get('status', 'pendiente'))
            
            # Solo procesar órdenes activas (pendiente o recogido)
            if status in ['cancelada', 'completo', 'pospuesta']:
                skipped_orders.append({
                    'order_id': order.get('id'),
                    'reason': f'status_{status}'
                })
                continue
            
            result = self.add_order(order, training=training)
            results.append(result)
            
            if result['assigned']:
                successful_assignments += 1
            else:
                pending_orders.append(order.get('id'))
        
        summary = {
            'type': 'batch_add_orders',
            'total_orders': len(orders),
            'processed_orders': len(results),
            'skipped_orders': len(skipped_orders),
            'successful_assignments': successful_assignments,
            'pending_orders': len(pending_orders),
            'pending_order_ids': pending_orders,
            'skipped_order_details': skipped_orders,
            'assignment_rate': successful_assignments / len(results) if results else 0,
            'details': results
        }
        
        self.operations_log.append(summary)
        
        return summary
    
    def get_current_routes(self) -> List[Dict]:
        """Retorna las rutas actuales"""
        return self.current_routes
    
    def get_pending_orders(self) -> List[Dict]:
        """Retorna las órdenes pendientes de asignación"""
        return self.env.pending_orders
    
    def get_vehicle_positions(self) -> Dict:
        """Retorna las posiciones actuales de todos los vehículos"""
        return self.env.vehicle_positions.copy()
    
    def save_model(self):
        """Guarda el modelo DQN entrenado"""
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Guardar modelo
        self.agent.save_model(self.model_path)
        
        # Guardar log de operaciones
        log_path = self.model_path.replace('.pth', '_operations.json')
        with open(log_path, 'w') as f:
            json.dump(self.operations_log, f, indent=2)
        
        return {
            'model_path': self.model_path,
            'operations_logged': len(self.operations_log)
        }
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del sistema"""
        total_distance = sum(route['total_distance'] for route in self.current_routes)
        
        # Utilización de vehículos
        vehicle_utilization = []
        for route in self.current_routes:
            vehicle = next((v for v in self.vehicles if v['id'] == route['vehicle_id']), None)
            if vehicle:
                current_load = sum(
                    stop.get('demand', 0) 
                    for stop in route['stops'] 
                    if stop['type'] == 'pickup'
                )
                utilization = (current_load / vehicle['capacity']) * 100
                
                # Posición actual del vehículo
                position = self.env.vehicle_positions.get(vehicle['id'], self.depot)
                
                vehicle_utilization.append({
                    'vehicle_id': vehicle['id'],
                    'utilization': utilization,
                    'distance_used': route['total_distance'],
                    'distance_capacity': vehicle['max_distance'],
                    'current_position': position
                })
        
        # Estadísticas del agente
        agent_stats = self.agent.get_stats()
        
        # Contar órdenes por estado en rutas actuales
        order_states = {'pendiente': 0, 'recogido': 0, 'completada': 0}
        processed_orders = set()
        
        for route in self.current_routes:
            for stop in route['stops']:
                order_id = stop.get('order_id')
                if order_id and order_id not in processed_orders:
                    processed_orders.add(order_id)
                    # Determinar estado basado en paradas presentes
                    has_pickup = any(
                        s.get('order_id') == order_id and s['type'] == 'pickup' 
                        for s in route['stops']
                    )
                    has_delivery = any(
                        s.get('order_id') == order_id and s['type'] == 'delivery' 
                        for s in route['stops']
                    )
                    
                    if has_pickup and has_delivery:
                        order_states['pendiente'] += 1
                    elif has_delivery and not has_pickup:
                        order_states['recogido'] += 1
        
        return {
            'total_distance': total_distance,
            'num_vehicles_used': len(self.current_routes),
            'num_pending_orders': len(self.env.pending_orders),
            'vehicle_utilization': vehicle_utilization,
            'avg_vehicle_utilization': np.mean([v['utilization'] for v in vehicle_utilization]) if vehicle_utilization else 0,
            'agent_epsilon': self.agent.epsilon,
            'total_operations': len(self.operations_log),
            'order_states_in_routes': order_states,
            'agent_training_stats': agent_stats
        }
    
    def train_batch(self, num_episodes: int = 100) -> Dict:
        """
        Entrena el agente con las experiencias almacenadas.
        
        Args:
            num_episodes: Número de episodios de entrenamiento
            
        Returns:
            Dict con resultados del entrenamiento
        """
        training_results = {
            'episodes': num_episodes,
            'avg_loss': [],
            'final_epsilon': None
        }
        
        for episode in range(num_episodes):
            # Entrenar con batch del replay buffer
            if len(self.agent.memory) >= self.agent.batch_size:
                loss = self.agent.replay()
                if loss is not None:
                    training_results['avg_loss'].append(loss)
        
        training_results['final_epsilon'] = self.agent.epsilon
        training_results['avg_loss'] = np.mean(training_results['avg_loss']) if training_results['avg_loss'] else 0
        
        return training_results
    
    def reset_environment(self, new_routes: List[Dict] = None):
        """Reinicia el ambiente con nuevas rutas (útil después de re-ejecutar ACO)"""
        if new_routes:
            self.current_routes = new_routes
            self.env.reset(new_routes)
        else:
            self.env.reset(self.current_routes)
        
        return {
            'success': True,
            'message': 'Ambiente reiniciado correctamente'
        }
    
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