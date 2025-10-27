import os
import json
from typing import List, Dict, Optional, Tuple
from .dqn_environment import VRPEnvironment
from .dqn_agent import DQNAgent


class RouteManager:
    """
    Gestor central para reasignación dinámica de rutas usando DQN.
    
    Coordina:
    - Adición de nuevas órdenes
    - Cancelación de órdenes
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
        
        Args:
            order: Nueva orden a asignar
            training: Si es True, entrena el agente durante la asignación
            
        Returns:
            Dict con resultado de la operación
        """
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
            'assigned': info['assigned'],
            'vehicle_id': info.get('vehicle_id'),
            'reward': reward,
            'loss': loss,
            'epsilon': self.agent.epsilon
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
        
        Args:
            vehicle_id: ID del vehículo a remover
            training: Si es True, entrena durante la reasignación
            
        Returns:
            Dict con resultado de la operación
        """
        # Obtener órdenes que deben ser reasignadas
        orders_to_reassign = self.env.remove_vehicle(vehicle_id)
        
        reassignment_results = []
        successful_reassignments = 0
        pending_orders = []
        
        # Intentar reasignar cada orden
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
        
        for order in orders:
            result = self.add_order(order, training=training)
            results.append(result)
            
            if result['assigned']:
                successful_assignments += 1
            else:
                pending_orders.append(order.get('id'))
        
        summary = {
            'type': 'batch_add_orders',
            'total_orders': len(orders),
            'successful_assignments': successful_assignments,
            'pending_orders': len(pending_orders),
            'pending_order_ids': pending_orders,
            'assignment_rate': successful_assignments / len(orders) if orders else 0,
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
                vehicle_utilization.append({
                    'vehicle_id': vehicle['id'],
                    'utilization': utilization,
                    'distance_used': route['total_distance'],
                    'distance_capacity': vehicle['max_distance']
                })
        
        # Estadísticas del agente
        agent_stats = self.agent.get_stats()
        
        return {
            'total_distance': total_distance,
            'num_vehicles_used': len(self.current_routes),
            'num_pending_orders': len(self.env.pending_orders),
            'vehicle_utilization': vehicle_utilization,
            'avg_vehicle_utilization': np.mean([v['utilization'] for v in vehicle_utilization]) if vehicle_utilization else 0,
            'agent_epsilon': self.agent.epsilon,
            'total_operations': len(self.operations_log),
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


# Importar numpy si no está disponible globalmente
try:
    import numpy as np
except ImportError:
    import warnings
    warnings.warn("NumPy no está instalado. Algunas funciones pueden no funcionar correctamente.")