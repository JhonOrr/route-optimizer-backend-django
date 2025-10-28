from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

class ACORouteResult(models.Model):
    executed_at = models.DateTimeField(default=timezone.now)
    best_distance = models.FloatField()
    parameters = models.JSONField()  # Guarda num_ants, iterations, etc.
    routes = models.JSONField()      # Guarda las rutas generadas
    success = models.BooleanField(default=True)

    def __str__(self):
        return f"ACO Result - {self.executed_at.strftime('%Y-%m-%d %H:%M:%S')}"
    

class OptimizationExecution(models.Model):
    """
    Almacena cada ejecución de optimización (ACO o DQN)
    """
    STATUS_CHOICES = [
        ('pending', 'Pendiente'),
        ('running', 'En Ejecución'),
        ('completed', 'Completado'),
        ('failed', 'Fallido'),
    ]
    
    ALGORITHM_CHOICES = [
        ('aco', 'Ant Colony Optimization'),
        ('dqn', 'Deep Q-Network'),
    ]
    
    id = models.AutoField(primary_key=True)
    algorithm = models.CharField(max_length=10, choices=ALGORITHM_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    executed_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Parámetros de ejecución
    parameters = models.JSONField(default=dict)
    
    # Resultados
    best_distance = models.FloatField(null=True, blank=True)
    routes = models.JSONField(null=True, blank=True)
    
    # Información adicional
    num_orders_processed = models.IntegerField(default=0)
    num_vehicles_used = models.IntegerField(default=0)
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-executed_at']
        
    def __str__(self):
        return f"Execution #{self.id} - {self.algorithm} - {self.status}"


class OrderSnapshot(models.Model):
    """
    Snapshot de una orden en un momento dado
    """
    execution = models.ForeignKey(OptimizationExecution, on_delete=models.CASCADE, related_name='orders')
    
    # Datos de la orden desde Spring Boot
    order_id = models.IntegerField()
    weight = models.FloatField()
    status = models.CharField(max_length=10)
    
    # Direcciones
    pickup_lat = models.FloatField()
    pickup_lng = models.FloatField()
    delivery_lat = models.FloatField()
    delivery_lng = models.FloatField()
    
    # Cliente
    customer_data = models.JSONField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        indexes = [
            models.Index(fields=['execution', 'order_id']),
        ]
        
    def __str__(self):
        return f"Order {self.order_id} - Execution {self.execution_id}"


class VehicleSnapshot(models.Model):
    """
    Snapshot de un vehículo en un momento dado
    """
    execution = models.ForeignKey(OptimizationExecution, on_delete=models.CASCADE, related_name='vehicles')
    
    # Datos del vehículo desde Spring Boot
    vehicle_id = models.IntegerField()
    capacity = models.FloatField()
    max_distance = models.FloatField(default=100.0)
    status = models.IntegerField()
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        indexes = [
            models.Index(fields=['execution', 'vehicle_id']),
        ]
        
    def __str__(self):
        return f"Vehicle {self.vehicle_id} - Execution {self.execution_id}"


class RouteAssignment(models.Model):
    """
    Almacena la asignación de una orden a un vehículo en una ruta
    """
    execution = models.ForeignKey(OptimizationExecution, on_delete=models.CASCADE, related_name='assignments')
    
    order_id = models.IntegerField()
    vehicle_id = models.IntegerField()
    
    # Información de la ruta
    stop_sequence = models.IntegerField()  # Posición en la secuencia de paradas
    stop_type = models.CharField(max_length=10)  # 'pickup' o 'delivery'
    
    # Métricas
    distance_to_next = models.FloatField(null=True, blank=True)
    cumulative_distance = models.FloatField(null=True, blank=True)
    cumulative_load = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['vehicle_id', 'stop_sequence']
        indexes = [
            models.Index(fields=['execution', 'vehicle_id']),
            models.Index(fields=['execution', 'order_id']),
        ]
        
    def __str__(self):
        return f"Order {self.order_id} -> Vehicle {self.vehicle_id} (Exec {self.execution_id})"


class DQNState(models.Model):
    """
    Almacena el estado del modelo DQN
    """
    id = models.AutoField(primary_key=True)
    last_execution = models.ForeignKey(
        OptimizationExecution, 
        on_delete=models.SET_NULL, 
        null=True,
        related_name='dqn_state'
    )
    
    # Estado del modelo
    model_path = models.CharField(max_length=255, default='models/dqn_vrp_model.pth')
    epsilon = models.FloatField(default=1.0)
    
    # Estadísticas de entrenamiento
    total_episodes = models.IntegerField(default=0)
    total_operations = models.IntegerField(default=0)
    avg_reward = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        
    def __str__(self):
        return f"DQN State - Epsilon: {self.epsilon:.4f}"


class OperationLog(models.Model):
    """
    Log de operaciones dinámicas (añadir orden, cancelar, remover vehículo)
    """
    OPERATION_CHOICES = [
        ('add_order', 'Añadir Orden'),
        ('cancel_order', 'Cancelar Orden'),
        ('remove_vehicle', 'Remover Vehículo'),
        ('batch_add', 'Añadir Batch de Órdenes'),
    ]
    
    execution = models.ForeignKey(
        OptimizationExecution, 
        on_delete=models.CASCADE, 
        related_name='operation_logs'
    )
    
    operation_type = models.CharField(max_length=20, choices=OPERATION_CHOICES)
    
    # Datos de la operación
    order_id = models.IntegerField(null=True, blank=True)
    vehicle_id = models.IntegerField(null=True, blank=True)
    
    # Resultado
    success = models.BooleanField(default=False)
    assigned = models.BooleanField(default=False)
    reward = models.FloatField(null=True, blank=True)
    
    # Detalles adicionales
    details = models.JSONField(null=True, blank=True)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['execution', 'operation_type']),
        ]
        
    def __str__(self):
        return f"{self.operation_type} - {self.created_at}"
