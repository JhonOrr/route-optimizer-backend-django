"""
Views integradas para ACO y DQN con sincronización automática.
Archivo: optimization/views.py
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json

from optimization.models import OptimizationExecution, DQNState, OperationLog, OrderSnapshot, VehicleSnapshot
from optimization.service.sync_service import SyncService
from optimization.service.aco_vrp import ACOVRPPD_MultiVehicle
from optimization.service.route_manager import RouteManager


# Variable global para mantener el RouteManager
_route_manager = None


def get_route_manager():
    global _route_manager
    return _route_manager


def set_route_manager(manager):
    global _route_manager
    _route_manager = manager


@csrf_exempt
def run_aco(request):
    """
    Ejecuta el algoritmo ACO desde cero.
    Este endpoint se usa para la optimización inicial.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Parámetros del algoritmo
            num_ants = data.get('num_ants', 50)
            iterations = data.get('iterations', 100)
            evaporation_rate = data.get('evaporation_rate', 0.1)
            alpha = data.get('alpha', 1)
            beta = data.get('beta', 2)
            max_distance = data.get('max_distance', 60)
            
            # Configuración de depósito
            depot = data.get('depot', {'lat': -12.087000, 'lng': -76.97180})
            depot_lat = float(depot.get('lat', -12.087000))
            depot_lng = float(depot.get('lng', -76.97180))
            
            # Sincronizar datos
            sync_service = SyncService()
            orders_data, vehicles_data = sync_service.fetch_current_data()
            
            # Validar datos
            if not vehicles_data or len(vehicles_data) == 0:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron vehículos disponibles.'
                }, status=400)
            
            # Filtrar órdenes y vehículos activos
            active_orders = [o for o in orders_data if o.get('status') == '1']
            active_vehicles = [v for v in vehicles_data if v.get('status') == 1]
            
            if not active_orders:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron órdenes activas para procesar.'
                }, status=400)
            
            # Crear ejecución en BD
            execution = OptimizationExecution.objects.create(
                algorithm='aco',
                status='running',
                parameters={
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta,
                    'max_distance': max_distance
                },
                num_orders_processed=len(active_orders),
                num_vehicles_used=len(active_vehicles)
            )
            
            try:
                # Preparar vehículos para ACO
                vehicles = []
                vehicle_id_map = []
                
                for vehicle in active_vehicles:
                    capacity = float(vehicle.get('capacity', 1000))
                    vehicles.append([capacity, max_distance])
                    vehicle_id_map.append(vehicle.get('id'))
                
                # Construir nodos
                nodes = [['depot', depot_lat, depot_lng, 0]]
                node_to_order_map = {}
                
                for order in active_orders:
                    pickup_address = order.get('pickupAddress')
                    delivery_address = order.get('deliveryAddress')
                    
                    if not pickup_address or not delivery_address:
                        continue
                    
                    pickup_lat = float(pickup_address.get('latitude'))
                    pickup_lng = float(pickup_address.get('longitude'))
                    delivery_lat = float(delivery_address.get('latitude'))
                    delivery_lng = float(delivery_address.get('longitude'))
                    capacity = float(order.get('weight', 0))
                    
                    pickup_node_idx = len(nodes)
                    delivery_node_idx = len(nodes) + 1
                    
                    nodes.append(['pickup', pickup_lat, pickup_lng, capacity])
                    nodes.append(['delivery', delivery_lat, delivery_lng, -capacity])
                    
                    node_to_order_map[pickup_node_idx] = {
                        'order': order,
                        'type': 'pickup'
                    }
                    node_to_order_map[delivery_node_idx] = {
                        'order': order,
                        'type': 'delivery'
                    }
                
                if len(nodes) <= 1:
                    execution.status = 'failed'
                    execution.error_message = 'No hay órdenes válidas'
                    execution.save()
                    return JsonResponse({
                        'success': False,
                        'error': 'No se encontraron órdenes válidas para procesar.'
                    }, status=400)
                
                # Ejecutar ACO
                aco = ACOVRPPD_MultiVehicle(
                    num_ants=num_ants,
                    iterations=iterations,
                    evaporation_rate=evaporation_rate,
                    alpha=alpha,
                    beta=beta,
                    vehicles=vehicles,
                    nodes=nodes
                )
                
                best_routes, best_distance = aco.run()
                
                # Construir respuesta
                routes = []
                for route_idx, route in enumerate(best_routes):
                    vehicle_id = vehicle_id_map[route_idx]
                    
                    route_info = {
                        'vehicle_id': vehicle_id,
                        'vehicle_capacity': vehicles[route_idx][0],
                        'vehicle_max_distance': vehicles[route_idx][1],
                        'total_distance': 0,
                        'stops': []
                    }
                    
                    # Calcular distancia total
                    for i in range(len(route) - 1):
                        from_node = route[i]
                        to_node = route[i + 1]
                        segment_distance = aco.distances[from_node][to_node]
                        route_info['total_distance'] += segment_distance
                    
                    # Construir paradas
                    for node_idx in route:
                        if node_idx == 0:
                            route_info['stops'].append({
                                'type': 'depot',
                                'location': {
                                    'lat': depot_lat,
                                    'lng': depot_lng
                                }
                            })
                        else:
                            node_data = nodes[node_idx]
                            stop_info = {
                                'type': node_data[0],
                                'location': {
                                    'lat': node_data[1],
                                    'lng': node_data[2]
                                },
                                'demand': node_data[3]
                            }
                            
                            if node_idx in node_to_order_map:
                                order_info = node_to_order_map[node_idx]
                                stop_info['order_id'] = order_info['order'].get('id')
                                stop_info['customer'] = order_info['order'].get('customer')
                            
                            route_info['stops'].append(stop_info)
                    
                    routes.append(route_info)
                
                # Actualizar ejecución
                execution.status = 'completed'
                execution.completed_at = timezone.now()
                execution.best_distance = best_distance
                execution.routes = routes
                execution.save()
                
                # Guardar snapshot
                sync_service.save_snapshot(execution, active_orders, active_vehicles)
                sync_service.save_route_assignments(execution, routes)
                
                # Inicializar DQN con estas rutas
                depot_dict = {'lat': depot_lat, 'lng': depot_lng}
                vehicles_dqn = [
                    {
                        'id': v['id'],
                        'capacity': v['capacity'],
                        'max_distance': max_distance
                    }
                    for v in active_vehicles
                ]
                
                route_manager = RouteManager(
                    depot=depot_dict,
                    vehicles=vehicles_dqn,
                    initial_routes=routes,
                    model_path='models/dqn_vrp_model.pth'
                )
                set_route_manager(route_manager)
                
                # Actualizar o crear DQN State
                dqn_state, created = DQNState.objects.get_or_create(
                    pk=1,
                    defaults={
                        'last_execution': execution,
                        'epsilon': route_manager.agent.epsilon
                    }
                )
                if not created:
                    dqn_state.last_execution = execution
                    dqn_state.epsilon = route_manager.agent.epsilon
                    dqn_state.save()
                
                return JsonResponse({
                    'success': True,
                    'execution_id': execution.id,
                    'executed_at': execution.executed_at.isoformat(),
                    'best_distance': best_distance,
                    'parameters': execution.parameters,
                    'routes': routes,
                    'dqn_initialized': True
                })
            
            except Exception as e:
                execution.status = 'failed'
                execution.error_message = str(e)
                execution.save()
                raise
        
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'JSON inválido en el cuerpo de la solicitud.'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e),
                'type': type(e).__name__
            }, status=500)
    else:
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)


@csrf_exempt
def sync_and_optimize_dqn(request):
    """
    Sincroniza datos desde Spring Boot, detecta cambios y aplica DQN.
    Este es el endpoint principal para optimización dinámica.
    
    POST /optimization/sync-and-optimize/
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    try:
        # Verificar si hay un RouteManager inicializado
        route_manager = get_route_manager()
        if not route_manager:
            # Intentar cargar desde la última ejecución
            last_execution = OptimizationExecution.objects.filter(
                status='completed'
            ).order_by('-executed_at').first()
            
            if not last_execution:
                return JsonResponse({
                    'success': False,
                    'error': 'No hay ejecución previa. Debe ejecutar ACO primero (/optimization/run-aco/).'
                }, status=400)
            
            # Reconstruir RouteManager desde última ejecución
            sync_service = SyncService()
            previous_orders, previous_vehicles = sync_service.load_previous_data(last_execution)
            
            depot = {'lat': -12.087, 'lng': -76.9718}
            vehicles_dqn = [
                {
                    'id': v['id'],
                    'capacity': v['capacity'],
                    'max_distance': v['max_distance']
                }
                for v in previous_vehicles if v.get('status') == 1
            ]
            
            route_manager = RouteManager(
                depot=depot,
                vehicles=vehicles_dqn,
                initial_routes=last_execution.routes,
                model_path='models/dqn_vrp_model.pth'
            )
            set_route_manager(route_manager)
        
        # Sincronizar y detectar cambios
        sync_service = SyncService()
        sync_result = sync_service.sync_and_detect()
        
        changes = sync_result['changes']
        
        if not changes['has_changes']:
            return JsonResponse({
                'success': True,
                'message': 'No hay cambios detectados. Rutas actuales se mantienen.',
                'sync_info': sync_result,
                'current_routes': route_manager.get_current_routes(),
                'pending_orders': route_manager.get_pending_orders()
            })
        
        # Crear nueva ejecución para registrar cambios DQN
        execution = OptimizationExecution.objects.create(
            algorithm='dqn',
            status='running',
            parameters={
                'changes_detected': changes
            }
        )
        
        try:
            operation_results = []
            
            # 1. Procesar vehículos removidos
            for vehicle_id in changes['removed_vehicles']:
                result = route_manager.remove_vehicle(vehicle_id, training=True)
                operation_results.append(result)
                
                # Registrar operación
                OperationLog.objects.create(
                    execution=execution,
                    operation_type='remove_vehicle',
                    vehicle_id=vehicle_id,
                    success=True,
                    details=result
                )
            
            # 2. Procesar órdenes canceladas
            for order_id in changes['cancelled_orders']:
                result = route_manager.cancel_order(order_id)
                operation_results.append(result)
                
                OperationLog.objects.create(
                    execution=execution,
                    operation_type='cancel_order',
                    order_id=order_id,
                    success=result['success'],
                    details=result
                )
            
            # 3. Procesar cambios de estado de órdenes
            for status_change in changes['status_changed_orders']:
                if status_change['new_status'] != '1':
                    # Si cambió a inactivo, cancelar
                    result = route_manager.cancel_order(status_change['order_id'])
                    operation_results.append(result)
                    
                    OperationLog.objects.create(
                        execution=execution,
                        operation_type='cancel_order',
                        order_id=status_change['order_id'],
                        success=result['success'],
                        details=result
                    )
            
            # 4. Procesar nuevas órdenes
            if changes['new_orders']:
                result = route_manager.batch_add_orders(changes['new_orders'], training=True)
                operation_results.append(result)
                
                # Registrar cada orden
                for detail in result.get('details', []):
                    OperationLog.objects.create(
                        execution=execution,
                        operation_type='add_order',
                        order_id=detail.get('order_id'),
                        vehicle_id=detail.get('vehicle_id'),
                        success=True,
                        assigned=detail.get('assigned', False),
                        reward=detail.get('reward'),
                        details=detail
                    )
            
            # Obtener rutas actualizadas
            updated_routes = route_manager.get_current_routes()
            pending_orders = route_manager.get_pending_orders()
            
            # Calcular distancia total
            total_distance = sum(route['total_distance'] for route in updated_routes)
            
            # Actualizar ejecución
            execution.status = 'completed'
            execution.completed_at = timezone.now()
            execution.best_distance = total_distance
            execution.routes = updated_routes
            execution.num_orders_processed = len(changes['new_orders'])
            execution.num_vehicles_used = len(updated_routes)
            execution.save()
            
            # Guardar snapshot actualizado
            active_orders = sync_service.get_active_orders()
            active_vehicles = sync_service.get_active_vehicles()
            sync_service.save_snapshot(execution, active_orders, active_vehicles)
            sync_service.save_route_assignments(execution, updated_routes)
            
            # Actualizar DQN State
            dqn_state = DQNState.objects.first()
            if dqn_state:
                dqn_state.last_execution = execution
                dqn_state.epsilon = route_manager.agent.epsilon
                dqn_state.total_operations += len(operation_results)
                dqn_state.save()
            
            # Guardar modelo DQN
            route_manager.save_model()
            
            return JsonResponse({
                'success': True,
                'execution_id': execution.id,
                'message': 'Optimización dinámica completada',
                'sync_info': sync_result,
                'changes_applied': {
                    'new_orders_added': len(changes['new_orders']),
                    'orders_cancelled': len(changes['cancelled_orders']),
                    'vehicles_removed': len(changes['removed_vehicles']),
                    'status_changes': len(changes['status_changed_orders'])
                },
                'operation_results': operation_results,
                'updated_routes': updated_routes,
                'pending_orders': pending_orders,
                'total_distance': total_distance,
                'statistics': route_manager.get_statistics()
            })
        
        except Exception as e:
            execution.status = 'failed'
            execution.error_message = str(e)
            execution.save()
            raise
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'JSON inválido en el cuerpo de la solicitud.'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_sync_status(request):
    """
    Obtiene el estado de sincronización sin ejecutar optimización.
    
    GET /optimization/sync-status/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    try:
        sync_service = SyncService()
        sync_result = sync_service.sync_and_detect()
        
        return JsonResponse({
            'success': True,
            'sync_info': sync_result
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_execution_history(request):
    """
    Obtiene el historial de ejecuciones.
    
    GET /optimization/execution-history/?limit=10
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    try:
        limit = int(request.GET.get('limit', 10))
        
        executions = OptimizationExecution.objects.all()[:limit]
        
        history = []
        for execution in executions:
            history.append({
                'id': execution.id,
                'algorithm': execution.algorithm,
                'status': execution.status,
                'executed_at': execution.executed_at.isoformat(),
                'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                'best_distance': execution.best_distance,
                'num_orders_processed': execution.num_orders_processed,
                'num_vehicles_used': execution.num_vehicles_used,
                'parameters': execution.parameters,
                'error_message': execution.error_message
            })
        
        return JsonResponse({
            'success': True,
            'history': history,
            'count': len(history)
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_execution_detail(request, execution_id):
    """
    Obtiene el detalle completo de una ejecución.
    
    GET /optimization/execution/<id>/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    try:
        execution = OptimizationExecution.objects.get(id=execution_id)
        
        # Obtener logs de operaciones si es DQN
        operation_logs = []
        if execution.algorithm == 'dqn':
            logs = OperationLog.objects.filter(execution=execution)
            operation_logs = [
                {
                    'operation_type': log.operation_type,
                    'order_id': log.order_id,
                    'vehicle_id': log.vehicle_id,
                    'success': log.success,
                    'assigned': log.assigned,
                    'reward': log.reward,
                    'created_at': log.created_at.isoformat(),
                    'details': log.details
                }
                for log in logs
            ]
        
        # Obtener órdenes procesadas
        orders = OrderSnapshot.objects.filter(execution=execution)
        order_list = [
            {
                'order_id': order.order_id,
                'weight': order.weight,
                'status': order.status,
                'pickup': {
                    'lat': order.pickup_lat,
                    'lng': order.pickup_lng
                },
                'delivery': {
                    'lat': order.delivery_lat,
                    'lng': order.delivery_lng
                }
            }
            for order in orders
        ]
        
        # Obtener vehículos
        vehicles = VehicleSnapshot.objects.filter(execution=execution)
        vehicle_list = [
            {
                'vehicle_id': vehicle.vehicle_id,
                'capacity': vehicle.capacity,
                'max_distance': vehicle.max_distance,
                'status': vehicle.status
            }
            for vehicle in vehicles
        ]
        
        return JsonResponse({
            'success': True,
            'execution': {
                'id': execution.id,
                'algorithm': execution.algorithm,
                'status': execution.status,
                'executed_at': execution.executed_at.isoformat(),
                'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
                'best_distance': execution.best_distance,
                'num_orders_processed': execution.num_orders_processed,
                'num_vehicles_used': execution.num_vehicles_used,
                'parameters': execution.parameters,
                'routes': execution.routes,
                'error_message': execution.error_message
            },
            'orders': order_list,
            'vehicles': vehicle_list,
            'operation_logs': operation_logs
        })
    
    except OptimizationExecution.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': f'Ejecución con ID {execution_id} no encontrada.'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_current_routes(request):
    """
    Obtiene las rutas actuales del RouteManager.
    
    GET /optimization/current-routes/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    try:
        route_manager = get_route_manager()
        
        if not route_manager:
            # Obtener última ejecución
            last_execution = OptimizationExecution.objects.filter(
                status='completed'
            ).order_by('-executed_at').first()
            
            if not last_execution:
                return JsonResponse({
                    'success': False,
                    'error': 'No hay rutas disponibles. Ejecute ACO primero.'
                }, status=404)
            
            return JsonResponse({
                'success': True,
                'routes': last_execution.routes,
                'execution_id': last_execution.id,
                'executed_at': last_execution.executed_at.isoformat(),
                'from_database': True
            })
        
        routes = route_manager.get_current_routes()
        pending = route_manager.get_pending_orders()
        stats = route_manager.get_statistics()
        
        return JsonResponse({
            'success': True,
            'routes': routes,
            'pending_orders': pending,
            'statistics': stats,
            'from_database': False
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_dqn_statistics(request):
    """
    Obtiene estadísticas del sistema DQN.
    
    GET /optimization/dqn-statistics/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    try:
        dqn_state = DQNState.objects.first()
        
        if not dqn_state:
            return JsonResponse({
                'success': False,
                'error': 'Sistema DQN no inicializado.'
            }, status=404)
        
        route_manager = get_route_manager()
        
        stats = {
            'dqn_state': {
                'epsilon': dqn_state.epsilon,
                'total_episodes': dqn_state.total_episodes,
                'total_operations': dqn_state.total_operations,
                'last_execution_id': dqn_state.last_execution_id,
                'updated_at': dqn_state.updated_at.isoformat()
            }
        }
        
        if route_manager:
            stats['current_statistics'] = route_manager.get_statistics()
        
        # Estadísticas de operaciones
        total_operations = OperationLog.objects.count()
        successful_assignments = OperationLog.objects.filter(
            operation_type='add_order',
            assigned=True
        ).count()
        
        stats['operation_statistics'] = {
            'total_operations': total_operations,
            'successful_assignments': successful_assignments,
            'assignment_rate': successful_assignments / total_operations if total_operations > 0 else 0
        }
        
        return JsonResponse({
            'success': True,
            'statistics': stats
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)