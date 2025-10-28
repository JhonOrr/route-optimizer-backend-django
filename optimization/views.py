"""
Views integradas para ACO y DQN con sincronización automática.
Incluye manejo de estados de órdenes y posiciones de vehículos.
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
            
            # Filtrar órdenes PENDIENTES y vehículos activos
            # ACO SOLO procesa órdenes en estado "pendiente" (1)
            all_orders = sync_service.fetch_current_data()[0]
            pending_orders = [
                o for o in all_orders 
                if sync_service._normalize_status(o.get('status')) == 'pendiente'
            ]
            active_vehicles = sync_service.get_active_vehicles()
            
            if not pending_orders:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron órdenes PENDIENTES para procesar. ACO requiere órdenes en estado "pendiente" (1).'
                }, status=400)
            
            if not active_vehicles:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron vehículos activos.'
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
                    'max_distance': max_distance,
                    'note': 'ACO solo procesa órdenes PENDIENTES (estado 1)'
                },
                num_orders_processed=len(pending_orders),
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
                
                # Construir nodos - TODAS las órdenes pendientes incluyen pickup + delivery
                nodes = [['depot', depot_lat, depot_lng, 0]]
                node_to_order_map = {}
                
                for order in pending_orders:
                    pickup_address = order.get('pickupAddress')
                    delivery_address = order.get('deliveryAddress')
                    
                    # Validar que tenga ambas direcciones
                    if not pickup_address or not delivery_address:
                        continue
                    
                    pickup_lat = float(pickup_address.get('latitude'))
                    pickup_lng = float(pickup_address.get('longitude'))
                    delivery_lat = float(delivery_address.get('latitude'))
                    delivery_lng = float(delivery_address.get('longitude'))
                    capacity = float(order.get('weight', 0))
                    
                    # Siempre agregar pickup + delivery (todas son pendientes)
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
                
                # Extraer posiciones de vehículos
                depot_dict = {'lat': depot_lat, 'lng': depot_lng}
                vehicle_positions = sync_service.extract_vehicle_positions_from_routes(routes, depot_dict)
                
                # Agregar posiciones a los vehículos
                for vehicle in active_vehicles:
                    vehicle['current_position'] = vehicle_positions.get(
                        vehicle['id'], 
                        depot_dict
                    )
                
                # Guardar snapshot
                sync_service.save_snapshot(execution, pending_orders, active_vehicles)
                sync_service.save_route_assignments(execution, routes)
                
                # Inicializar DQN con estas rutas
                vehicles_dqn = [
                    {
                        'id': v['id'],
                        'capacity': v['capacity'],
                        'max_distance': max_distance,
                        'current_position': v.get('current_position', depot_dict)
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
                    'dqn_initialized': True,
                    'orders_processed': {
                        'total': len(pending_orders),
                        'status': 'All orders are PENDIENTE (state 1) - pickup + delivery included'
                    },
                    'note': 'ACO initial optimization completed. Use /sync-and-optimize/ for dynamic reassignments.'
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
    Maneja estados de órdenes y posiciones de vehículos.
    
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
                    'max_distance': v['max_distance'],
                    'current_position': v.get('current_position', depot)
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
                'pending_orders': route_manager.get_pending_orders(),
                'vehicle_positions': route_manager.get_vehicle_positions()
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
                
                OperationLog.objects.create(
                    execution=execution,
                    operation_type='remove_vehicle',
                    vehicle_id=vehicle_id,
                    success=True,
                    details=result
                )
            
            # 2. Procesar cambios de estado de órdenes
            for status_change in changes['status_changed_orders']:
                order_id = status_change['order_id']
                new_status = status_change['new_status']
                
                result = route_manager.update_order_status(order_id, new_status, training=True)
                operation_results.append(result)
                
                OperationLog.objects.create(
                    execution=execution,
                    operation_type='status_change',
                    order_id=order_id,
                    success=result['success'],
                    details=result
                )
            
            # 3. Procesar órdenes canceladas explícitas
            for order_id in changes['cancelled_orders']:
                # Verificar si no fue procesada ya en cambios de estado
                already_processed = any(
                    sc['order_id'] == order_id 
                    for sc in changes['status_changed_orders']
                )
                
                if not already_processed:
                    result = route_manager.cancel_order(order_id)
                    operation_results.append(result)
                    
                    OperationLog.objects.create(
                        execution=execution,
                        operation_type='cancel_order',
                        order_id=order_id,
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
            vehicle_positions = route_manager.get_vehicle_positions()
            
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
            
            # Agregar posiciones actuales a vehículos
            for vehicle in active_vehicles:
                vehicle['current_position'] = vehicle_positions.get(
                    vehicle['id'],
                    {'lat': -12.087, 'lng': -76.9718}
                )
            
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
                'vehicle_positions': vehicle_positions,
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
def get_current_routes(request):
    """
    Obtiene las rutas actuales del RouteManager con posiciones de vehículos.
    
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
        positions = route_manager.get_vehicle_positions()
        
        return JsonResponse({
            'success': True,
            'routes': routes,
            'pending_orders': pending,
            'vehicle_positions': positions,
            'statistics': stats,
            'from_database': False
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)