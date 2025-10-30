"""
Views integradas para ACO y DQN con sincronización automática.
Incluye manejo de estados de órdenes y posiciones de vehículos.
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json

from optimization.models import OptimizationExecution, DQNState, OperationLog, LastACORoute
from optimization.service.sync_service import SyncService
from optimization.service.aco_vrp import ACOVRPPD_MultiVehicle
from optimization.service.route_manager import RouteManager

from optimization.service.dynamic_aco_vrp import ACOVRPPD_MultiVehicle_Dynamic


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
            
            # Se extrae parámetros del algoritmo
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
            
            # Sincronizar datos con Spring Boot
            sync_service = SyncService()
            all_orders, vehicles_data = sync_service.fetch_current_data()
            
            # Filtrar órdenes PENDIENTES y vehículos activos
            # ACO SOLO procesa órdenes en estado "pendiente" (1)
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
            
            LastACORoute.objects.all().delete()  # Eliminar rutas anteriores
            LastACORoute.objects.create(
                executed_at=timezone.now(),
                best_distance=best_distance,
                parameters={
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta,
                    'max_distance': max_distance
                },
                routes=routes,
                num_orders_processed=len(pending_orders),
                num_vehicles_used=len(active_vehicles)
            )

            return JsonResponse({
                'success': True,
                'executed_at': timezone.now().isoformat(),
                'best_distance': best_distance,
                'parameters': {
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta,
                    'max_distance': max_distance
                },
                'routes': routes,
                'orders_processed': {
                    'total': len(pending_orders),
                    'status': 'All orders are PENDIENTE (state 1) - pickup + delivery included'
                }
            })
        
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
def run_aco_dynamic(request):
    """
    Ejecuta el algoritmo ACO dinámico para vehículos que ya están en camino.
    
    Este endpoint optimiza rutas considerando:
    - Estado actual de cada vehículo (posición, capacidad, distancia recorrida)
    - Estado de las órdenes:
        * PENDIENTE (1): incluir pickup + delivery
        * RECOGIDO (2): solo incluir delivery (pickup ya realizado)
        * COMPLETA (3): ignorar
        * POSPUESTA (4): ignorar
    
    Puede ser llamado SIN parámetros (usa valores por defecto) o CON parámetros personalizados.
    
    Si un vehículo no tiene datos de estado actual:
    - current_latitude = depot_lat
    - current_longitude = depot_lng
    - current_capacity = 0
    - current_distance = 0
    """
    if request.method == 'POST':
        try:
            # Intentar parsear el body, si está vacío o es inválido, usar diccionario vacío
            try:
                data = json.loads(request.body) if request.body else {}
            except json.JSONDecodeError:
                data = {}
            
            # Parámetros del algoritmo ACO con valores por defecto (iguales a run_aco)
            num_ants = data.get('num_ants', 50)
            iterations = data.get('iterations', 100)
            evaporation_rate = data.get('evaporation_rate', 0.1)
            alpha = data.get('alpha', 1)
            beta = data.get('beta', 2)
            max_distance = data.get('max_distance', 60)  # Mismo default que run_aco
            
            # Configuración de depósito (valores por defecto para Lima)
            depot = data.get('depot', {'lat': -12.087000, 'lng': -76.97180})
            depot_lat = float(depot.get('lat', -12.087000))
            depot_lng = float(depot.get('lng', -76.97180))
            
            # Sincronizar datos con Spring Boot
            sync_service = SyncService()
            all_orders, vehicles_data = sync_service.fetch_current_data()
            
            # Filtrar vehículos activos
            active_vehicles = sync_service.get_active_vehicles()
            
            if not active_vehicles:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron vehículos activos.'
                }, status=400)
            
            # Filtrar órdenes según estado
            # PENDIENTE (1) -> incluir pickup + delivery
            # RECOGIDO (2) -> solo delivery
            # COMPLETA (3) y POSPUESTA (4) -> ignorar
            
            pending_orders = []  # Estado 1
            picked_up_orders = []  # Estado 2
            
            for order in all_orders:
                status = sync_service._normalize_status(order.get('status'))
                
                if status == 'pendiente':  # Estado 1
                    pending_orders.append(order)
                elif status == 'recogido':  # Estado 2
                    picked_up_orders.append(order)
                # Estados 3 (completa) y 4 (pospuesta) se ignoran automáticamente
            
            if not pending_orders and not picked_up_orders:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron órdenes activas para procesar. Se requieren órdenes en estado PENDIENTE (1) o RECOGIDO (2).'
                }, status=400)
            
            # Preparar vehículos para ACO dinámico
            # Formato: (capacidad_actual, capacidad_maxima, distancia_recorrida, distancia_maxima, lat_actual, lon_actual)
            vehicles = []
            vehicle_id_map = []
            vehicle_info_map = {}  # Para almacenar info completa del vehículo
            
            for vehicle in active_vehicles:
                # Extraer capacidad máxima (requerida)
                max_capacity = vehicle.get('capacity')
                if max_capacity is None:
                    max_capacity = 1000.0  # Valor por defecto si no existe
                else:
                    max_capacity = float(max_capacity)
                
                # Extraer capacidad actual - si es None o no existe, usar 0
                current_capacity = vehicle.get('currentCapacity')
                if current_capacity is None:
                    current_capacity = 0.0
                else:
                    current_capacity = float(current_capacity)
                
                # Extraer distancia recorrida - si es None o no existe, usar 0
                current_distance = vehicle.get('currentDistance')
                if current_distance is None:
                    current_distance = 0.0
                else:
                    current_distance = float(current_distance)
                
                # Calcular distancia máxima permitida
                max_distance_per_shift = max_distance
                
                # Extraer posición actual - si es None o no existe, usar posición del depósito
                current_lat = vehicle.get('currentLatitude')
                if current_lat is None:
                    current_lat = depot_lat
                else:
                    current_lat = float(current_lat)
                
                current_lng = vehicle.get('currentLongitude')
                if current_lng is None:
                    current_lng = depot_lng
                else:
                    current_lng = float(current_lng)
                
                vehicle_id = vehicle.get('id')
                
                vehicles.append([
                    current_capacity,           # capacidad_actual
                    max_capacity,               # capacidad_maxima
                    current_distance,           # distancia_recorrida
                    max_distance_per_shift,     # distancia_maxima
                    current_lat,                # lat_inicial (posición actual)
                    current_lng                 # lon_inicial (posición actual)
                ])
                vehicle_id_map.append(vehicle_id)
                vehicle_info_map[vehicle_id] = {
                    'capacity': max_capacity,
                    'max_distance': max_distance_per_shift
                }
            
            # Construir nodos dinámicamente según estado de órdenes
            nodes = [['depot', depot_lat, depot_lng, 0]]  # Nodo 0: depósito
            node_to_order_map = {}
            order_id_to_indices = {}  # Mapear order_id a índices de nodos
            
            # Procesar órdenes PENDIENTES (1): agregar pickup + delivery
            for order in pending_orders:
                pickup_address = order.get('pickupAddress')
                delivery_address = order.get('deliveryAddress')
                
                if not pickup_address or not delivery_address:
                    continue
                
                try:
                    pickup_lat = float(pickup_address.get('latitude'))
                    pickup_lng = float(pickup_address.get('longitude'))
                    delivery_lat = float(delivery_address.get('latitude'))
                    delivery_lng = float(delivery_address.get('longitude'))
                    weight = float(order.get('weight', 0))
                except (TypeError, ValueError):
                    # Si hay problemas con las coordenadas, saltar esta orden
                    continue
                
                pickup_node_idx = len(nodes)
                delivery_node_idx = len(nodes) + 1
                
                nodes.append(['pickup', pickup_lat, pickup_lng, weight])
                nodes.append(['delivery', delivery_lat, delivery_lng, -weight])
                
                order_id = order.get('id')
                
                node_to_order_map[pickup_node_idx] = {
                    'order': order,
                    'type': 'pickup',
                    'order_status': 'pendiente'
                }
                node_to_order_map[delivery_node_idx] = {
                    'order': order,
                    'type': 'delivery',
                    'order_status': 'pendiente'
                }
                
                order_id_to_indices[order_id] = {
                    'pickup': pickup_node_idx,
                    'delivery': delivery_node_idx
                }
            
            # Procesar órdenes RECOGIDAS (2): agregar solo delivery
            for order in picked_up_orders:
                delivery_address = order.get('deliveryAddress')
                
                if not delivery_address:
                    continue
                
                try:
                    delivery_lat = float(delivery_address.get('latitude'))
                    delivery_lng = float(delivery_address.get('longitude'))
                    weight = float(order.get('weight', 0))
                except (TypeError, ValueError):
                    # Si hay problemas con las coordenadas, saltar esta orden
                    continue
                
                # Para órdenes ya recogidas, el peso es negativo (se entregará)
                delivery_node_idx = len(nodes)
                nodes.append(['delivery', delivery_lat, delivery_lng, -weight])
                
                order_id = order.get('id')
                
                node_to_order_map[delivery_node_idx] = {
                    'order': order,
                    'type': 'delivery',
                    'order_status': 'recogido'
                }
                
                order_id_to_indices[order_id] = {
                    'delivery': delivery_node_idx
                }
            
            if len(nodes) <= 1:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron nodos válidos para procesar. Verifique que las órdenes tengan direcciones válidas.'
                }, status=400)
            
            # Ejecutar ACO DINÁMICO
            aco = ACOVRPPD_MultiVehicle_Dynamic(
                num_ants=num_ants,
                iterations=iterations,
                evaporation_rate=evaporation_rate,
                alpha=alpha,
                beta=beta,
                vehicles=vehicles,
                nodes=nodes
            )
            
            best_routes, best_distance = aco.run()
            
            # Construir respuesta en el MISMO formato que run_aco
            routes = []
            orders_processed = 0
            vehicles_used = 0
            
            for route_idx, route in enumerate(best_routes):
                vehicle_id = vehicle_id_map[route_idx]
                vehicle_info = vehicle_info_map[vehicle_id]
                
                # Contar si el vehículo tiene stops útiles (más que solo depot)
                has_useful_stops = False
                stops = []
                route_distance = 0.0
                orders_in_route = set()
                
                # Calcular distancia de la ruta
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    
                    # Calcular distancia del segmento
                    segment_distance = aco.distances[from_node][to_node]
                    route_distance += segment_distance
                
                # Construir paradas de la ruta
                for node_idx in route:
                    # Nodo de inicio del vehículo (posición actual) - OMITIR en la salida
                    if node_idx >= len(nodes):
                        continue
                    
                    # Depósito
                    if node_idx == 0:
                        stops.append({
                            'type': 'depot',
                            'location': {
                                'lat': depot_lat,
                                'lng': depot_lng
                            }
                        })
                    # Nodos de pickup/delivery
                    else:
                        has_useful_stops = True
                        node_data = nodes[node_idx]
                        stop_info = {
                            'type': node_data[0],  # 'pickup' o 'delivery'
                            'location': {
                                'lat': node_data[1],
                                'lng': node_data[2]
                            },
                            'demand': node_data[3]
                        }
                        
                        if node_idx in node_to_order_map:
                            order_info = node_to_order_map[node_idx]
                            order_id = order_info['order'].get('id')
                            
                            stop_info['order_id'] = order_id
                            stop_info['customer'] = order_info['order'].get('customer')
                            
                            orders_in_route.add(order_id)
                        
                        stops.append(stop_info)
                
                # Solo incluir vehículos que tengan al menos una parada útil
                if has_useful_stops:
                    vehicles_used += 1
                    orders_processed += len(orders_in_route)
                
                # Construir objeto de ruta (siempre, incluso si está vacío)
                route_info = {
                    'vehicle_id': vehicle_id,
                    'vehicle_capacity': vehicle_info['capacity'],
                    'vehicle_max_distance': vehicle_info['max_distance'],
                    'total_distance': route_distance,
                    'stops': stops
                }
                
                routes.append(route_info)
            
            # Guardar resultados en la base de datos
            LastACORoute.objects.all().delete()
            LastACORoute.objects.create(
                executed_at=timezone.now(),
                best_distance=best_distance,
                parameters={
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta,
                    'max_distance': max_distance
                },
                routes=routes,
                num_orders_processed=orders_processed,
                num_vehicles_used=vehicles_used
            )
            
            return JsonResponse({
                'success': True,
                'executed_at': timezone.now().isoformat(),
                'best_distance': best_distance,
                'parameters': {
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta,
                    'max_distance': max_distance
                },
                'routes': routes,
                'orders_processed': orders_processed,
                'vehicles_used': vehicles_used
            })
        
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'JSON inválido en el cuerpo de la solicitud.'
            }, status=400)
        except Exception as e:
            import traceback
            return JsonResponse({
                'success': False,
                'error': str(e),
                'type': type(e).__name__,
                'traceback': traceback.format_exc()
            }, status=500)
    else:
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)

def get_last_route(request):
    """
    Obtiene la última ruta ejecutada por ACO.
    Endpoint: GET /get-last-route/
    """
    if request.method == 'GET':
        try:
            # Obtener la última ruta guardada
            last_route = LastACORoute.objects.first()
            
            if not last_route:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontró ninguna ruta ejecutada previamente.'
                }, status=404)
            
            return JsonResponse({
                'success': True,
                'executed_at': last_route.executed_at.isoformat(),
                'best_distance': last_route.best_distance,
                'parameters': last_route.parameters,
                'routes': last_route.routes,
                'orders_processed': last_route.num_orders_processed,
                'vehicles_used': last_route.num_vehicles_used
            })
        
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e),
                'type': type(e).__name__
            }, status=500)
    else:
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
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