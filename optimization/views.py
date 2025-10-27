# vrp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json
from optimization.service.api_service import obtener_ordenes, obtener_vehiculos
from optimization.models import ACORouteResult

from optimization.service.aco_vrp import ACOVRPPD_MultiVehicle

@csrf_exempt
def run_aco(request):
   
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # 1. Extraer parámetros del algoritmo
            num_ants = data.get('num_ants', 20)
            iterations = data.get('iterations', 200)
            evaporation_rate = data.get('evaporation_rate', 0.1)
            alpha = data.get('alpha', 1)
            beta = data.get('beta', 3)
            max_distance = data.get('max_distance', 100)
            
            # 2. Obtener configuración de depósito
            depot = data.get('depot', {'lat': -12.087000, 'lng': -76.97180})
            depot_lat = float(depot.get('lat', -12.087000))
            depot_lng = float(depot.get('lng', -76.97180))
            
            # 3. Consumir API de órdenes
            try:
                orders_data = obtener_ordenes()
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=500)
            
            # 4. Consumir API de vehículos
            try:
                vehicles_data = obtener_vehiculos()
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=500)
            
            # 5. Validar que existan datos
            if not vehicles_data or len(vehicles_data) == 0:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron vehículos disponibles en el sistema.'
                }, status=400)
            
            if not orders_data or len(orders_data) == 0:
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron órdenes para procesar.'
                }, status=400)
            
            # 6. Transformar datos de vehículos al formato esperado por ACO
            # Formato: [capacity, max_distance]
            vehicles = []
            vehicle_id_map = []  # Para mantener el mapeo de índice a ID real
            
            for vehicle in vehicles_data:
                if vehicle.get('status') == 1:  # Solo vehículos activos
                    capacity = float(vehicle.get('capacity', 1000))
                    
                    vehicles.append([capacity, max_distance])
                    vehicle_id_map.append(vehicle.get('id'))
            
            if len(vehicles) == 0:
                return JsonResponse({
                    'success': False,
                    'error': 'No hay vehículos activos disponibles.'
                }, status=400)
            
            # 7. Construir lista de nodos a partir de las órdenes
            nodes = []
            
            # Agregar depósito (nodo 0)
            nodes.append(['depot', depot_lat, depot_lng, 0])
            
            # Mapeo de nodos a órdenes para la respuesta
            node_to_order_map = {}
            
            # Agregar nodos de órdenes (pickup y delivery)
            for order in orders_data:
                # Validar que la orden tenga el status adecuado
                if order.get('status') != '1':
                    continue
                
                # Extraer datos de pickup y delivery
                pickup_address = order.get('pickupAddress')
                delivery_address = order.get('deliveryAddress')
                
                if not pickup_address or not delivery_address:
                    continue
                
                pickup_lat = float(pickup_address.get('latitude'))
                pickup_lng = float(pickup_address.get('longitude'))
                delivery_lat = float(delivery_address.get('latitude'))
                delivery_lng = float(delivery_address.get('longitude'))
                capacity = float(order.get('weight', 0))
                
                # Índices de nodos para esta orden
                pickup_node_idx = len(nodes)
                delivery_node_idx = len(nodes) + 1
                
                # Nodo PICKUP (demanda positiva)
                nodes.append(['pickup', pickup_lat, pickup_lng, capacity])
                
                # Nodo DELIVERY (demanda negativa)
                nodes.append(['delivery', delivery_lat, delivery_lng, -capacity])
                
                # Guardar mapeo
                node_to_order_map[pickup_node_idx] = {
                    'order': order,
                    'type': 'pickup'
                }
                node_to_order_map[delivery_node_idx] = {
                    'order': order,
                    'type': 'delivery'
                }
            
            if len(nodes) <= 1:  # Solo depósito
                return JsonResponse({
                    'success': False,
                    'error': 'No se encontraron órdenes válidas para procesar.'
                }, status=400)
            
            # 8. Ejecutar el algoritmo ACO
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

            # 9. Preparar respuesta estructurada
            result = {
                'success': True,
                'best_distance': best_distance,
                'algorithm_params': {
                    'num_ants': num_ants,
                    'iterations': iterations,
                    'evaporation_rate': evaporation_rate,
                    'alpha': alpha,
                    'beta': beta
                },
                'routes': []
            }
            
            # 10. Construir respuesta detallada para cada ruta
            for route_idx, route in enumerate(best_routes):
                # Obtener el ID real del vehículo
                vehicle_id = vehicle_id_map[route_idx]
                
                route_info = {
                    'vehicle_id': vehicle_id,  # ID real del vehículo
                    'vehicle_capacity': vehicles[route_idx][0], 
                    'vehicle_max_distance': vehicles[route_idx][1],
                    'total_distance': 0,
                    'stops': []
                }
                
                # Calcular distancia total para esta ruta
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    segment_distance = aco.distances[from_node][to_node]
                    route_info['total_distance'] += segment_distance
                
                # Construir detalles de cada parada
                for node_idx in route:
                    if node_idx == 0:  # Depósito
                        route_info['stops'].append({
                            'type': 'depot',
                            'location': {
                                'lat': depot_lat,
                                'lng': depot_lng
                            }
                        })
                    else:
                        # Obtener información del nodo
                        node_data = nodes[node_idx]
                        
                        stop_info = {
                            'type': node_data[0],  # 'pickup' o 'delivery'
                            'location': {
                                'lat': node_data[1],
                                'lng': node_data[2]
                            },
                            'demand': node_data[3]
                        }
                        
                        # Agregar información de la orden si existe
                        if node_idx in node_to_order_map:
                            order_info = node_to_order_map[node_idx]
                            stop_info['order_id'] = order_info['order'].get('id')
                            stop_info['customer'] = order_info['order'].get('customer')
                        
                        route_info['stops'].append(stop_info)
                
                result['routes'].append(route_info)
            
            #11 Guardar el resultado en la base de datos
            try:
                ACORouteResult.objects.create(
                    executed_at=timezone.now(),
                    best_distance=best_distance,
                    parameters={
                        'num_ants': num_ants,
                        'iterations': iterations,
                        'evaporation_rate': evaporation_rate,
                        'alpha': alpha,
                        'beta': beta,
                        'max_distance': max_distance,
                    },
                    routes=result['routes'],
                    success=True
                )
            except Exception as e:
                # Si ocurre un error al guardar, lo registramos pero no interrumpimos la respuesta
                print(f"Error al guardar el resultado en la BD: {e}")

            return JsonResponse(result, safe=False)
        
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'JSON inválido en el cuerpo de la solicitud.'
            }, status=400)
        except KeyError as e:
            return JsonResponse({
                'success': False,
                'error': f'Campo requerido faltante: {str(e)}'
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
    

def get_last_route(request):
    
    try:
        last_result = ACORouteResult.objects.order_by('-executed_at').first()
        
        if not last_result:
            return JsonResponse({
                'success': False,
                'error': 'No hay rutas registradas aún.'
            }, status=404)

        data = {
            'success': True,
            'id': last_result.id,
            'executed_at': last_result.executed_at,
            'best_distance': last_result.best_distance,
            'parameters': last_result.parameters,
            'routes': last_result.routes
        }

        return JsonResponse(data, safe=False, status=200)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
    


    


#nuevos metodos para RL


"""
Nuevos endpoints para reasignación dinámica de rutas usando DQN.
Este archivo debe agregarse como optimization/views_dqn.py
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from optimization.service.route_manager import RouteManager

# Instancia global del RouteManager (en producción, considerar usar caché o base de datos)
_route_manager_instance = None


def get_route_manager():
    """Obtiene o crea la instancia del RouteManager"""
    global _route_manager_instance
    return _route_manager_instance


def set_route_manager(manager):
    """Establece la instancia del RouteManager"""
    global _route_manager_instance
    _route_manager_instance = manager


@csrf_exempt
def initialize_dqn(request):
    """
    Inicializa el sistema DQN con las rutas del algoritmo ACO.
    
    POST /optimization/dqn/initialize/
    Body: {
        "depot": {"lat": -12.087, "lng": -76.9718},
        "vehicles": [...],
        "routes": [...],
        "model_path": "models/dqn_vrp_model.pth" (opcional)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    try:
        data = json.loads(request.body)
        
        depot = data.get('depot', {'lat': -12.087, 'lng': -76.9718})
        vehicles = data.get('vehicles', [])
        routes = data.get('routes', [])
        model_path = data.get('model_path', 'models/dqn_vrp_model.pth')
        
        if not vehicles:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionaron vehículos.'
            }, status=400)
        
        if not routes:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionaron rutas iniciales.'
            }, status=400)
        
        # Crear RouteManager
        manager = RouteManager(
            depot=depot,
            vehicles=vehicles,
            initial_routes=routes,
            model_path=model_path
        )
        
        set_route_manager(manager)
        
        return JsonResponse({
            'success': True,
            'message': 'Sistema DQN inicializado correctamente',
            'state_size': manager.env.get_state_size(),
            'action_size': manager.env.get_action_size(),
            'epsilon': manager.agent.epsilon
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


@csrf_exempt
def add_order_dynamic(request):
    """
    Añade una nueva orden al sistema y la asigna dinámicamente usando DQN.
    
    POST /optimization/dqn/add-order/
    Body: {
        "order": {
            "id": 16,
            "weight": 500,
            "pickupAddress": {"latitude": -12.0, "longitude": -77.0},
            "deliveryAddress": {"latitude": -12.1, "longitude": -77.1},
            "customer": {...}
        },
        "training": true (opcional, default: true)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado. Llame a /dqn/initialize/ primero.'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        
        order = data.get('order')
        training = data.get('training', True)
        
        if not order:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionó la orden.'
            }, status=400)
        
        # Añadir orden
        result = manager.add_order(order, training=training)
        
        return JsonResponse({
            'success': True,
            'result': result,
            'updated_routes': manager.get_current_routes(),
            'pending_orders': manager.get_pending_orders()
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


@csrf_exempt
def cancel_order_dynamic(request):
    """
    Cancela una orden existente.
    
    POST /optimization/dqn/cancel-order/
    Body: {
        "order_id": 16
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        
        order_id = data.get('order_id')
        
        if not order_id:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionó el order_id.'
            }, status=400)
        
        # Cancelar orden
        result = manager.cancel_order(order_id)
        
        return JsonResponse({
            'success': True,
            'result': result,
            'updated_routes': manager.get_current_routes()
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


@csrf_exempt
def remove_vehicle_dynamic(request):
    """
    Remueve un vehículo y reasigna sus órdenes.
    
    POST /optimization/dqn/remove-vehicle/
    Body: {
        "vehicle_id": 25,
        "training": true (opcional, default: true)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        
        vehicle_id = data.get('vehicle_id')
        training = data.get('training', True)
        
        if not vehicle_id:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionó el vehicle_id.'
            }, status=400)
        
        # Remover vehículo
        result = manager.remove_vehicle(vehicle_id, training=training)
        
        return JsonResponse({
            'success': True,
            'result': result,
            'updated_routes': manager.get_current_routes(),
            'pending_orders': manager.get_pending_orders()
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


@csrf_exempt
def batch_add_orders(request):
    """
    Añade múltiples órdenes en batch.
    
    POST /optimization/dqn/batch-add-orders/
    Body: {
        "orders": [
            {"id": 16, "weight": 500, ...},
            {"id": 17, "weight": 600, ...}
        ],
        "training": true (opcional, default: true)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        
        orders = data.get('orders', [])
        training = data.get('training', True)
        
        if not orders:
            return JsonResponse({
                'success': False,
                'error': 'No se proporcionaron órdenes.'
            }, status=400)
        
        # Añadir órdenes en batch
        result = manager.batch_add_orders(orders, training=training)
        
        return JsonResponse({
            'success': True,
            'result': result,
            'updated_routes': manager.get_current_routes(),
            'pending_orders': manager.get_pending_orders()
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


@csrf_exempt
def get_current_routes(request):
    """
    Obtiene las rutas actuales.
    
    GET /optimization/dqn/current-routes/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        routes = manager.get_current_routes()
        pending_orders = manager.get_pending_orders()
        
        return JsonResponse({
            'success': True,
            'routes': routes,
            'pending_orders': pending_orders,
            'num_routes': len(routes),
            'num_pending': len(pending_orders)
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def get_statistics(request):
    """
    Obtiene estadísticas del sistema.
    
    GET /optimization/dqn/statistics/
    """
    if request.method != 'GET':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use GET.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        stats = manager.get_statistics()
        
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


@csrf_exempt
def save_model(request):
    """
    Guarda el modelo DQN entrenado.
    
    POST /optimization/dqn/save-model/
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        result = manager.save_model()
        
        return JsonResponse({
            'success': True,
            'message': 'Modelo guardado correctamente',
            'result': result
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }, status=500)


@csrf_exempt
def train_batch(request):
    """
    Entrena el agente DQN con las experiencias almacenadas.
    
    POST /optimization/dqn/train/
    Body: {
        "num_episodes": 100 (opcional, default: 100)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        data = json.loads(request.body) if request.body else {}
        num_episodes = data.get('num_episodes', 100)
        
        # Entrenar
        result = manager.train_batch(num_episodes=num_episodes)
        
        return JsonResponse({
            'success': True,
            'message': 'Entrenamiento completado',
            'result': result
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


@csrf_exempt
def reset_environment(request):
    """
    Reinicia el ambiente (útil después de re-ejecutar ACO).
    
    POST /optimization/dqn/reset/
    Body: {
        "routes": [...] (opcional, usa las rutas actuales si no se proporciona)
    }
    """
    if request.method != 'POST':
        return JsonResponse({
            'success': False,
            'error': 'Método no permitido. Use POST.'
        }, status=405)
    
    manager = get_route_manager()
    if not manager:
        return JsonResponse({
            'success': False,
            'error': 'Sistema DQN no inicializado.'
        }, status=400)
    
    try:
        data = json.loads(request.body) if request.body else {}
        new_routes = data.get('routes')
        
        result = manager.reset_environment(new_routes)
        
        return JsonResponse({
            'success': True,
            'result': result,
            'current_routes': manager.get_current_routes()
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