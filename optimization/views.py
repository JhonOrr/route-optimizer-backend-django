# vrp/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
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
                orders_response = requests.get('http://localhost:8080/api/orders', timeout=10)
                orders_response.raise_for_status()
                orders_data = orders_response.json()
            except requests.RequestException as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Error al consumir API de órdenes: {str(e)}'
                }, status=500)
            
            # 4. Consumir API de vehículos
            try:
                vehicles_response = requests.get('http://localhost:8080/api/vehicles', timeout=10)
                vehicles_response.raise_for_status()
                vehicles_data = vehicles_response.json()
            except requests.RequestException as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Error al consumir API de vehículos: {str(e)}'
                }, status=500)
            
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