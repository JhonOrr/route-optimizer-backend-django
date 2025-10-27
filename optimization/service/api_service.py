import requests

def obtener_ordenes():
    try:
        response = requests.get('http://localhost:8080/api/orders', timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise Exception(f"Error al consumir API de órdenes: {str(e)}")
  
def obtener_vehiculos():
    try:
        response = requests.get('http://localhost:8080/api/vehicles', timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise Exception(f"Error al consumir API de vehículos: {str(e)}")