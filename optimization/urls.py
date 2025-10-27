from django.urls import path
from . import views

urlpatterns = [
  path('run-aco/', views.run_aco, name='prueba_api'),
  path('get-last-route/', views.get_last_route, name='get_last_route'),
]