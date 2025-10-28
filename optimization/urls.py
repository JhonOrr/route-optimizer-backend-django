from django.urls import path
from . import views

urlpatterns = [
  path('run-aco/', views.run_aco, name='prueba_api'),
  path('sync-and-optimize/', views.sync_and_optimize_dqn, name='prueba_api'),
  path('sync-status/', views.get_sync_status
       , name='prueba_api'),
  path('current-routes/', views.get_current_routes, name='prueba_api'),
] 