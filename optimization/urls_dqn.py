"""
URLs para los endpoints de reasignación dinámica con DQN.
Este archivo debe agregarse como optimization/urls_dqn.py
"""

from django.urls import path
from . import views_dqn

urlpatterns = [
    # Inicialización del sistema DQN
    path('initialize/', views_dqn.initialize_dqn, name='dqn_initialize'),
    
    # Operaciones dinámicas
    path('add-order/', views_dqn.add_order_dynamic, name='dqn_add_order'),
    path('cancel-order/', views_dqn.cancel_order_dynamic, name='dqn_cancel_order'),
    path('remove-vehicle/', views_dqn.remove_vehicle_dynamic, name='dqn_remove_vehicle'),
    path('batch-add-orders/', views_dqn.batch_add_orders, name='dqn_batch_add'),
    
    # Consultas
    path('current-routes/', views_dqn.get_current_routes, name='dqn_current_routes'),
    path('statistics/', views_dqn.get_statistics, name='dqn_statistics'),
    
    # Entrenamiento y persistencia
    path('train/', views_dqn.train_batch, name='dqn_train'),
    path('save-model/', views_dqn.save_model, name='dqn_save_model'),
    path('reset/', views_dqn.reset_environment, name='dqn_reset'),
]