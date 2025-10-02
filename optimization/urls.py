from django.urls import path
from . import views

urlpatterns = [
  path('run-aco/', views.run_aco, name='prueba_api'),
]