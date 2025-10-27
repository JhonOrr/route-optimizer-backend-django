from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

class ACORouteResult(models.Model):
    executed_at = models.DateTimeField(default=timezone.now)
    best_distance = models.FloatField()
    parameters = models.JSONField()  # Guarda num_ants, iterations, etc.
    routes = models.JSONField()      # Guarda las rutas generadas
    success = models.BooleanField(default=True)

    def __str__(self):
        return f"ACO Result - {self.executed_at.strftime('%Y-%m-%d %H:%M:%S')}"
