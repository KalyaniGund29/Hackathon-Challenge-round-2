from django.db import models

class DetectionResult(models.Model):
    username = models.CharField(max_length=255)
    confidence = models.FloatField()
    is_bot = models.BooleanField()

    def __str__(self):
        return f"{self.username} - {'Bot' if self.is_bot else 'Genuine'}"