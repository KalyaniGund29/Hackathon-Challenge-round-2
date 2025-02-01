from rest_framework import serializers
from .models import DetectionResult

class DetectionResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionResult
        fields = ['username', 'confidence', 'is_bot']