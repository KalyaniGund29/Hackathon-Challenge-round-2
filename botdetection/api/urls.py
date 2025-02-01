from django.urls import path
from .views import BotDetectionView

urlpatterns = [
    path('detect-bots/', BotDetectionView.as_view(), name='detect-bots'),
]