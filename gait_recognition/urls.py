from django.urls import path
from gait_recognition.views import recognize_gait

urlpatterns = [
    path('api/recognize/', recognize_gait, name='recognize'),
]
