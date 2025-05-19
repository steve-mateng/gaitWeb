from django.urls import path
from gait_recognition.views import recognize_gait, register_gait

urlpatterns = [
    path('api/recognize/', recognize_gait, name='recognize'),
    path('api/register-gait/', register_gait, name='register_gait'),
]
