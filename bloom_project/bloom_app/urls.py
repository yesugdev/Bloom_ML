from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict/', views.predict_bloom_level_api, name='predict_bloom_level_api'),
    path('api/emotion/', views.predict_emotion_api, name='predict_emotion'),
    path('api/emotion/excel/', views.export_emotion_excel, name='export_emotion_excel'),
]