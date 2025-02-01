from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predict/', views.predict_bloom_level_api, name='predict_bloom_level_api'),
]