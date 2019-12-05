from django.urls import path
from . import views

urlpatterns = [
    path('api/image/', views.images),
    path('api/image/autocomplete', views.autocomplete),
    path('api/image/gps', views.gpssearch)
]
