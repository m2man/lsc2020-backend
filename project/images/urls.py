from django.urls import path

from . import views

urlpatterns = [
    path('api/image/', views.images),
    path('api/timeline/', views.timeline)]
