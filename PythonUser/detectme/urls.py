from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #path('', views.home, name="home"),
    path('detectme', views.detectme, name='detectme'),
    path('', views.db_list, name='db_list'),
    path('home', views.db_list, name='db_list'),
    path('statistics', views.statistics, name='statistics'),
    path('analysis', views.analysis, name='analysis'),
]