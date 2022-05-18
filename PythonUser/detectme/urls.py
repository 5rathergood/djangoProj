from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = 'home'
urlpatterns = [
    #path('', views.home, name="home"),
    path('detectme', views.detectme, name='detectme'),
    path('', views.home, name='home'),
    path('home', views.home, name='home'),
    path('db_list', views.db_list, name='db_list'),
    path('statistics', views.statistics, name='statistics'),
    path('InitTodayTraffic', views.InitTodayTraffic, name='InitTodayTraffic'),
    path('InitTodayRecord', views.InitTodayRecord, name='InitTodayRecord'),
    path('analysis', views.AnalysisCreateView.as_view(), name='analysis'),
    path('analysis_create',views.AnalysisCreateView.as_view(),name='analysis_create'),
    path('summary', views.summary, name='summary'),
    path('write_line',views.write_line,name='write_line'),
