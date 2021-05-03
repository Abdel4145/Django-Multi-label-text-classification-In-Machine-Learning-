from django.urls import path, include
from . import views


urlpatterns = [
 path('', views.home, name="home"),
 path('predict', views.predict, name="predict"),
 path('team', views.team, name="team"),
 path('about', views.about, name="about"),

]