from django.contrib import admin
from django.urls import path, include
from django.conf.urls import handler404


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('bonge.urls')),
]


handler404 = 'bonge.views.handler404'
