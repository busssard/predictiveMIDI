from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/corpus/", include("corpus.urls")),
    path("api/training/", include("training.urls")),
]
