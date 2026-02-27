from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="/static/dashboard.html", permanent=False)),
    path("admin/", admin.site.urls),
    path("api/corpus/", include("corpus.urls")),
    path("api/training/", include("training.urls")),
]
