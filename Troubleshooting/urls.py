# troubleshooter/urls.py (Example)
from django.urls import path
from . import views # Assuming your views file is named views.py

urlpatterns = [
    # The main chat interface view
    path('', views.chat_view, name='chat_interface'), 
    
    # The main chat API for user messages
    path('api/chat/', views.chat_api, name='chat_api'),
    
    # NEW: URL for the initial greeting message on page load
    path('api/initial-message/', views.initial_message_api, name='initial_message_api'), 
]