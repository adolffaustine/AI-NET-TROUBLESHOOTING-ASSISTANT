# troubleshooter/views.py

from django.shortcuts import render
from django.http import JsonResponse
import json
from .agent import run_gpon_agent_async
import logging
import traceback

# Setup logging for better error reporting in your console
logger = logging.getLogger(__name__)

def chat_view(request):
    """
    Renders the chat interface.
    """
    return render(request, 'troubleshooter/chat.html')

async def initial_message_api(request):
    """
    Handles the initial greeting from the AI assistant.
    """
    if request.method == 'GET':
        try:
            # This is the special prompt the agent is conditioned to respond to first.
            initial_prompt = "Initial greeting from Liquid Technical Support."
            
            # Use an empty history for the very first message
            response_content = await run_gpon_agent_async(initial_prompt, [])
            
            # Initialize history with the assistant's first message
            initial_history = [{'role': 'assistant', 'content': response_content}]
            
            return JsonResponse({'message': response_content, 'history': initial_history})
        
        except Exception as e:
            # CRITICAL FIX: Log the full traceback to help diagnose the 500 error
            logger.error(f"Error during initial message setup: {e}", exc_info=True)
            # Return a generic error to the client
            return JsonResponse({'error': f'Failed to get initial message due to server error: {e}'}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)


async def chat_api(request):
    """
    Handles chat messages via an API endpoint asynchronously.
    """
    if request.method == 'POST':
        try:
            # Ensure the request body is valid JSON
            try:
                data = json.loads(request.body)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)

            user_input = data.get('message')
            history_data = data.get('history', [])
            
            if user_input:
                response_content = await run_gpon_agent_async(user_input, history_data)
                
                # Append both user input and assistant response to history
                new_history = history_data + [{'role': 'user', 'content': user_input}, {'role': 'assistant', 'content': response_content}]
                
                return JsonResponse({'message': response_content, 'history': new_history})
            
            return JsonResponse({'error': 'No message provided'}, status=400)
        
        except Exception as e:
            # Log all other errors during the chat session
            logger.error(f"Error during chat API call: {e}", exc_info=True)
            return JsonResponse({'error': f'An unexpected error occurred during the chat session: {e}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)