import os
import json
import requests
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return ('', 200, headers)

    try:
        # Get Fireworks API key from environment variables
        api_key = os.environ.get('FIREWORKS_API_KEY')
        if not api_key:
            return {
                'error': 'Server configuration error',
                'message': 'API key not configured. Please check server environment variables.'
            }, 500, headers

        # Extract the request body
        data = request.get_json()
        model = data.get('model')
        messages = data.get('messages')
        stream = data.get('stream', False)

        # Validate required fields
        if not model or not messages:
            return {
                'error': 'Bad request',
                'message': 'Missing required fields: model and messages'
            }, 400, headers

        # Prepare the request to Fireworks API
        fireworks_payload = {
            'model': model,
            'messages': messages,
            'temperature': data.get('temperature', 0.6),
            'top_p': data.get('top_p', 1),
            'top_k': data.get('top_k', 40),
            'max_tokens': 25000,  # Set to 25k as requested
            'presence_penalty': data.get('presence_penalty', 0),
            'frequency_penalty': data.get('frequency_penalty', 0),
            'stream': stream
        }

        # Add tools if provided
        tools = data.get('tools', [])
        if tools:
            fireworks_payload['tools'] = tools
            if 'tool_choice' in data:
                fireworks_payload['tool_choice'] = data['tool_choice']

        fireworks_headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        url = 'https://api.fireworks.ai/inference/v1/chat/completions'

        if stream:
            # Handle streaming response
            fireworks_headers['Accept'] = 'text/event-stream'
            response = requests.post(
                url,
                headers=fireworks_headers,
                json=fireworks_payload,
                stream=True
            )

            if not response.ok:
                return {
                    'error': 'API request failed',
                    'message': response.text
                }, response.status_code, headers

            def generate():
                for chunk in response.iter_lines():
                    if chunk:
                        yield f'data: {chunk.decode()}\n\n'

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={
                    **headers,
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
        else:
            # Handle non-streaming response
            response = requests.post(
                url,
                headers=fireworks_headers,
                json=fireworks_payload
            )

            if not response.ok:
                return {
                    'error': 'API request failed',
                    'message': response.text
                }, response.status_code, headers

            return response.json(), 200, headers

    except Exception as e:
        return {
            'error': 'Internal server error',
            'message': str(e)
        }, 500, headers

if __name__ == '__main__':
    app.run(debug=True)
