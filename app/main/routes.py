from flask import jsonify, request
from . import main
from .services import create_user_service, get_user_service

@main.route('/')
def index():
    return jsonify({'message': 'Welcome to the main Module...'})

@main.route('/data', methods=['POST'])
def create_user():#POST
    print(request.data)  # Print raw request data for debugging
    try:
        data = request.get_json()  # Attempt to parse the JSON
        if data is None:
            return "Bad Request: No JSON received", 400
        create_user_service(data)
        return jsonify(data)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400
    
@main.route('/users/<string:user_name>', methods=['GET'])
def get_user(user_name):
    result = get_user_service(str(user_name))
    return jsonify(message=result)
    