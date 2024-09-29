from flask import jsonify, request
from . import auth
from .services import login_service, register_service

@auth.route('/register', methods=['POST'])
def register():
    print(request.data)  # Print raw request data for debugging
    try:
        data = request.get_json()  # Attempt to parse the JSON
        if data is None:
            return "Bad Request: No JSON received", 400
        result = register_service(data)
        return jsonify(result)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400

@auth.route('/login', methods=['POST'])
def login():
    print(request.data)  # Print raw request data for debugging
    try:
        data = request.get_json()  # Attempt to parse the JSON
        if data is None:
            return "Bad Request: No JSON received", 400
        response = login_service(data)
        return jsonify(response)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400    
    