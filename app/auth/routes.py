from flask import jsonify, request
from . import auth
from .services import login_service, register_service

@auth.route('/')
def index():
    return jsonify({'message': 'Welcome to the auth Module...'})

@auth.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if data is None:
            return "Bad Request: No JSON received", 400
        response, status_code = register_service(data)
        return (jsonify(response)), status_code
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400

@auth.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if data is None:
            return "Bad Request: No JSON received", 400
        response, status_code = login_service(data)
        return (jsonify(response)), status_code
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400    
    