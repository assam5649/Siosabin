from flask import jsonify, request
from . import main
from .services import create_user_service, get_user_service, get_location_service, get_salinity_service

@main.route('/')
def index():
    return jsonify({'message': 'Welcome to the main Module...'})

@main.route('/data', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        if data is None:
            return "Bad Request: No JSON received", 400
        response, status_code = create_user_service(data)
        return jsonify(response), status_code
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return "Bad Request: Invalid JSON", 400
    
@main.route('/users/<int:device_id>', methods=['GET'])
def get_user(device_id):
    response, status_code = get_user_service(device_id)
    return jsonify(response), status_code

@main.route('/location', methods=['GET'])
def get_location():
    response, status_code = get_location_service()
    return jsonify(response), status_code

@main.route('/salinity', methods=['GET'])
def get_salinity():
    response, status_code = get_salinity_service()
    # 修正 5時間ごとにする。 現在は5つごとになってる
    return jsonify(response), status_code