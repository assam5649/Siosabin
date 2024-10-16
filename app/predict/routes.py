from flask import jsonify, request
from . import predict
from .services import salinity, days_salinity

@predict.route('/')
def index():
    return jsonify({'message': 'Welcome to the predict Module...'})
    
@predict.route('/hour', methods=['GET'])
def get_salinity():
    response, status_code = salinity()
    return jsonify(response), status_code

@predict.route('/day', methods=['GET'])
def get_days_salinity():
    response, status_code = days_salinity()
    return jsonify(response), status_code