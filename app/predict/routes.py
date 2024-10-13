from flask import jsonify, request
from . import predict
from .services import salinity

@predict.route('/')
def index():
    return jsonify({'message': 'Welcome to the predict Module...'})
    
@predict.route('/salinity', methods=['GET'])
def get_salinity():
    response, status_code = salinity()
    return jsonify(response), status_code