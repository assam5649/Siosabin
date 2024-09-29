from flask import jsonify, request
from . import auth
from .services import login_service, register_service

@auth.route('/register', methods=['POST'])
def register():
    data = request.json
    user = register_service(data)
    return jsonify(user), 201

@auth.route('/login', methods=['POST'])
def login():
    data = request.json
    response = login_service(data)
    return jsonify(response)