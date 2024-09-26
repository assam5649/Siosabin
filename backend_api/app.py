# app.py
from flask import Flask, request, jsonify
import mysql.connector
from get import Get

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    name = request.args.get('name')
    result = Get(str(name))
    return jsonify(message=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)
