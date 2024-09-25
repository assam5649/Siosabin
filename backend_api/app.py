# app.py
from flask import Flask, jsonify
import mysql.connector
from get import Get

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    name = 'a'
    result = Get(name)
    return jsonify(message=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)
