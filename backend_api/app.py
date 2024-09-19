# app.py
from flask import Flask, jsonify
import mysql.connector

app = Flask(__name__)

@app.route('/api/hello', methods=['GET'])
def hello():
    Get()
    return jsonify(message="Hello, World!")



def Get():
    conn = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='dbdata'
    )

    conn.ping(reconnect=True)
    print(conn.is_connected())

    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE id = %s", [1])
    cur.statement
    print(cur.fetchone())

    cur.close()
    conn.close()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)
