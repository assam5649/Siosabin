# run.py
import time
import mysql.connector
from flask import Flask
from app import create_app

def wait_for_mysql():
    while True:
        try:
            connection = mysql.connector.connect(
                host='mysql-container',
                user='root',
                password='pass',
                database='db'
            )
            connection.close()
            print("MySQL is up and running!")
            break
        except mysql.connector.Error:
            print("Waiting for MySQL to be ready...")
            time.sleep(5)  # 5秒待つ

if __name__ == '__main__':
    wait_for_mysql()  # MySQLが起動するのを待つ
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)