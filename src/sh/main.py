import requests
import mysql.connector

url = 'https://www.exsample.com/'

r = requests.get(url)

conn = mysql.connector.connect(
    host='mysql-container',
    port='3306',
    user='root',
    password='pass',
    database='dbdata'
)

conn.ping(reconnect=True)

print(conn.is_connected())
print(r.text)