import mysql.connector


def connect(data):
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
    
    cur.execute("INSERT INTO users (name, password, location, remaining, salinity) VALUES (%s, %s, %s, %s, %s)", (data['name'], data['password'], data['location'], data['remaining'], data['salinity']))

    conn.commit()

    cur.close()
    conn.close()
