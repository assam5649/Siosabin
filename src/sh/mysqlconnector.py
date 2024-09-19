import mysql.connector


def connect():
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
    
    cur.execute("INSERT INTO users (id, name, password, location, remaining, salinity) VALUES (2, 'python', 'pythonpass', 'POINT(130.10 30.20)', 60, 0.4)")

    cur.execute("INSERT INTO data (id, max_temp, min_temp, ave_temp, ave_humidity, ave_windvelocity, max_windvelocity) VALUES (2, 20.1, 20.0, 20.2, 80, 2.4, 2.6)")

    conn.commit()

    cur.close()
    conn.close()
