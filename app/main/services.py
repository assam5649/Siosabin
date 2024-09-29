import mysql.connector

def create_user_service(data):
    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)
    print(config.is_connected())

    cur = config.cursor()
    
    cur.execute("INSERT INTO users (name, password, location, remaining, salinity) VALUES (%s, %s, %s, %s, %s)", (data['name'], data['password'], data['location'], data['remaining'], data['salinity']))

    config.commit()

    cur.close()
    config.close()

def get_user_service(user_name):
    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)
    print(config.is_connected())

    cur = config.cursor()

    cur.execute("SELECT * FROM users WHERE name = %s ORDER BY id DESC LIMIT 1", (user_name,))

    
    cur.statement
    result = cur.fetchone()

    cur.close()
    config.close()

    return result