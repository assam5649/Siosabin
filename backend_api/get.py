import mysql.connector

def Get(name):
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

    cur.execute("SELECT * FROM users WHERE name = %s ORDER BY id DESC LIMIT 1", (name,))

    
    cur.statement
    result = cur.fetchone()

    cur.close()
    conn.close()

    return result