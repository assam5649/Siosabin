from werkzeug.security import generate_password_hash, check_password_hash
from .models import user
import mysql.connector

def register_service(data):
    hashed_password = generate_password_hash(data['password'], method='sha256')
    # new_user = user(username=data['username'], password_hash=hashed_password)
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
    
    insert_user_query =  """
        INSERT INTO data
        (name, password)
        VALUES
        (%s, %s);"""


    cur.execute(insert_user_query, (data['username'], hashed_password))

    config.commit()

    get_name_query =  "SELECT * FROM users WHERE name = %s;"

    cur.execute(get_name_query, (hashed_password))

    
    cur.statement
    new_user_name = cur.fetchone()

    cur.close()
    config.close()

    return {'username': new_user_name}

def login_service(data):
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

    cur.execute("SELECT * FROM users WHERE name = %s", (data['name'],))
    
    cur.statement
    password_hash = cur.fetchone()

    cur.close()
    config.close()
    
    if password_hash and check_password_hash(password_hash, data['password']):
        return {'message': 'Login successful'}
    
    return {'message': 'Invalid credentials'}, 401