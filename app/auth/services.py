from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import IntegrityError

def register_service(data):
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
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
        INSERT INTO users
        (name, password)
        VALUES
        (%s, %s)"""
    try:
        cur.execute(insert_user_query, (data['name'], hashed_password))
        config.commit()  # トランザクションをコミット
    except IntegrityError as e:
        return {'message': f"Error: {e}"}, 401  # エラーを表示

    cur.close()
    config.close()

    return {'message': 'Register successful'}    

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
    user_record = cur.fetchone()
    if user_record is None:
        print(user_record)
        return {'message': 'Invalid credentials'}, 401
    password_hash = user_record[2]

    cur.close()
    config.close()
    
    if password_hash and check_password_hash(password_hash, data['password']):
        return {'message': 'Login successful'}
    
    return {'message': 'Invalid credentials'}, 401