from werkzeug.security import generate_password_hash, check_password_hash
import mysql.connector
from mysql.connector import IntegrityError, Error

def register_service(data):
    try:
        hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
        config = mysql.connector.connect(
            host='mysql-container',
            port='3306',
            user='root',
            password='pass',
            database='db'
        )

        config.ping(reconnect=True)

        cur = config.cursor()
        
        insert_user_query =  """
            INSERT INTO users
            (name, password)
            VALUES
            (%s, %s)"""
        
        cur.execute(insert_user_query, (data['name'], hashed_password))
        config.commit()

        return {'message': 'Register successful'}, 200
    
    except Error as e:
        print(f"Database error occurred: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'User already exists'}, 409

    finally:
        if cur:
            cur.close()
        if config:
            config.close()  

def login_service(data):
    try:
        config = mysql.connector.connect(
            host='mysql-container',
            port='3306',
            user='root',
            password='pass',
            database='db'
        )

        config.ping(reconnect=True)

        cur = config.cursor()

        cur.execute("SELECT * FROM users WHERE name = %s", (data['name'],))
        
        cur.statement
        user_record = cur.fetchone()
        if user_record is None:
            return ({'message': 'user not found'}), 401
        password_hash = user_record[2]

        if password_hash and check_password_hash(password_hash, data['password']):
            return {'message': 'Login successful'}, 200
        
        return {'message': 'Invalid credentials'}, 401
    
    except Error as e:
        print(f"Database error occurred: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 409

    finally:
        if cur:
            cur.close()
        if config:
            config.close()
    