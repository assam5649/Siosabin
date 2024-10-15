import mysql.connector
from mysql.connector import Error, IntegrityError
from .utils import categorize
import numpy as np

def create_user_service(data):
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

        category = categorize(data[salinity])

        cur.execute("INSERT INTO data (device_id, location, in_tank, out_tank, salinity) VALUES (%s, %s, %s, %s, %s)", (data['device_id'], data['location'], data['in_tank'], data['out_tank'], data['salinity']))

        config.commit()
        
        return data, 200
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'Database error occurred'}, 409

    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500

    finally:
        if cur:
            cur.close()
        if config:
            config.close()

def get_user_service(device_id):
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

        cur.execute("SELECT * FROM data WHERE device_id = %s ORDER BY id DESC LIMIT 4", (device_id,))
        
        cur.statement
        result = cur.fetchone()
        
        if result is None:
            return ({'message': 'device_id not found'}), 401

        return result, 200
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'User already exists'}, 409
    
    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500
        
    finally:
        if cur:
            cur.close()
        if config:
            config.close()

def get_location_service():
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

        cur.execute("SELECT device_id, MIN(location) AS location FROM data GROUP BY device_id;")
        
        cur.statement
        result = cur.fetchall()

        return result, 200
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'User already exists'}, 409
    
    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500
        
    finally:
        if cur:
            cur.close()
        if config:
            config.close()

def get_salinity_service():
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

        get_salinity_query = """
        SELECT
            salinity,
            FLOOR(TIMESTAMPDIFF(HOUR, created_at, NOW()) / 5) AS time_group
        FROM
            data
        WHERE
            created_at >= NOW() - INTERVAL 25 HOUR
        ORDER BY
            time_group;"""

        cur.execute(get_salinity_query)
        
        cur.statement
        result = cur.fetchall()
        result = np.array(result)
        result.reshape(-1)
        
        return result, 200
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'User already exists'}, 409
    
    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}, 500
        
    finally:
        if cur:
            cur.close()
        if config:
            config.close()