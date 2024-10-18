import mysql.connector
from mysql.connector import Error, IntegrityError
from .utils import categorize
import numpy as np
import json
from collections import defaultdict

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

        category = categorize(data['salinity'])

        print("=-----------------------")

        cur.execute("INSERT INTO data (device_id, location, in_tank, out_tank, salinity) VALUES (%s, %s, %s, %s, %s)", (data['device_id'], data['location'], data['in_tank'], data['out_tank'], category))

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

        cur.execute("SELECT * FROM data WHERE device_id = %s ORDER BY id DESC LIMIT 1", (device_id,))
        
        cur.statement
        result = cur.fetchall()
        
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
        SELECT salinity, location,
            CASE
                WHEN created_at >= NOW() - INTERVAL 5 HOUR THEN 'Group 1'
                WHEN created_at >= NOW() - INTERVAL 10 HOUR THEN 'Group 2'
                WHEN created_at >= NOW() - INTERVAL 15 HOUR THEN 'Group 3'
                WHEN created_at >= NOW() - INTERVAL 20 HOUR THEN 'Group 4'
                WHEN created_at >= NOW() - INTERVAL 25 HOUR THEN 'Group 5'
            END AS time_group
        FROM data
        WHERE created_at >= NOW() - INTERVAL 25 HOUR
        ORDER BY id DESC;"""

        cur.execute(get_salinity_query)

        data = cur.fetchall()

        grouped_data = defaultdict(list)

        for entry in data:
            grouped_data[entry[2]].append(entry)

        result = [grouped_data[f"Group {i+1}"] for i in range(5)]
        
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