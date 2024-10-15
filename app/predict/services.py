import mysql.connector
from mysql.connector import Error, IntegrityError
import torch
import numpy as np

def insert_features(forecast):
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
        
        insert_features_query = """
        INSERT INTO features
        (year, month, day, hour, precipitation, tempMax, tempMin)
        VALUES
        (%s, %s, %s, %s, %s, %s, %s);"""

        cur.execute(insert_features_query, (forecast['year'], forecast['month'], forecast['day'], forecast['hour'], forecast['precipitation'], forecast['tempMax'], forecast['tempMin']))

        config.commit()
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'Database error occurred'}

    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}

    finally:
        if cur:
            cur.close()
        if config:
            config.close()

def save_target(hour, target):
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

        insert_target_query = """
        INSERT INTO target
        (future_offset, future_value)
        VALUES
        (%s, %s);"""

        if hour == 5.0:
            hour = 17
        elif hour == 17.0:
            hour = 5
        else:
            return {'message': 'hour is not five or fifteen'}

        cur.execute(insert_target_query, (hour, target))

        config.commit()
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'Database error occurred'}

    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}

    finally:
        if cur:
            cur.close()
        if config:
            config.close()

def salinity():
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

        cur.execute("SELECT * FROM target ORDER BY id DESC LIMIT 2")
        
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