import mysql.connector
from mysql.connector import Error, IntegrityError
import torch
import numpy as np
from .utils import categorize

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

def insert_featuresDays(forecast):
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
        INSERT INTO featuresDays
        (year, month, day, precipitation, tempMax, tempMin)
        VALUES
        (%s, %s, %s, %s, %s, %s);"""

        for i in range(len(forecast)):
           cur.execute(insert_features_query, (forecast[i]['year'], forecast[i]['month'], forecast[i]['day'], forecast[i]['precipitation'], forecast[i]['tempMax'], forecast[i]['tempMin']))

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

def saveTarget(hour, target):
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
        (period_unit, future_offset, future_value)
        VALUES
        ('hour', %s, %s);"""

        if hour == 5.0:
            hour = 17
        elif hour == 17.0:
            hour = 5
        elif hour == 11.0:
            hour = 17
        elif hour == 22.0:
            hour = 5
        else:
            return {'message': 'hour is not five or fifteen'}
        
        category = categorize(target)

        cur.execute(insert_target_query, (hour, category))

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

def saveTargetDays(day, target):
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
        (period_unit, future_offset, future_value)
        VALUES
        ('day', %s, %s);"""

        category = categorize(target)

        cur.execute(insert_target_query, (day, category))

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

        cur.execute("SELECT * FROM target WHERE period_unit = 'hour' ORDER BY id DESC LIMIT 2")
        
        cur.statement
        result = cur.fetchall()
        
        if result is None:
            return ({'message': 'salinity not found'}), 401

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

def days_salinity():
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

        cur.execute("SELECT * FROM target WHERE period_unit = 'day' ORDER BY id DESC LIMIT 6;")
        
        cur.statement
        result = cur.fetchall()
        
        if result is None:
            return ({'message': 'salinity not found'}), 401

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