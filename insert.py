import mysql.connector
from mysql.connector import Error, IntegrityError

def initialize_service():
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

        # # initialize users
        # initialize_query1 = """
        # INSERT INTO users 
        # (name, password)
        # VALUES 
        # ('a', 'aPass');"""

        # initialize_query2 = """
        # INSERT INTO users 
        # (name, password)
        # VALUES 
        # ('b', 'bPass');"""

        # initialize_query3 = """
        # INSERT INTO users 
        # (name, password)
        # VALUES 
        # ('c', 'cPass');"""

        # initialize_query4 = """
        # INSERT INTO users 
        # (name, password)
        # VALUES 
        # ('d', 'dPass');"""

        # initialize_query5 = """
        # INSERT INTO users 
        # (name, password)
        # VALUES 
        # ('e', 'ePass');"""

        # cur.execute(initialize_query1)
        # cur.execute(initialize_query2)
        # cur.execute(initialize_query3)
        # cur.execute(initialize_query4)
        # cur.execute(initialize_query5)
        # config.commit()

        # initialize data
        initialize_query1 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (1, 'POINT(10.00, 10.00)', 100, 0, 3);"""

        initialize_query2 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (2, 'POINT(20.00, 20.00)', 90, 10, 4);"""

        initialize_query3 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (3, 'POINT(30.00, 30.00)', 80, 20, 5);"""

        initialize_query4 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (4, 'POINT(40.00, 40.00)', 70, 30, 2);"""

        initialize_query5 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (5, 'POINT(50.00, 50.00)', 60, 40, 1);"""

        initialize_query6 = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (6, 'POINT(60.00, 60.00)', 50, 50, 3);"""

        cur.execute(initialize_query1)
        cur.execute(initialize_query2)
        cur.execute(initialize_query3)
        cur.execute(initialize_query4)
        cur.execute(initialize_query5)
        cur.execute(initialize_query6)
        config.commit()

        # # initialize features
        # initialize_query1 = """
        # INSERT INTO featuresDays
        # (year, month, day, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 22, 1, 30, 25);"""

        # initialize_query2 = """
        # INSERT INTO featuresDays
        # (year, month, day, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 22, 1, 31, 26);"""

        # initialize_query3 = """
        # INSERT INTO featuresDays
        # (year, month, day, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 1, 32, 27);"""

        # initialize_query4 = """
        # INSERT INTO featuresDays
        # (year, month, day, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 1, 29, 24);"""

        # initialize_query5 = """
        # INSERT INTO featuresDays
        # (year, month, day, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 24, 1, 28, 23);"""

        # cur.execute(initialize_query1)
        # cur.execute(initialize_query2)
        # cur.execute(initialize_query3)
        # cur.execute(initialize_query4)
        # cur.execute(initialize_query5)
        # config.commit()

        # # initialize featuresDays
        # initialize_query1 = """
        # INSERT INTO features
        # (year, month, day, hour, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 22, 5, 1, 30, 25);"""

        # initialize_query2 = """
        # INSERT INTO features
        # (year, month, day, hour, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 5, 1, 30, 25);"""

        # initialize_query3 = """
        # INSERT INTO features
        # (year, month, day, hour, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 5, 1, 30, 25);"""

        # initialize_query4 = """
        # INSERT INTO features
        # (year, month, day, hour, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 5, 1, 30, 25);"""

        # initialize_query5 = """
        # INSERT INTO features
        # (year, month, day, hour, precipitation, tempMax, tempMin)
        # VALUES
        # (2024, 8, 23, 5, 1, 30, 25);"""

        # cur.execute(initialize_query1)
        # cur.execute(initialize_query2)
        # cur.execute(initialize_query3)
        # cur.execute(initialize_query4)
        # cur.execute(initialize_query5)
        # config.commit()

        initialize_query1 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('hour', 5, 3);"""

        initialize_query2 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('hour', 17, 4)"""

        initialize_query3 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('hour', 5, 5)"""

        initialize_query4 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('hour', 17, 1)"""

        initialize_query5 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 1, 2)"""

        initialize_query6 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 2, 3)"""

        initialize_query7 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 3, 4)"""

        initialize_query8 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 4, 5)"""

        initialize_query9 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 5, 1)"""

        initialize_query9 = """
        INSERT INTO target
        (period_unit, future_offset, future_value)
        VALUES
        ('day', 6, 2)"""

        cur.execute(initialize_query1)
        cur.execute(initialize_query2)
        cur.execute(initialize_query3)
        cur.execute(initialize_query4)
        cur.execute(initialize_query5)
        cur.execute(initialize_query6)
        cur.execute(initialize_query7)
        cur.execute(initialize_query8)
        cur.execute(initialize_query9)
        config.commit()
        
    
    except IntegrityError as e:
        print(f"Integrity error occurred: {e}")
        return {'message': 'Database error occured'}
    
    except Error as e:
        print(f"Error: {e}")
        return {'message': 'Database error occurred', 'error': str(e)}
        
    finally:
        if cur:
            cur.close()
        if config:
            config.close()
initialize_service()