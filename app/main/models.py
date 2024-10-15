import mysql.connector
from mysql.connector import errorcode

def data():
    try:
        cnx = mysql.connector.connect(
                host='mysql-container',
                port='3306',
                user='root',
                password='pass',
                database='db'
        )

        cursor = cnx.cursor()

        cursor.execute("USE db;")

        create_data_query = """
        CREATE TABLE IF NOT EXISTS data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            device_id INT,
            location VARCHAR(255),
            in_tank INT,
            out_tank INT,
            salinity FLOAT
        );
        """
        
        cursor.execute(create_data_query)

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("ユーザー名またはパスワードが間違っています。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("データベースが見つかりません。")
        else:
            print(err)
            
    finally:
        if cursor:
            cursor.close()
        if cnx:
            cnx.close()