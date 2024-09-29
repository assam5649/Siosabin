import mysql.connector
from mysql.connector import errorcode

def data():
    try:
        # データベース接続の設定
        cnx = mysql.connector.connect(
                host='mysql-container',
                port='3306',
                user='root',
                password='pass',
                database='db'
        )

        # データベースに接続
        cursor = cnx.cursor()

        cursor.execute("USE db;")

        create_data_query = """
        CREATE TABLE IF NOT EXISTS data (
            id INT PRIMARY KEY AUTO_INCREMENT,
            max_temp FLOAT,
            min_temp FLOAT,
            ave_temp FLOAT,
            ave_humidity INT,
            ave_windvelocity FLOAT,
            max_windvelocity FLOAT
        )
        """
        cursor.execute(create_data_query)

        
        insert_data_query = """
        INSERT INTO data
        (max_temp, min_temp, ave_temp, ave_humidity, ave_windvelocity, max_windvelocity)
        VALUES
        (25.1, 22.0, 23.2, 80, 2.4, 2.6);"""

        cursor.execute(insert_data_query)
        cnx.commit()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("ユーザー名またはパスワードが間違っています。")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("データベースが見つかりません。")
        else:
            print(err)
    finally:
        # 接続を閉じる
        cursor.close()
        cnx.close()