import mysql.connector
from mysql.connector import errorcode

def features():
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

        create_features_query = """
        CREATE TABLE IF NOT EXISTS features(
            id INT PRIMARY KEY AUTO_INCREMENT,
            max_temp FLOAT,
            min_temp FLOAT,
            ave_temp FLOAT,
            ave_humidity INT,
            ave_windvelocity FLOAT,
            max_windvelocity FLOAT
        );
        """
        cursor.execute(create_features_query)
        
        #  -- オフセット (1~7 for day, 1~6 for hour)
        create_target_query = """
        CREATE TABLE IF NOT EXISTS target(
            id INT PRIMARY KEY AUTO_INCREMENT,
            data_id INT,
            future_type ENUM('day', 'hour'),
            future_offset INT,
            future_value FLOAT,
            FOREIGN KEY (data_id) REFERENCES data(id)
        );
        """ 
        cursor.execute(create_target_query)

        insert_features_query = """
        INSERT INTO features
            (id, max_temp, min_temp, ave_temp, ave_humidity, ave_windvelocity, max_windvelocity)
        VALUES
            (1, 25.1, 22.0, 23.2, 80, 2.4, 2.6);"""

        cursor.execute(insert_features_query)
        cnx.commit()

        insert_target_query = """
        INSERT INTO target
        (data_id, future_type, future_offset, future_value)
        VALUES
        (1, 'day', 1, 23.45);"""
        
        cursor.execute(insert_target_query)
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