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
            device_id INT,
            location VARCHAR(255),
            in_tank INT,
            out_tank INT,
            salinity INT
        );
        """
        
        cursor.execute(create_data_query)

        
        insert_data_query = """
        INSERT INTO data
        (device_id, location, in_tank, out_tank, salinity)
        VALUES
        (11, 'POINT(137.10 35.20)', 70, 30, 20);"""

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