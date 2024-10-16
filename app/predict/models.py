import mysql.connector
from mysql.connector import errorcode

def features():
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

        create_features_query = """
        CREATE TABLE IF NOT EXISTS features(
            id INT PRIMARY KEY AUTO_INCREMENT,
            year INT,
            month INT,
            day INT,
            hour INT,
            precipitation INT,
            tempMax FLOAT,
            tempMin FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_features_query)

        create_featuresDays_query = """
        CREATE TABLE IF NOT EXISTS featuresDays(
            id INT PRIMARY KEY AUTO_INCREMENT,
            year INT,
            month INT,
            day INT,
            precipitation INT,
            tempMax FLOAT,
            tempMin FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_featuresDays_query)

        create_target_query = """
        CREATE TABLE IF NOT EXISTS target(
            id INT PRIMARY KEY AUTO_INCREMENT,
            period_unit ENUM('hour', 'day') NOT NULL,
            future_offset INT,
            future_value INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        );
        """ 
        cursor.execute(create_target_query)

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