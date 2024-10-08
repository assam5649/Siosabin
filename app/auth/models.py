import mysql.connector
from mysql.connector import errorcode

def user():
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

        create_users_query = """
        CREATE TABLE IF NOT EXISTS users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """

        cursor.execute(create_users_query)

        insert_users_query = """
        INSERT INTO users 
        (name, password)
        VALUES 
        ('a', 'aPass');"""

        cursor.execute(insert_users_query)
        cnx.commit()
        
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