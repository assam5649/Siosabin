import mysql.connector
from mysql.connector import Error, IntegrityError
import os
import numpy as np
import joblib
import torch

def connect():
    D_in = 6

    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)

    cur = config.cursor()

    cur.execute("SELECT * FROM features ORDER BY id DESC LIMIT 1")
    cur.statement
    latest_features = cur.fetchone()

    cur.close()
    config.close()

    precipitation = latest_features[5]
    tempMax = latest_features[6]
    tempMin = latest_features[7]

    precipitation = np.array(precipitation).reshape(-1, 1)
    tempMax = np.array(tempMax).reshape(-1, 1)
    tempMin = np.array(tempMin).reshape(-1, 1)

    save_directory = "models"
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    scaler_tempMax = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    scaler_tempMin = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    scaler_precipitation = joblib.load(save_path)

    tempMax = scaler_tempMax.fit_transform(tempMax)
    tempMin = scaler_tempMin.fit_transform(tempMin)
    precipitation = scaler_precipitation.fit_transform(precipitation)

    months = np.array(latest_features[2])
    days = np.array(latest_features[3])
    hour = np.array(latest_features[4])
    tempMax = np.array(tempMax)
    tempMin = np.array(tempMin)
    precipitation = np.array(precipitation)

    data = np.column_stack([months, days, hour, tempMax, tempMin, precipitation])
    data = data.reshape(-1, D_in)

    # latest_features = np.expand_dims(latest_features, axis=0)

    # data = np.append(data, latest_features, axis=0)

    forecast = torch.tensor(data, dtype=torch.float32)

    return forecast