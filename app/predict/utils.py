import mysql.connector
from mysql.connector import Error, IntegrityError
import os
import numpy as np
import joblib
import torch
import pandas as pd

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

    if latest_features is None:
        raise ValueError("No features found in the database.")
    cur.close()
    config.close()

    precipitation = latest_features[5]
    tempMax = latest_features[6]
    tempMin = latest_features[7]

    precipitation = np.array(precipitation).reshape(-1, 1)
    tempMax = np.array(tempMax).reshape(-1, 1)
    tempMin = np.array(tempMin).reshape(-1, 1)

    save_directory = "./app/predict/models"
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    scaler_tempMax = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    scaler_tempMin = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    scaler_precipitation = joblib.load(save_path)

    tempMax = scaler_tempMax.transform(tempMax)
    tempMin = scaler_tempMin.transform(tempMin)
    precipitation = scaler_precipitation.transform(precipitation)

    months = np.array(latest_features[2])
    days = np.array(latest_features[3])
    hour = np.array(latest_features[4])
    tempMax = np.array(tempMax)
    tempMin = np.array(tempMin)
    precipitation = np.array(precipitation)

    data = np.column_stack([months, days, hour, tempMax, tempMin, precipitation])
    forecast = data.reshape(-1, D_in)

    # latest_features = np.expand_dims(latest_features, axis=0)

    # data = np.append(data, latest_features, axis=0)

    # forecast = torch.tensor(data, dtype=torch.float32)

    return forecast


def connectDaysPredict():
    D_in = 5

    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)

    cur = config.cursor()

    cur.execute("SELECT * FROM featuresDays ORDER BY id DESC LIMIT 6")
    cur.statement
    latest_features = cur.fetchall()

    if latest_features is None:
        raise ValueError("No features found in the database.")
    cur.close()
    config.close()

    precipitation = []
    tempMax = []
    tempMin = []
    months = []
    days = []

    for i in range(len(latest_features)):
        months.append(latest_features[i][2])
        days.append(latest_features[i][3])
        precipitation.append(latest_features[i][4])
        tempMax.append(latest_features[i][5])
        tempMin.append(latest_features[i][6])
    
    months = np.array(months).reshape(1, -1).flatten()
    days = np.array(days).reshape(1, -1).flatten()
    precipitation = np.array(precipitation).reshape(1, -1).flatten()
    tempMax = np.array(tempMax).reshape(1, -1).flatten()
    tempMin = np.array(tempMin).reshape(1, -1).flatten()

    save_directory = "./app/predict/models/days"
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    scaler_tempMax = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    scaler_tempMin = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    scaler_precipitation = joblib.load(save_path)

    for i in range(len(latest_features)):
        tempMax[i] = scaler_tempMax.transform(tempMax[i].reshape(-1, 1))
        tempMin[i] = scaler_tempMin.transform(tempMin[i].reshape(-1, 1))
        precipitation[i] = scaler_precipitation.transform(precipitation[i].reshape(-1, 1))
    
    # for i in range(len(latest_features)):
    #     months = np.array(latest_features[i][2])
    #     days = np.array(latest_features[i][3])
    #     tempMax[i] = np.array(tempMax[i])
    #     tempMin[i] = np.array(tempMin[i])
    #     precipitation[i] = np.array(precipitation[i])


    data = []
    for i in range(len(latest_features)):
        data.append(np.column_stack([months[i], days[i], tempMax[i], tempMin[i], precipitation[i]]))
    forecast = data.reshape(-1, D_in)
    # latest_features = np.expand_dims(latest_features, axis=0)

    # data = np.append(data, latest_features, axis=0)

    # forecast = torch.tensor(data, dtype=torch.float32)
    print(forecast.shape)
    return forecast

def connectDays():
    D_in = 5

    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)

    cur = config.cursor()

    cur.execute("SELECT * FROM featuresDays ORDER BY id DESC LIMIT 1")
    cur.statement
    latest_features = cur.fetchone()

    if latest_features is None:
        raise ValueError("No features found in the database.")
    cur.close()
    config.close()

    precipitation = latest_features[4]
    tempMax = latest_features[5]
    tempMin = latest_features[6]

    precipitation = np.array(precipitation).reshape(-1, 1)
    tempMax = np.array(tempMax).reshape(-1, 1)
    tempMin = np.array(tempMin).reshape(-1, 1)

    save_directory = "./app/predict/models/days"
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    scaler_tempMax = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    scaler_tempMin = joblib.load(save_path)

    save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    scaler_precipitation = joblib.load(save_path)

    tempMax = scaler_tempMax.transform(tempMax)
    tempMin = scaler_tempMin.transform(tempMin)
    precipitation = scaler_precipitation.transform(precipitation)

    months = np.array(latest_features[2])
    days = np.array(latest_features[3])
    tempMax = np.array(tempMax)
    tempMin = np.array(tempMin)
    precipitation = np.array(precipitation)

    data = np.column_stack([months, days, tempMax, tempMin, precipitation])
    forecast = data.reshape(-1, D_in)
    # latest_features = np.expand_dims(latest_features, axis=0)

    # data = np.append(data, latest_features, axis=0)

    # forecast = torch.tensor(data, dtype=torch.float32)
    
    return forecast

def salinity():
    config = mysql.connector.connect(
        host='mysql-container',
        port='3306',
        user='root',
        password='pass',
        database='db'
    )

    config.ping(reconnect=True)

    cur = config.cursor()

    cur.execute("SELECT salinity FROM data ORDER BY id DESC LIMIT 1")
    cur.statement
    salinity_data = cur.fetchone()

    if salinity_data is None:
        raise ValueError("No features found in the database.")
    cur.close()
    config.close()

    return salinity_data

def categorize(salinity_data):
    Dataset = pd.read_csv('app/predict/sensor_data.csv')
    data = Dataset['field1'].values.reshape(-1, 1)
    data = np.array(data)

    mean = np.mean(data)
    std_dev = np.std(data)

    # 標準偏差に基づいて段階分け
    # 1: 平均 - 2σ 未満
    # 2: 平均 - 1σ 以上 平均 - 2σ 未満
    # 3: 平均 ± 1σ の範囲
    # 4: 平均 + 1σ 以上 平均 + 2σ 未満
    # 5: 平均 + 2σ 以上

    if salinity_data < mean - 2 * std_dev:
        salinity_data = 1
    elif mean - 2 * std_dev <= salinity_data < mean - 1 * std_dev:
        salinity_data = 2
    elif mean - 1 * std_dev <= salinity_data < mean + 1 * std_dev:
        salinity_data = 3
    elif mean + 1 * std_dev <= salinity_data < mean + 2 * std_dev:
        salinity_data = 4
    else:
        salinity_data = 5

    return salinity_data