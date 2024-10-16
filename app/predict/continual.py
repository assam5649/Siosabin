import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import datetime
import os
import joblib
from .utils import connect, salinity

def updateModel(forecast, salinity_data):
    D_in = 6
    H = 200
    D_out = 1
    epoch = 100

    load_directory = "./app/predict"
    
    load_path = os.path.join(load_directory, 'sensor_data.csv')
    Dataset = pd.read_csv(load_path)
    # Weather = pd.read_csv('NAHAdata.csv', encoding='ISO-8859-1')
    load_path = os.path.join(load_directory, 'NAHAdata.csv')
    df = pd.read_csv(load_path, parse_dates=['Date'])

    #Dataset Index(['created_at', 'entry_id', 'field1', 'latitude', 'longitude','elevation', 'status'],dtype='object')

    Dataset['created_at'] = pd.to_datetime(Dataset['created_at'])

    sensor_date = Dataset['created_at'].values

    months = sensor_date.astype('datetime64[M]').astype(int) % 12 + 1
    days = (sensor_date - sensor_date.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1

    daytime = Dataset['created_at'].values.reshape(-1, 1)
    label = Dataset['field1'].values.reshape(-1, 1)
    # 5 hour and 17 hour select
    UseWeather = df[(df['Date'].dt.time == pd.to_datetime('05:00:00').time()) | (df['Date'].dt.time == pd.to_datetime('17:00:00').time())]
    UseWeather = UseWeather.copy()
    UseWeather.loc[:, 'hour'] = UseWeather['Date'].dt.hour  # .locを使用

    # hourを変数に格納
    hours = UseWeather['hour'].values  # hourの値を変数に格納

    tempMax = UseWeather['Max'].values.reshape(-1, 1)
    tempMin = UseWeather['Min'].values.reshape(-1, 1)
    precipitation = UseWeather['Precipitation'].values.reshape(-1, 1)

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

    label = np.array(label)
    months = np.array(months)
    days = np.array(days)
    hours = np.array(hours)
    tempMax = np.array(tempMax)
    tempMin = np.array(tempMin)
    precipitation = np.array(precipitation)
    salinity_data = np.array(salinity_data)

    data = np.column_stack([months, days, hours, tempMax, tempMin, precipitation])
    data = data.reshape(-1, D_in)
    salinity_data = np.expand_dims(salinity_data, axis=0)

    data = np.append(data, forecast, axis=0)
    label = np.append(label, salinity_data, axis=0)

    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    dataset = TensorDataset(data, label)

    class Net(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(Net, self).__init__()
            self.lstm = nn.LSTM(D_in, H, batch_first=True, num_layers=1)
            self.linear = nn.Linear(H, D_out)
        
        def forward(self, x):
            output, (hidden, cell) = self.lstm(x)
            output = self.linear(output)
            return output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(D_in, H, D_out).to(device)

    print("Device: {}".format(device))

    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.02)

    train_loss_list = []
    test_loss_list = []

    loo = LeaveOneOut()

    save_directory = 'app/predict/models'
    os.makedirs(save_directory, exist_ok=True)

    for i in range(epoch):
        print('--------------------------------')
        print("Epoch: {}/{}".format(i+1, epoch))

        train_loss = 0
        test_loss = 0

        net.train()

        for train_index, test_index in loo.split(dataset):
            train_data = [dataset[j] for j in train_index]
            test_data = [dataset[j] for j in test_index]

            train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
            test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

            for data, label in train_loader:
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()
                data = data.float()
                y_pred = net(data).float()
                loss = criterion(y_pred, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        batch_train_loss = train_loss / len(train_data)

        net.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data = data.to(device)
                label = label.to(device)
                data = data.float()
                y_pred = net(data).float()
                loss = criterion(y_pred, label)
                test_loss += loss.item()

        batch_test_loss = test_loss / len(test_data)

        print("Train_Loss: {:.2E} Test_Loss: {:.2E}".format(batch_train_loss, batch_test_loss))
        train_loss_list.append(batch_train_loss)
        test_loss_list.append(batch_test_loss)

    avg_loss = sum(test_loss_list) / len(test_loss_list)
    print(avg_loss)
    target_avg_loss = 0.8  # 最低目標平均損失
    avg_loss = sum(test_loss_list) / len(test_loss_list)
    # if avg_loss < target_avg_loss:
    #     save_path = os.path.join(save_directory, 'model_after_LOO_CV.pth')
    #     torch.save(net.state_dict(), save_path)

    #     save_path = os.path.join(save_directory, 'scaler_label.joblib')
    #     joblib.dump(scaler_label, save_path)

    #     save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    #     joblib.dump(scaler_tempMax, save_path)

    #     save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    #     joblib.dump(scaler_tempMin, save_path)

    #     save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    #     joblib.dump(scaler_precipitation, save_path)

    #     print(f'Model saved at epoch {epoch} with average loss: {avg_loss}')
    # else:
    #     if count < 10:
    #         return 1
    #     update_model(connect(), salinity())
