import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

D_in = 6
H = 200
D_out = 1
epoch = 100

Dataset = pd.read_csv('sensor_data.csv')
# Weather = pd.read_csv('NAHAdata.csv', encoding='ISO-8859-1')
df = pd.read_csv('NAHAdata.csv', parse_dates=['Date'])

#Dataset Index(['created_at', 'entry_id', 'field1', 'latitude', 'longitude','elevation', 'status'],dtype='object')

Dataset['created_at'] = pd.to_datetime(Dataset['created_at'])

sensor_date = Dataset['created_at'].values

months = sensor_date.astype('datetime64[M]').astype(int) % 12 + 1
days = (sensor_date - sensor_date.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1

daytime = Dataset['created_at'].values.reshape(-1, 1)
label = Dataset['field1'].values.reshape(-1, 1)
# 5 hour and 17 hour select
UseWeather = df[(df['Date'].dt.time == pd.to_datetime('05:00:00').time()) | (df['Date'].dt.time == pd.to_datetime('17:00:00').time())]
UseWeather.loc[:, 'hour'] = UseWeather['Date'].dt.hour  # .locを使用

# hourを変数に格納
hour = UseWeather['hour'].values  # hourの値を変数に格納

tempMax = UseWeather['Max'].values.reshape(-1, 1)
tempMin = UseWeather['Min'].values.reshape(-1, 1)
precipitation = UseWeather['Precipitation'].values.reshape(-1, 1)

scaler_label = StandardScaler()
scaler_tempMax = StandardScaler()
scaler_tempMin = StandardScaler()
scaler_precipitation = StandardScaler()

label = scaler_label.fit_transform(label)
tempMax = scaler_tempMax.fit_transform(tempMax)
tempMin = scaler_tempMin.fit_transform(tempMin)
precipitation = scaler_precipitation.fit_transform(precipitation)

label = np.array(label)
months = np.array(months)
days = np.array(days)
tempMax = np.array(tempMax)
tempMin = np.array(tempMin)
precipitation = np.array(precipitation)

data = np.column_stack([months, days, hour, tempMax, tempMin, precipitation])
data = data.reshape(-1, D_in)

data = torch.tensor(data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.float32)
dataset = TensorDataset(data, label)

####################################################

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

optimizer = optim.Adam(net.parameters(), lr=0.05)

train_loss_list = []
test_loss_list = []

loo = LeaveOneOut()

save_directory = 'models'
os.makedirs(save_directory, exist_ok=True)

for i in range(epoch):
    print('--------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    train_loss = 0
    test_loss = 0

    # ------------------学習パート------------------ #
    net.train()

    for train_index, test_index in loo.split(dataset):
        train_data = [dataset[j] for j in train_index]
        test_data = [dataset[j] for j in test_index]
        
        # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        # y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

    # ---------評価パート--------- #
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


target_avg_loss = 0.65  # 目標平均損失
avg_loss = sum(test_loss_list) / len(test_loss_list)
if avg_loss < target_avg_loss:
    save_path = os.path.join(save_directory, 'model_after_LOO_CV.pth')
    torch.save(net.state_dict(), save_path)

    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    joblib.dump(scaler_label, save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMax.joblib')
    joblib.dump(scaler_tempMax, save_path)

    save_path = os.path.join(save_directory, 'scaler_tempMin.joblib')
    joblib.dump(scaler_tempMin, save_path)

    save_path = os.path.join(save_directory, 'scaler_precipitation.joblib')
    joblib.dump(scaler_precipitation, save_path)

    print(f'Model saved at epoch {epoch} with average loss: {avg_loss}')
else:
    print(f'avg: {avg_loss}')

plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()
plt.show()

net.eval()
with torch.no_grad():
    pred_ma = []
    true_ma = []
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        data = data.float()
        y_pred = net(data).float()
        pred_ma.append(y_pred.view(-1).tolist())
        true_ma.append(label.view(-1).tolist())

pred_ma = [elem for lst in pred_ma for elem in lst]
true_ma = [elem for lst in true_ma for elem in lst]

pred_ma = np.array(pred_ma)
pred_ma = scaler_label.inverse_transform(pred_ma.reshape(-1, 1))
true_ma = np.array(true_ma)
true_ma = scaler_label.inverse_transform(true_ma.reshape(-1, 1))

print(f"真のラベルの形状: {true_ma.shape}")
print(f"予測の形状: {pred_ma.shape}")
# 平均絶対誤差を計算
mae = mean_absolute_error(true_ma, pred_ma)
print("MAE: {:.3f}".format(mae))

print(true_ma)
print(pred_ma)

# date = sensor_date.reshape(-1, 1)

# plt.figure()
# plt.title('pred view')
# plt.xlabel('Date')
# plt.ylabel('label')
# plt.plot(date, true_ma, color='dodgerblue',
#          linestyle='-', label='true')
# plt.plot(date, pred_ma, color='red', 
#          linestyle='--', label='pred')
# plt.legend()
# plt.xticks(rotation=30)
# plt.show()