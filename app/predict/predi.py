import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim

dataset2022 = pd.read_csv('data2022.csv', encoding='MacRoman')

dataset2023 = pd.read_csv('data2023.csv', encoding='MacRoman')

# Dataset [Date, AveTemp, MaxTemp, MinTemp, AveWindow, MaxWindow, AveHumidity]

dataset2022['Date'] = pd.to_datetime(dataset2022['Date'])

dataset2023['Date'] = pd.to_datetime(dataset2023['Date'])

# dataset2022.columns -- Index(['Date', 'AveTemp', 'MaxTemp', 'MinTemp', 'AveWindow', 'MaxWindow', 'AveHumidity'],dtype='object')
# dataset2023.columns -- Index(['Date', 'AveTemp', 'MaxTemp', 'MinTemp', 'AveWindow', 'MaxWindow', 'AveHumidity'],dtype='object')

#標準化前処理
d_2022 = dataset2022['Date'].values # datetime64[ns]
d_2023 = dataset2023['Date'].values
l_2022 = dataset2022['AveHumidity'].values # int64
l_2023 = dataset2023['AveHumidity'].values
ave_tmp_2022 = dataset2022['AveTemp'].values # float64
ave_tmp_2023 = dataset2023['AveTemp'].values
max_tmp_2022 = dataset2022['MaxTemp'].values
max_tmp_2023 = dataset2023['MaxTemp'].values
min_tmp_2022 = dataset2022['MinTemp'].values
min_tmp_2023 = dataset2023['MinTemp'].values
ave_window_2022 = dataset2022['AveWindow'].values
ave_window_2023 = dataset2023['AveWindow'].values
max_window_2022 = dataset2022['MaxWindow'].values
max_window_2023 = dataset2023['MaxWindow'].values
# ----shape of d_2022: (366,)---- #
# ----shape of d_2023: (367,)---- #
# ----shape of l_2022: (366,)---- #
# ----shape of l_2022: (367,)---- #

months_2022 = d_2022.astype('datetime64[M]').astype(int) % 12 + 1  # 'M'で月を取得、1を足して1月を1とする
days_2022 = (d_2022 - d_2022.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1

months_2023 = d_2023.astype('datetime64[M]').astype(int) % 12 + 1  # 'M'で月を取得、1を足して1月を1とする
days_2023 = (d_2023 - d_2023.astype('datetime64[M]')).astype('timedelta64[D]').astype(int) + 1

#reshape
numeric_months_2022 = months_2022.reshape(-1, 1)
numeric_months_2023 = months_2023.reshape(-1, 1)
numeric_days_2022 = days_2022.reshape(-1, 1)
numeric_days_2023 = days_2023.reshape(-1, 1)
numeric_l_2022 = l_2022.reshape(-1, 1) # int64
numeric_l_2023 = l_2023.reshape(-1, 1)
numeric_ave_tmp_2022 = ave_tmp_2022.reshape(-1, 1) # float64
numeric_ave_tmp_2023 = ave_tmp_2023.reshape(-1, 1)
numeric_max_tmp_2022 = max_tmp_2022.reshape(-1, 1)
numeric_max_tmp_2023 = max_tmp_2023.reshape(-1, 1)
numeric_min_tmp_2022 = min_tmp_2022.reshape(-1, 1)
numeric_min_tmp_2023 = min_tmp_2023.reshape(-1, 1)
numeric_ave_window_2022 = ave_window_2022.reshape(-1, 1)
numeric_ave_window_2023 = ave_window_2023.reshape(-1, 1)
numeric_max_window_2022 = max_window_2022.reshape(-1, 1)
numeric_max_window_2023 = max_window_2023.reshape(-1, 1)
# ----shape of numeric_ave_tmp_2022: (366, 1)---- #
# ----type of numeric_ave_tmp_2022: float64---- #
# ----shape of numeric_l_2022: (366, 1)---- #
# ----type of numeric_l_2022: int64---- #

# すべての特徴量を同じスケーラーで標準化
scaler_l = StandardScaler()
numeric_l_2022_std = scaler_l.fit_transform(numeric_l_2022)
numeric_l_2023_std = scaler_l.transform(numeric_l_2023)

scaler_ave_tmp = StandardScaler()
numeric_ave_tmp_2022_std = scaler_ave_tmp.fit_transform(numeric_ave_tmp_2022)
numeric_ave_tmp_2023_std = scaler_ave_tmp.transform(numeric_ave_tmp_2023)

scaler_max_tmp = StandardScaler()
numeric_max_tmp_2022_std = scaler_max_tmp.fit_transform(numeric_max_tmp_2022)
numeric_max_tmp_2023_std = scaler_max_tmp.transform(numeric_max_tmp_2023)

scaler_min_tmp = StandardScaler()
numeric_min_tmp_2022_std = scaler_min_tmp.fit_transform(numeric_min_tmp_2022)
numeric_min_tmp_2023_std = scaler_min_tmp.transform(numeric_min_tmp_2023)

scaler_ave_window = StandardScaler()
numeric_ave_window_2022_std = scaler_ave_window.fit_transform(numeric_ave_window_2022)
numeric_ave_window_2023_std = scaler_ave_window.transform(numeric_ave_window_2023)

scaler_max_window = StandardScaler()
numeric_max_window_2022_std = scaler_max_window.fit_transform(numeric_max_window_2022)
numeric_max_window_2023_std = scaler_max_window.transform(numeric_max_window_2023)


months2022 = []
months2023 = []
days2022 = []
days2023 = []
label2022 = []
label2023 = []
AveTemp2022 = []
AveTemp2023 = []
MaxTemp2022 = []
MaxTemp2023 = []
MinTemp2022 = []
MinTemp2023 = []
AveWindow2022 = []
AveWindow2023 = []
MaxWindow2022 = []
MaxWindow2023 = []

for i in range(len(d_2022)):
    months2022.append(numeric_months_2022[i])
    months2023.append(numeric_months_2023[i])
    days2022.append(numeric_days_2022[i])
    days2023.append(numeric_days_2023[i])
    label2022.append(numeric_l_2022_std[i])
    label2023.append(numeric_l_2023_std[i])
    AveTemp2022.append(numeric_ave_tmp_2022_std[i])
    AveTemp2023.append(numeric_ave_tmp_2023_std[i])
    MaxTemp2022.append(numeric_max_tmp_2022_std[i])
    MaxTemp2023.append(numeric_max_tmp_2022_std[i])
    MinTemp2022.append(numeric_min_tmp_2022_std[i])
    MinTemp2023.append(numeric_min_tmp_2023_std[i])
    AveWindow2022.append(numeric_ave_window_2022_std[i])
    AveWindow2023.append(numeric_ave_window_2023_std[i])
    MaxWindow2022.append(numeric_max_window_2022_std[i])
    MaxWindow2023.append(numeric_max_window_2023_std[i])

months2022 = np.array(months2022)
months2023 = np.array(months2023)
days2022 = np.array(days2022)
days2023 = np.array(days2023)
label2022 = np.array(label2022) # int64
label2023 = np.array(label2023)
AveTemp2022 = np.array(AveTemp2022) # faloat64
AveTemp2023 = np.array(AveTemp2023)
MaxTemp2022 = np.array(MaxTemp2022)
MaxTemp2023 = np.array(MaxTemp2023)
MinTemp2022 = np.array(MinTemp2022)
MinTemp2023 = np.array(MinTemp2023)
AveWindow2022 = np.array(AveWindow2022)
AveWindow2023 = np.array(AveWindow2023)
MaxWindow2022 = np.array(MaxWindow2022)
MaxWindow2023 = np.array(MaxWindow2023)
# ----shape of dates2022: (366, 1)---- #
# ----shape of label2022: (366, 1)---- #
# ----shape of AveTemp2022: (366, 1)---- #

# 訓練データとテストデータのサイズを決定
test_len = len(months2023)
train_len = len(months2022)

# 2022が学習、2023がテスト
data_2022 = np.column_stack([months2022, days2022, AveTemp2022, MaxTemp2022, MinTemp2022, AveWindow2022, MaxWindow2022])
data_2023 = np.column_stack([months2023, days2023, AveTemp2023, MaxTemp2023, MinTemp2023, AveWindow2023, MaxWindow2023])
# ----data2022: size (366, 7)---- #
# ----data2023: size (366, 7)---- #
# ----label2022: size (366, 1)---- #
# ----label2023: size (366, 1)---- #

# 一応dataとlabelのデータの作り方に違いがある。データ自体は一緒だと思われる
train_x = torch.Tensor(data_2022)
test_x = torch.Tensor(data_2023)

train_y = torch.Tensor(label2022)
test_y = torch.Tensor(label2023)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_batch = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0)

test_batch = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0)

####################################################

D_in = 7
H = 200
D_out = 1
epoch = 300

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
# デバイスの確認
print("Device: {}".format(device))

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())

train_loss_list = [] #学習損失
test_loss_list = [] #評価損失

for i in range(epoch):
    print('--------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    train_loss = 0
    test_loss = 0

    # ------------------学習パート------------------ #
    net.train()

    for data, label in train_batch:
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        y_pred = net(data)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    batch_train_loss = train_loss / len(train_batch)

    # ---------評価パート--------- #
    net.eval()
    with torch.no_grad():
        for data, label in test_batch:
            data = data.to(device)
            label = label.to(device)
            y_pred = net(data)
            loss = criterion(y_pred, label)
            test_loss += loss.item()

    batch_test_loss = test_loss / len(test_batch)

    print("Train_Loss: {:.2E} Test_Loss: {:.2E}".format(batch_train_loss, batch_test_loss))
    train_loss_list.append(batch_train_loss)
    test_loss_list.append(batch_test_loss)


#-----------------------------------------------------------
# # 損失の可視化
# plt.figure()
# plt.title('Train and Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(range(1, epoch+1), train_loss_list, color='blue',
#          linestyle='-', label='Train_Loss')
# plt.plot(range(1, epoch+1), test_loss_list, color='red',
#          linestyle='--', label='Test_Loss')
# plt.legend()  # 凡例
# plt.show()  # 表示


# テストデータに対する予測の評価
net.eval()
with torch.no_grad():
    pred_ma = []
    true_ma = []
    for data, label in test_batch:
        data = data.to(device)
        label = label.to(device)
        y_pred = net(data)
        pred_ma.append(y_pred.view(-1).tolist())
        true_ma.append(label.view(-1).tolist())

pred_ma = [elem for lst in pred_ma for elem in lst]  # listを1次元配列に
true_ma = [elem for lst in true_ma for elem in lst]

pred_ma = np.array(pred_ma)
pred_ma = scaler_l.inverse_transform(pred_ma.reshape(-1, 1))
# 標準化を解除して元の湿度に変換
true_ma = np.array(true_ma)
true_ma = scaler_l.inverse_transform(true_ma.reshape(-1, 1))

# 平均絶対誤差を計算
mae = mean_absolute_error(true_ma, pred_ma)
print("MAE: {:.3f}".format(mae))


#-----------------------------------------------------------
# date = d_2022.reshape(-1, 1)  # 2022年の日付

# plt.figure()
# plt.title('pred view')
# plt.xlabel('Date')
# plt.ylabel('Humidity')
# plt.plot(date, true_ma, color='dodgerblue',
#          linestyle='-', label='true')
# plt.plot(date, pred_ma, color='red',
#          linestyle='--', label='pred')
# plt.legend()  # 凡例
# plt.xticks(rotation=30)  # x軸ラベルを30度回転して表示
# plt.show()