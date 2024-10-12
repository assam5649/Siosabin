import torch
import torch.nn as nn
import mysql.connector
from mysql.connector import Error, IntegrityError
# from . import services

def predict(forecast):
    D_in = 6
    H = 200
    D_out = 1

    #####################################################

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

    load_directory = 'models'
    load_path = os.path.join(load_directory, 'model_after_LOO_CV.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net().to(device)
    net.load_state_dict(torch.load(load_directory, map_location=device))
    net.eval()

    # 予測データの準備（データが必要な形状に変換されていることを確認）
    # 例: data = torch.tensor([[...]], dtype=torch.float32).to(device)
    # 入力データの形状が (batch_size, sequence_length, input_size) であることを確認
    save_directory = "models"
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    forecast = scaler_label.fit_transform(input_data)

    input_data = forecast.to(device)

    # input_dataに対するスケーリングの適用
    with torch.no_grad():
        prediction = net(input_data)

    # predictionに対するアンチスケーリングの適用
    print("Predicted value:", prediction.item())

    prediction = np.array(prediction)
    prediction = scaler_label.inverse_transform(prediction.reshape(-1, 1))

    print("Predicted value:", prediction.item())

    # predictionをdbに保存

    # save_target(predict.item())    