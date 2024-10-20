import torch
import torch.nn as nn
import mysql.connector
from mysql.connector import Error, IntegrityError
import os
import joblib
import numpy as np
from .services import saveTarget, saveTargetDays

def predictDays(forecast):
    # for i in range(len(forecast)):
    D_in = 5
    H = 160
    D_out = 1

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

    save_directory = "./app/predict/models/days"
    save_path = os.path.join(save_directory, 'model_after_LOO_CV.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(save_path, map_location=device))
    net.eval()
    
    save_path = os.path.join(save_directory, 'scaler_label.joblib')
    scaler_label = joblib.load(save_path)

    input_data = torch.tensor(forecast, dtype=torch.float)

    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(0).unsqueeze(0)
    input_data = input_data.to(device)

    with torch.no_grad():
        prediction = net(input_data)

    prediction = np.array(prediction)



    prediction = scaler_label.inverse_transform(prediction.reshape(-1, 1))
    saveTargetDays(forecast[0][1], prediction.item())