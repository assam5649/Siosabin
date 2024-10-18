import pandas as pd
import numpy as np

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
        category = 1
    elif mean - 2 * std_dev <= salinity_data < mean - 1 * std_dev:
        category = 2
    elif mean - 1 * std_dev <= salinity_data < mean + 1 * std_dev:
        category = 3
    elif mean + 1 * std_dev <= salinity_data < mean + 2 * std_dev:
        category = 4
    else:
        category = 5
        
    return category