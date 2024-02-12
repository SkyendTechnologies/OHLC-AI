import matplotlib.pyplot as plt
import pandas as pd
from os import path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# Класс для загрузки данных из CSV-файла
class ForexData(Dataset):
    def __init__(self, file_name='data/fx_data.csv', symbol='USDRUB'):
        if path.exists(file_name):
            df = pd.read_csv(file_name)
            self.data = df.loc[:, [symbol + '_Open', symbol + '_High', symbol + '_Low', symbol + '_Close']].values
        else:
            print('Файл не найден')
            self.data = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# Функция для получения данных OHLC для конкретной даты и символа валюты
def get_candles(file_name='data/fx_data.csv', date='2019.01.01', symbol='USDRUB'):
    if path.exists(file_name):
        df = pd.read_csv(file_name)
        #return df.loc[df['<Date>'] == date, [symbol + '_Open', symbol + '_High', symbol + '_Low', symbol + '_Close']].values
        return_data = df.head(df.shape[-0] - 1).loc[:, [symbol + '_Open', symbol + '_High', symbol + '_Low', symbol + '_Close']].values

        for i in range(return_data.shape[0]):
            return_data[i][0] = return_data[i][0]
            return_data[i][1] = return_data[i][1] 
            return_data[i][2] = return_data[i][2] 
            return_data[i][3] = return_data[i][3] 
            # print(return_data[i][0], return_data[i][1], return_data[i][2], return_data[i][3])

        return return_data
    else:
        print('Файл не найден')
        return None

# Функция для построения графика данных OHLC для валютной пары 'CNYRUB'
def plot_data(symbol='EURUSD'):
    # Загрузка данных из CSV-файла в DataFrame
    path_D = path.exists('data/fx_data.csv')
    if path_D:
        df = pd.read_csv('data/fx_data.csv')
    else:
        print('Файл не найден')
        return None
    
    # Преобразование столбца с датой в формат datetime, если он еще не в таком формате
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)


    
    # Построение графика
    plt.figure(figsize=(12, 8))
    plt.plot(df['Date'], df[symbol + '_Open'], label='Open')
    plt.plot(df['Date'], df[symbol + '_High'], label='High')
    plt.plot(df['Date'], df[symbol + '_Low'], label='Low')
    plt.plot(df['Date'], df[symbol + '_Close'], label='Close')
    plt.legend()
    plt.title('График котировок валютной пары ' + symbol)
    plt.show()

