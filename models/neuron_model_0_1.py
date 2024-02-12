# Библиотеки для работы с нейронной сетью
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Локальные библиотеки
from data import geting_data as gd
from data import get_canles_online as gco

# Класс нейрона (простой линейный слой)
class Neuron(nn.Module):
    def __init__(self, input_size):
        super(Neuron, self).__init__()
        self.fc1 = nn.Linear(input_size, 4)
        self.fc2 = nn.Linear(4,4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc5 = nn.Linear(4, 4)
        self.fc6 = nn.Linear(4, 4)
        self.fc7 = nn.Linear(4, 4)
        self.fc8 = nn.Linear(4, 4)
        self.zero_grad()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        x7 = self.fc7(x)
        x8 = self.fc8(x)
        return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

# Функция для обучения нейрона на данных из dataloader
def train_neuron(neuron, dataloader, epochs, learning_rate=1e-4):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(neuron.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            output = neuron(data)
            # Проверка размерности данных перед изменением формы
            if data.dim() == 1:
                data = data.view(1, -1)
                # print('data:', data, end='\n\n')
            loss = loss_fn(output, data.view(-1, 4))
            loss.backward()
            optimizer.step()
            # выводим лосс каждые 500 эпох и 'Output:', покозать что выводит (пример: open, high, low, close) 
            if epoch % 1 == 0:
                print('DateTime:', pd.Timestamp.now(), 
                      'Epoch:', epochs,
                      'Loss:', loss.item(),
                    #   'Optimizer:', optimizer.state_dict(),
                      'Output:', pd.DataFrame(output.detach().numpy().reshape(-1, 4), columns=['open', 'high', 'low', 'close']),
                      'Dataset:', pd.DataFrame(data.detach().numpy().reshape(-1, 4), columns=['open', 'high', 'low', 'close']),
                    #   file=open('logs/loss.txt', 'a'), 
                      sep=' \n', end='\n\n')
                time.sleep(0.05)
             
def main_neuron():
    # условие для отчистки файла loss.txt при запуске программы 
    if True:
        if open('logs/loss.txt', 'w').close() == None:
            print('Файл loss.txt очищен')
        else:
            print('Файл не очищен')


    # Пример данных (замените этот блок на загрузку ваших временных рядов)
    # Получаем данные OHLC для валютной пары 'USDRUB'
    data = gd.get_candles(symbol='USDRUB')
    # print('data:', data, end='\n\n')

    # Открываем файл для записи data и выводим в него данные (open, high, low, close)
    print('data:', data, file=open('logs/data.txt', 'a'), sep='\n', end='\n\n')


    # Создаем датасет и загружаем в него данные
    dataset = torch.tensor(data, dtype=torch.float32)

    # Проверка наличия NaN в данных
    if torch.isnan(dataset).any():
        print('Обнаружены NaN в данных. Обработка...')
        
        # Обработка NaN (например, удаление строк с NaN или замена их на среднее значение)
        dataset = dataset[~torch.isnan(dataset).any(dim=1)]  # Удаление строк с NaN

        # Проверка наличия NaN в данных после обработки
        if torch.isnan(dataset).any():
            print('Обнаружены NaN в данных после обработки')
            return None 
        print('Обработка NaN завершена')

    if dataset.numel() == 0:
        print('Нет данных после обработки NaN')
        return None

    print('dataset:', dataset, end='\n\n')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = 4  # количество аспектов
    neuron = Neuron(input_size)

    # Обучаем нейрон на данных из dataloader
    train_neuron(neuron, dataloader, epochs=100)

    # Сохраняем данные из результата обучения нейрона в файл по строкам (open, high, low, close)
    # print('Выводим данные:', pd.DataFrame(neuron(torch.tensor(data, dtype=torch.float32)).detach().numpy().reshape(-1, 4), columns=['open', 'high', 'low', 'close']), file=open('output.txt', 'a'), sep='\n', end='\n\n')

    # Сохраняем нейронную сеть в файл
    torch.save(neuron.state_dict(), 'neuron_model_0_1.pth')

    # Загружаем нейронную сеть из файла
    neuron.load_state_dict(torch.load('neuron_model_0_1.pth'))
    # Выводим данные из нейронной сети в виде таблицы (open, high, low, close) и сохраняем в файл output.txt
    print('Выводим данные:', pd.DataFrame(neuron(torch.tensor(data, dtype=torch.float32)).detach().numpy().reshape(-1, 4), columns=['open', 'high', 'low', 'close']),
            sep='\n', end='\n\n')

    # Получаем новые данные OHLC для валютной пары 'USDRUB'
    # new_data = gco.get_canles_online(symbols='USDRUB')
    print('new_data:', neuron, end='\n\n')

  