from os import path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Нейронная сеть для предсказания свечей на основе предыдущих свечей и текущих данных о свече (open, close, high, low, volume)from os import path
class NeuronModel(nn.Module):
    # Конструктор класса NeuronModel
    def __init__(self, dataloader=None, input_size=4, hidden_size=4096, output_size=4, model_path=None):
        super(NeuronModel, self).__init__() # Вызов конструктора класса nn.Module
        self.dataloader = dataloader    # dataloader - данные для обучения нейронной сети (по умолчанию None)
        self.input_size = input_size    # input_size - количество аспектов во входном слое (по умолчанию 4)
        self.hidden_size = hidden_size  # hidden_size - количество нейронов в скрытом слое (по умолчанию 32)
        self.output_size = output_size  # output_size - количество аспектов в выходном слое (количество предсказываемых свечей)
        self.model_path = model_path    # model_path - путь к файлу модели нейронной сети (по умолчанию None)
    
        self.model = self.load_model()  # model - модель нейронной сети (последовательная модель)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # optimizer - оптимизатор модели (Adam)
        self.criterion = nn.MSELoss()   # criterion - функция потерь (MSELoss)
    
    # Загрузка модели из файла или создание новой модели (если файл не найден)
    def load_model(self):
        if self.model_path is not None and path.exists(self.model_path):
            print('Загрузка модели из файла:', self.model_path)
            model = torch.load(self.model_path)
        else:
            print('Создание новой модели')
            model = self.create_model()
        return model
    
    # Создание модели нейронной сети (последовательная модель)
    def create_model(self):
        # Создание 4 уровней нейронной сети
        model_l0 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l2 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l3 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l4 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l5 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l6 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        model_l7 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Identity()
        )
        # Создание последовательной модели
        model = nn.Sequential(
            model_l0,
            model_l1,
            model_l2,
            model_l3,
            model_l4,
            model_l5,
            model_l6,
            model_l7
        )
        return model
    
    # Предсказание на основе данных из dataloader (по умолчанию - предыдущих свечей)
    def forward(self, x):
        output = self.model(x)
        return output
    
    # Предсказание на основе входных данных
    def predict(self, data):
        data = self.data_to_tensor(data)
        output = self.forward(data)
        return output
    
    # Преобразование данных в тензор PyTorch
    def data_to_tensor(self, data):
        # Проверка данных на наличие тензора. Если данные являются тензором, то возвращаем их
        # Если данные не являются тензором, то преобразуем их в тензор
        if isinstance(data, torch.Tensor) or data is None:
            # Вывод информации о том, что данные являются тензором
            print('Данные являются тензором. SEED:', torch.initial_seed())

            return data
        else:
            print('Преобразование данных в тензор. SEED:', torch.initial_seed())

            # Преобразование данных в тензор
            data = torch.from_numpy(data).float64()
            # Вывод информации о том, что данные преобразованы в тензор
            print('Данные преобразованы в тензор. SEED:', torch.initial_seed())
            return data
        
    # Обучение модели на основе входных данных и целевых значений
    # data - входные данные (данные о свечах)
    # targets - целевые значения (предсказание свечи)
    def train(self, data, targets=None, epochs=None):
        # Преобразование данных и целей в тензоры
        dataset, target_tensor = self.data_to_tensor(data).view(-1, self.input_size), self.data_to_tensor(targets).view(-1, self.output_size)
        # epochs - количество эпох обучения модели
        for epoch in range(epochs):
            # Получение предсказания на основе входных данных
            output = self.forward(dataset)
            # Вычисление ошибки предсказания
            loss = self.criterion(output, target_tensor)
            # Очистка градиентов
            self.optimizer.zero_grad()
            # Обратное распространение ошибки
            loss.backward()
            # Обновление весов
            self.optimizer.step()
            # Вывод информации об обучении и сочик эпохи
            if epoch % 1 == 0:
                print('Epoch:', epoch, 'Loss:', loss.item(), sep=' ')
                print('Output:', pd.DataFrame(output.detach().numpy().reshape(-1, 4), columns=['open', 'high', 'low', 'close']), sep='\n', end='\n\n')
        # Возвращение предсказания
    
    # Сохранение модели в файл (если путь к файлу не указан, то сохранение не производится)
    # При сохранении модели в файл сохраняются веса модели, и сама модель (со всеми параметрами) 
    # И при сохранении модели в файл и весов в файл, если есть такими названия то создается новый файл
    def save_model(self):
        if self.model_path is not None:
            print('Сохранение модели в файл:', self.model_path)
            torch.save(self.model, self.model_path)
        else:
            print('Ошибка: Не указан путь к файлу модели')
    
    # Получение пути к файлу модели
    def get_model_path(self):
        return self.model_path
    
    # Установка нового пути к файлу модели
    def set_model_path(self, model_path):
        self.model_path = model_path

# Пример использования класса NeuronModel