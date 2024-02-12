import sys

from models import neuron_model_0_2
from data.geting_data import get_candles
from data.get_canles_online import get_candles_online, get_dataset_online
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time

# Функция для подготовки данных для нейронной сети
def prepare_data(data, targets):
    if data is None or targets is None:
        print('Ошибка: Данные отсутствуют.')
        return None, None

    # Преобразование данных и целей в тензоры
    dataset = torch.tensor(data, dtype=torch.float32)
    target_tensor = torch.tensor(targets, dtype=torch.float32)

    # Проверка наличия NaN в данных
    if torch.isnan(dataset).any() or torch.isnan(target_tensor).any():
        print('Обнаружены NaN в данных. Обработка...')
        # Обработка NaN (например, удаление строк с NaN или замена их на среднее значение)
        dataset = dataset[~torch.isnan(dataset).any(dim=1)]  # Удаление строк с NaN
        target_tensor = target_tensor[~torch.isnan(target_tensor)]

        # Проверка наличия NaN в данных после обработки
        if torch.isnan(dataset).any() or torch.isnan(target_tensor).any():
            print('Обнаружены NaN в данных после обработки')
            return None, None
        print('Обработка NaN завершена')

    if dataset.numel() == 0 or target_tensor.numel() == 0:
        print('Нет данных после обработки NaN')
        return None, None

    # print('dataset после обработки:', dataset, end='\n\n')
    # print('targets после обработки:', target_tensor, end='\n\n')

    return dataset, target_tensor

# функция для расчёта данных в (разницу, процент, объем, линейный процент)
def calculate_metrics(original_data, predicted_data):
    # Вычисляем разницу
    difference = predicted_data - original_data

    # Вычисляем процентное изменение
    percentage_change = (difference / original_data) * 100

    # Объем - просто предсказанные данные
    volume = predicted_data


# функция для запуска нейронной сети и получения результата
def neuron_manager():
    # Получаем данные OHLC для валютной пары 'USDRUB' (пример данных: open, high, low, close)
    data = get_dataset_online('C:USDRUB')
    print('Данные по свечам получены', pd.DataFrame(data, columns=['open', 'high', 'low', 'close']), sep='\n', end='\n\n')

    # получаем данные реального времени (пример данных: open, high, low, close)
    data_online = get_candles_online('C:USDRUB')
    print('Данные по свечам получены', pd.DataFrame(data_online, columns=['open', 'high', 'low', 'close']), sep='\n', end='\n\n')


    # Имеется ли ваши цели (targets)? Пример: targets = [0, 1, 1, 0, ...]
    targets = data.copy()

    # Подготавливаем данные для нейронной сети
    dataloader, target_tensor = prepare_data(data, targets)
    if dataloader is None or target_tensor is None:
        print('Ошибка: Данные отсутствуют.')
        return None
    
    # Подготовка новых данных для нейронной сети в реальном времени (пример данных: open, high, low, close)
    dataloader_online, target_tensor_online = prepare_data(data_online, targets)
    if dataloader_online is None or target_tensor_online is None:
        print('Ошибка: Данные отсутствуют.')
        return None

    print('Данные подготовлены')
    # Выведем первый батч для примера (первые 32 строки)
    for batch in dataloader:
        print('batch:', batch, sep='\n', end='\n\n')
        break
        

    # Предполагаем, что 'data' - это DataLoader
    # for batch in data:
    #     tensor = torch.from_numpy(batch).float()

    input_size = 4  # количество аспектов
    epochs = 800  # количество эпох обучения
    path_to_model = 'saved_model/neuron_model_seed_{}_data_{}.pth'.format(torch.get_rng_state().numpy().sum(), time.strftime("%Y-%m-%d_%H-%M-%S"))  # путь к файлу модели (если путь к файлу не указан, то сохранение не производится)
    model_path = "saved_model/neuron_model_seed_313851.pth"  # путь к файлу модели (если путь к файлу не указан, то сохранение не производится)
    enable_train = input('Начать обучение модели? (y/n) или нажмите Enter любое другое значение для начала предсказания цены: ')

    # Запуск нейронной сети и seed для воспроизводимости результатов
    print('Семя: {} (для воспроизводимости результатов)'.format(torch.initial_seed()))

    while True:
        try:
            # Создаем экземпляр модели
            neuron = neuron_model_0_2.NeuronModel(dataloader, input_size, epochs)
            # Начинаем обучение модели
            if enable_train == 'y':
                print('Начинаем обучение модели')
                neuron.train(dataloader, target_tensor, epochs)
                # Сохраняем модель в файл (если путь к файлу указан то сохранение производится)
                if neuron.set_model_path(path_to_model) is None:
                    neuron.save_model()
            else:
                # Выводим модель
                # print(neuron.model)
                # Выводим путь к файлу модели
                print(neuron.get_model_path())
                neuron.set_model_path(model_path)
                # Загружаем модель из файла
                neuron.load_model()
                # Выводим модель
                # print(neuron.model)

                # Выводим время hh:mm:ss (пример вывода: _*20 РЕАЛЬНОЕ ВРЕМЯ: 17:33:00 _*20)
                print('_' * 20, 'РЕАЛЬНОЕ ВРЕМЯ: ', time.strftime("%H:%M:%S"), '_' * 20)

                # предсказание по данным тестовой выборки (пример данных: open, high, low, close)
                for original_data in dataloader_online:
                    # Предсказание с использованием нейронной сети
                    prediction_data = neuron.predict(original_data)

                    # Расчет метрик для текущего временного шага
                    result_metrics = calculate_metrics(original_data, prediction_data)

                # Здесь вы можете делать что-то с результатами, например, сохранять их или выводить
                print(result_metrics)
                # Пауза 20 секунд
                print('_' * 20, 'ПАУЗА 10 СЕКУНД', '_' * 20)
                time.sleep(10)

        # выход из цикла при нажатии esc
        except EOFError:
            break