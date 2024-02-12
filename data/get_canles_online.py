import requests
from datetime import datetime, timedelta
import pandas as pd

# функция для получения данных свечей с помощью API Polygon.io
def get_polygon_data(api_key, symbol, start_date, end_date, timeframe, limit=10000):
    # формирование URL запроса с параметрами
    base_url = "https://api.polygon.io/v2/aggs/ticker/"
    request_url = f"{base_url}{symbol}/range/1/{timeframe}/{start_date}/{end_date}?limit={limit}&apiKey={api_key}"
    print('URL запроса:', request_url)

    # Запрос данных свечей
    response = requests.get(request_url)

    # Проверка статуса запроса
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

# функция для получения dataset из данных свечей (пример данных: open, high, low, close. или o, h, l, c)
def get_dataset_online(symbol):
    # API ключ и валютный пара (пример: 'WVs0xrELiZcusUA36kJgV8Y72UQJRmS5', 'C:USDRUB')
    api_key = 'WVs0xrELiZcusUA36kJgV8Y72UQJRmS5'
    # symbol = 'C:USDRUB'

    # текущая дата и время в формате "год-месяц-день"
    current_date = datetime.now()

    # начальная и конечная даты в формате "год-месяц-день"
    # начальная дата - 1 день назад
    start_date = '2015-08-15'
    end_date = current_date.strftime("%Y-%m-%d")

    # временной интервал в часах (пример: 'hour', 'day', 'week', 'month')
    timeframe = 'hour'

    # получение данных свечей с помощью API Polygon.io
    data = get_polygon_data(api_key, symbol, start_date, end_date, timeframe)

    # создание DataFrame из полученных данных
    df = pd.DataFrame(data['results'])
    df = df[['o', 'h', 'l', 'c']]
    print(df)

    return df.values

# функция для получения данных свечей и пердаваемых в нейронную сеть
def get_candles_online(symbol):
    # API ключ и валютный пара (пример: 'WVs0xrELiZcusUA36kJgV8Y72UQJRmS5', 'C:USDRUB')
    api_key = 'WVs0xrELiZcusUA36kJgV8Y72UQJRmS5'
    # symbol = 'C:USDRUB'

    # текущая дата и время в формате "год-месяц-день"
    current_date = datetime.now()

    # начальная и конечная даты в формате "год-месяц-день"
    # начальная дата - 1 день назад
    start_date = (current_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = current_date.strftime("%Y-%m-%d")


    # временной интервал в часах (пример: 'hour', 'day', 'week', 'month')
    timeframe = 'hour'

    # получаем данные свечей
    data = get_polygon_data(api_key, symbol, start_date, end_date, timeframe)

    # создаем DataFrame из полученных данных
    df = pd.DataFrame(data['results'])
    df = df[['o', 'h', 'l', 'c']]
    print(df)

    return df.values
