import time

import asyncio

import pandas as pd

import torch

async def adef():

    # print('______________________start function adef______________________')

    for i in range(3):

        print('seed:', torch.initial_seed(), 'name:', asyncio.current_task().get_name(), 'i:', i)

        print('DateTime:', pd.Timestamp.now(), 'i:', i)

        # Добавляем асинхронную паузу
        # await asyncio.sleep(1)

async def bdef():

    # print('______________________start function bdef______________________')

    for i in range(3):

        print('seed:', torch.initial_seed(), 'name:', asyncio.current_task().get_name(), 'i:', i)

        print('DateTime:', pd.Timestamp.now(), 'i:', i)

        # Добавляем асинхронную паузу
        # await asyncio.sleep(1)
        '''Coordinate'''


# Определение корутины, которая запускает asyncio.gather
async def run_gather():
    await asyncio.gather(adef(), bdef())

# Передача корутины в asyncio.run
asyncio.run(run_gather(), debug=True)
