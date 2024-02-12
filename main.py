import errno

from models.neuron_manager import neuron_manager

if __name__ == '__main__':
    # Вызываем функцию neuron_manager() из models/neuron_manager.py
    if neuron_manager():
        print(errno)
