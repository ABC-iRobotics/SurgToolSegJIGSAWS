"""

Implementation from https://github.com/ternaus/robot-surgery-segmentation

"""
from pathlib import Path

def get_split(fold):
    folds = {0: [1, 4, 9],
             1: [2, 5, 8],
             2: [3, 6, 7]}

    data_path = Path('data')
    train_path = data_path / 'train_JIGSAWS'

    train_file_names = []
    val_file_names = []

    for instrument_id in range(1, 10):
        if instrument_id in folds[fold]:
            val_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))
        else:
            train_file_names += list((train_path / ('instrument_dataset_' + str(instrument_id)) / 'images').glob('*'))

    return train_file_names, val_file_names
