
"""

Implementation based on https://github.com/ternaus/robot-surgery-segmentation

"""

from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def accuracy(y_true, y_pred):
        y_pred = (y_pred>0).flatten()
        y_true = (y_true>0).flatten()

        tp = (y_true*y_pred).sum()
        tn = ((y_pred==0) * (y_true ==0)).sum()

        return ((tp+tn) / (y_pred.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--target_path', type=str, default='predictions', help='path with predictions')
    arg('--gt_path', type=str, default='data/train_JIGSAWS/instrument_dataset_7', help='path with gt')
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []
    result_accuracy = []
   

    for file_name in (Path(args.gt_path) / 'binary_masks').glob('*'):

        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = (Path(args.target_path) / file_name.name)

        y_pred = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)

        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
        result_accuracy += [accuracy(y_true, y_pred)]

    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Accuracy = ', np.mean(result_accuracy), np.std(result_accuracy))
