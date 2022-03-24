
"""

Implementation from https://github.com/ternaus/robot-surgery-segmentation

"""

import argparse
from dataset import RoboticsDatasetPred
import cv2
from models import UNet16, LinkNet34, UNet11, UNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Normalize


binary_factor = 255

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type):

    num_classes = 1

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size, to_path, img_transform):

    loader = DataLoader(
        dataset=RoboticsDatasetPred(from_file_names, transform=img_transform),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)
            
            outputs = model(inputs)

            for i, image_name in enumerate(paths):

                factor = binary_factor
                    
                t_mask = (((F.sigmoid(outputs[i, 0]).data.cpu().numpy())>0.8)*factor).astype(np.uint8)
                    
                mask = cv2.cvtColor(t_mask,cv2.COLOR_GRAY2RGB)
                img_original = cv2.imread(str(image_name))
                added_image = cv2.addWeighted(img_original,0.9,mask,0.5,0)

                cv2.imwrite(str(to_path / (Path(paths[i]).stem + '.png')), t_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    #arg('--model_path', type=str, default= 'runs/debug', help='path to model folder')
    arg('--model_path', type=str, default= 'data/models/article/LinkNet34', help='path to model folder')
    arg('--model_type', type=str, default='LinkNet34', help='network architecture',choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34'])
    arg('--output_path', type=str, help='path to save images', default='predictions')
    arg('--batch-size', type=int, default=1)
    arg('--fold', type=int, default=2, choices=[0, 1, 2])
    arg('--workers', type=int, default=0)
    arg('--pred_data_path', type=str, default='data/test_JIGSAWS/Suturing/Nothing')

    args = parser.parse_args()

    pred_data_path = Path(str(args.pred_data_path))
    file_names = list((pred_data_path).glob('*'))

    model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
                          model_type=args.model_type)

    print('num file_names = {}'.format(len(file_names)))

    output_path = Path(args.output_path)

    predict(model, file_names, args.batch_size, output_path,
                img_transform=img_transform(p=1))
