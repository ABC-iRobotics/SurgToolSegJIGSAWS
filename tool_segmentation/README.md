# SurgToolSegJIGSAWS

## Overview
In the case of Minimally Invasive Surgery (MIS), the surgeon reaches the internal organs through small skin incisions, and the operating area is visualized by an endoscopic camera. MIS can be traditional (manually performed) or Robot-Assisted (RAMIS). While, the basics of these techniques are similar, the used instruments and endoscopic cameras can be significantly different. Semantic surgical tool segmentation in endoscopic images can be an important step toward pose estimation, task automation and skill assessment in MIS operations. The goal of automated skill assessment solutions is to replace the time-consuming experts’ opinion-based assessment techniques. The most used dataset for skill assessment is JIGSAWS that incorporates video and kinematic data. Tool segmentation in this dataset is challenged by different illumination conditions, low resolution, lack of ground truth labelling and the different background, while the usual training images are made in front of organs. In this work, Deep Neural Network and traditional image processing solutions were examined, aiming to segment the surgical tools to derive information for automated technical skill assessment in the case of RAMIS. We tested four different Deep Neural Network architectures (UNet, TernausNet-11, TernausNet-16, Linknet-34). and we trained these models with JIGSAWS dataset as well. The best overall result was achieved with TernausNet-11 trained on JIGSAWS with Intersection over Union (IoU) = 70.96, Dice Coefficient = 79.91 Accuracy = 97.38. But Unet and LinkNet34 could also achieve good results on videos of specific surgical tasks. Moreover, an efficient ground truth labelling method was proposed for the JIGSAWS dataset with the help of the Optical Flow algorithm.
## Data 
We used JIGSAWS dataset, which can be access freely online. (The ground truth dataset is also available, write to dora.papp@irob.uni-obuda.hu or renata.elek@irob.uni-obuda.hu.)
## Dependencies
  * Numpy==1.20.2  
  * Albumentations==0.5.2
  * Opencv-python==4.5.3.56   
  * Torch==1.8.1+cu102  
  * Torchvision==0.9.1    
  * Tqdm==4.59.0  
  
To install these dependencies you can use requirements.txt file.   
## Instructions
This work is based on https://github.com/ternaus/robot-surgery-segmentation.
You should organize your folders the following way:

    ├── data
    │   ├── models
    │   │   ├── JIGSAWStrained
    │   │   │   ├── UNet
    │   │   │   └── UNet11
    |   |   ....................... 
    │   ├── test_JIGSAWS
    │   │   ├── Knot_tying
    │   │   │   ├── Lab
    │   │   │   └── Nothing
    │   │   ├── Needle_passing
    │   │   │   ├── Lab
    │   │   │   └── Nothing
    |   |   ....................... 
    │   └── train_JIGSAWS
    │       ├── instrument_dataset_1
    │       │   ├── binary_mask
    │       │   ├── images
    │       .......................
    ├── predictions
### 1. Training
For training put and organize the training data as it is shown above. We used 9 videos for training, therefore there are 9 instument_dataset folder.

We used 3-fold-cross validation, so with 'fold' it can be set how we split the data into validation and training set.

The main file for training is  -  ``train.py``. We used the following bash script:
    
    python train.py --device-ids 0 --batch-size 5 --fold 0 --workers 0 --lr 0.0001 --n-epochs 1  --jaccard-weight 1 --model LinkNet34

### 2. Mask generation
With -  ``generate_mask.py`` you can generate masks with the help of pre-trained models. 

First, the frames are made with the help of  -  ``JIGSAWS_prepare_data.py``, where the path of the video is set, and also LAB color space conversion can be done which is useful if we used models trained on MICCAI dataset.

We used the following script for creating the files from videos:
    
    python JIGSAWS_prepare_data.py --path_of_video (Your folder) --task_type Suturing --type_of_preprocess Nothing
    
Then we used the following script to create the masks:

    python generate_masks.py --model_path data/models/JIGSAWStrained/UNet --model_type UNet --output_path predictions --batch-size 1 --fold 0 --workers 0
    --pred_data_path data/test_JIGSAWS/Suturing/Nothing
    
(In drive folders you can find models, which were trained on JIGSAWS, write to dora.papp@irob.uni-obuda.hu or renata.elek@irob.uni-obuda.hu.)
    
### 3. Evaluation
Evaluation is done with -  ``evaluate.py``.

It calculates the Jaccard-index, Dice coefficients and Accuracy.

We used the following script:

    python evaluate.py --target_path predictions --gt_path data/train_JIGSAWS/instrument_dataset_1
