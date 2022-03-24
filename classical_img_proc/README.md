# SurgToolSegJIGSAWS

## Overview
In the case of Minimally Invasive Surgery (MIS), the surgeon reaches the internal organs through small skin incisions, and the operating area is visualized by an endoscopic camera. MIS can be traditional (manually performed) or Robot-Assisted (RAMIS). While, the basics of these techniques are similar, the used instruments and endoscopic cameras can be significantly different. Semantic surgical tool segmentation in endoscopic images can be an important step toward pose estimation, task automation and skill assessment in MIS operations. The goal of automated skill assessment solutions is to replace the time-consuming experts’ opinion-based assessment techniques. The most used dataset for skill assessment is JIGSAWS that incorporates video and kinematic data. Tool segmentation in this dataset is challenged by different illumination conditions, low resolution, lack of ground truth labelling and the different background, while the usual training images are made in front of organs. In this work, Deep Neural Network and traditional image processing solutions were examined, aiming to segment the surgical tools to derive information for automated technical skill assessment in the case of RAMIS. We tested four different Deep Neural Network architectures (UNet, TernausNet-11, TernausNet-16, Linknet-34). and we trained these models with JIGSAWS dataset as well. The best overall result was achieved with TernausNet-11 trained on JIGSAWS with Intersection over Union (IoU) = 70.96, Dice Coefficient = 79.91 Accuracy = 97.38. But Unet and LinkNet34 could also achieve good results on videos of specific surgical tasks. Moreover, an efficient ground truth labelling method was proposed for the JIGSAWS dataset with the help of the Optical Flow algorithm.

## Data 
We tried different datasets for this algorithm.
The three datasets are the following:
  * JIGSAWS
  * Synthetis MICCAI
  * MICCAI

## Dependencies
  * Numpy==1.20.2  
  * Opencv-python==4.5.3.56 
  * Scipy==1.6.2
  * Scikit-image==0.18.1   
  * Tqdm==4.59.0  
  
To install these dependencies you can use requirements.txt file.   
## Instructions
You should organize your folders the following way:

    ├── data
    │   ├── JIGSAWS
    │   │   ├── Instrument_dataset_1
    │   │   │   ├── binary_masks
    │   │   │   ├── images
    │   │   ├── Instrument_dataset_2
    │   │   ├── Instrument_dataset_3
    │   │    .......................
    │   ├── MICCAI
    │   │   ├── Instrument_dataset_1
    │   │   ├── Instrument_dataset_2
    │   │   ├── Instrument_dataset_3
    │   │    .......................
    │   ├── SYNTHETIC
    │   │    .......................
    ├── steps
    │   ├── Cropped
    │   │   ├── JIGSAWS
    │   │   ├── MICCAI
    │   │   ├── SYNTHETIC
    │   ├── optical_flow_hsv
    │   │   ├── JIGSAWS
    │   │   ├── MICCAI
    │   │   ├── SYNTHETIC
    │   ├── postprocess
    │   │   ├── JIGSAWS
    │   │   ├── MICCAI
    │   │   ├── SYNTHETIC
    │   ├── track
    │   │   ├── JIGSAWS
    │   │   ├── MICCAI
    │   │   ├── SYNTHETIC
    
To get a segmentation with classical image processing methods we run the following script:
    python train.py --device-ids 0 --batch-size 5 --fold 0 --workers 0 --lr 0.0001 --n-epochs 1  --jaccard-weight 1 --model LinkNet34
    python opticalflow.py --original_height 480 --original_width 640 --startx 96 --starty 0 --data_type JIGSAWS --cropped_train_path steps/Cropped/JIGSAWS
    
## Citation
