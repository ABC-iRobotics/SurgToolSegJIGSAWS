# SurgToolSegJIGSAWS

## Overview
In the case of Minimally Invasive Surgery (MIS), the surgeon reaches the internal organs through small skin incisions, and the operating area is visualized by an endoscopic camera. MIS can be traditional (manually performed) or Robot-Assisted (RAMIS). While, the basics of these techniques are similar, the used instruments and endoscopic cameras can be significantly different. Semantic surgical tool segmentation in endoscopic images can be an important step toward pose estimation, task automation and skill assessment in MIS operations. The goal of automated skill assessment solutions is to replace the time-consuming experts’ opinion-based assessment techniques. The most used dataset for skill assessment is JIGSAWS that incorporates video and kinematic data. Tool segmentation in this dataset is challenged by different illumination conditions, low resolution, lack of ground truth labelling and the different background, while the usual training images are made in front of organs. In this work, Deep Neural Network and traditional image processing solutions were examined, aiming to segment the surgical tools to derive information for automated technical skill assessment in the case of RAMIS. We tested four different Deep Neural Network architectures (UNet, TernausNet-11, TernausNet-16, Linknet-34). and we trained these models with JIGSAWS dataset as well. The best overall result was achieved with TernausNet-11 trained on JIGSAWS with Intersection over Union (IoU) = 70.96, Dice Coefficient = 79.91 Accuracy = 97.38. But Unet and LinkNet34 could also achieve good results on videos of specific surgical tasks. Moreover, an efficient ground truth labelling method was proposed for the JIGSAWS dataset with the help of the Optical Flow algorithm.

## Data 
We used JIGSAWS dataset, for which we created hand-labelled mask. 
The labes were created for the following videos from tha dataset:
   * Knot_Tying_B001_capture1.avi, Knot_Tying_B002_capture1.avi, Knot_Tying_B003_capture1.avi
   * Needle_Passing_B001_capture1.avi, Needle_Passing_B002_capture1.avi, Needle_Passing_B003_capture1.avi
   * Suturing_B001_capture1.avi, Suturing_B002_capture1.avi, Suturing_B003_capture1.avi
This can be found in the corresponding folders.

## Dependencies
  * Numpy==1.20.2  
  * Opencv-python==4.5.3.56 
  * Scipy==1.6.2   
  * Tqdm==4.59.0  
  
To install these dependencies you can use requirements.txt file.   
## Instructions
You should organize your folders the following way:

    ├── data
    │   ├── Knot_tying
    │   │   ├── B001
    │   │   ├── B002
    │   │   ├── B003
    │   ├── Needle_passing
    │   │   ├── B001
    │   │   ├── B002
    │   │   ├── B003
    │   ├── Suturing
    │   │   ├── B001
    │   │   ├── B002
    │   │   ├── B003
    ├── ground_truth
    │   ├── Knot_tying
    │   │   ├── B001
    │   │   ├── B002
    │   │   ├── B003
    │       .......................
    ├── mask
    │   ├── Knot_tying
    │   │   ├── B001
    │   │   ├── B002
    │   │   ├── B003
    │       .......................
    
First the frames are created with the  -  ``video_to_frame.py`` file where you have to set the directory of the JIGSAWS video. 
Then, the ground truth labels are generated for this video with the help of  -  ``ground_truth_generation.py`` file.
    
## Citation
