# SurgToolSegJIGSAWS

## Overview
In the case of Minimally Invasive Surgery (MIS), the surgeon reaches the internal organs through small skin incisions, and the operating area is visualized by an endoscopic camera. MIS can be traditional (manually performed) or Robot-Assisted (RAMIS). While, the basics of these techniques are similar, the used instruments and endoscopic cameras can be significantly different. Semantic surgical tool segmentation in endoscopic images can be an important step toward pose estimation, task automation and skill assessment in MIS operations. The goal of automated skill assessment solutions is to replace the time-consuming experts’ opinion-based assessment techniques. The most used dataset for skill assessment is JIGSAWS that incorporates video and kinematic data. Tool segmentation in this dataset is challenged by different illumination conditions, low resolution, lack of ground truth labelling and the different background, while the usual training images are made in front of organs. In this work, Deep Neural Network and traditional image processing solutions were examined, aiming to segment the surgical tools to derive information for automated technical skill assessment in the case of RAMIS. We tested four different Deep Neural Network architectures (UNet, TernausNet-11, TernausNet-16, Linknet-34). and we trained these models with JIGSAWS dataset as well. The best overall result was achieved with TernausNet-11 trained on JIGSAWS with Intersection over Union (IoU) = 70.96, Dice Coefficient = 79.91 Accuracy = 97.38. But Unet and LinkNet34 could also achieve good results on videos of specific surgical tasks. Moreover, an efficient ground truth labelling method was proposed for the JIGSAWS dataset with the help of the Optical Flow algorithm.

## List of Packages
* [classical_img_proc](https://github.com/ABC-iRobotics/SurgToolSegJIGSAWS/tree/main/classical_img_proc)
* [groundtruth_generation](https://github.com/ABC-iRobotics/SurgToolSegJIGSAWS/tree/main/groundtruth_generation)
* [tool_segmentation](https://github.com/ABC-iRobotics/SurgToolSegJIGSAWS/tree/main/tool_segmentation)

## Citation

If you find this work useful for your publications, please consider citing:

    @inproceedings{papp2022surgical,
      title={Surgical tool segmentation on the jigsaws dataset for autonomous image-based skill assessment},
      author={Papp, D{\'o}ra and Elek, Ren{\'a}ta Nagyn{\'e} and Haidegger, Tam{\'a}s},
      booktitle={2022 IEEE 10th Jubilee International Conference on Computational Cybernetics and Cyber-Medical Systems (ICCC)},
      pages={000049--000056},
      year={2022},
      organization={IEEE}
    }
