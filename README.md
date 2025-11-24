# 3D Medical Image Segmentation of the Heart
This project implements and compares augmentation techniques for medical image segmentation using a standardized 3D U-Net. The goal is to examine how different augmentation techniques affect model generalization on small volumetric datasets.

The repository contains:
- Full code for data loading, preprocessing, augmentation, training, evaluation, and inference
- Generated visualizations (augmented samples, loss curves, predicted segmentations)
- README.md with instructions
- requirements.txt for environment setup
- Experimental results for baseline, geometric–intensity augmentation, and elastic deformation augmentation

## Objectives
The main objectives of this project are as follows:
- Implement two distinct 3D augmentation pipelines: a geometric--intensity pipeline applying flipping, 3D rotation, translation, and intensity jitter, and an elastic deformation method introducing non-linear anatomical variations.
- Integrate both augmentation methods into a standardized 3D U-Net for fair and controlled experimentation.
- Compare baseline and augmented models using Dice score, Hausdorff distance, sensitivity, and precision to quantify the effect of augmentation.
- Visualize results, including augmented images, training curves, and predicted segmentations, to support qualitative analysis.

## Repository Structure

```
`
├── Logs                                    # TensorBoard for tracking training progress of each experiment.
│   ├── Baseline
│   │   └── events.out.tfevents.1763888258.dd0ba72e8b53.2208.0.v2
│   ├── Elastic Deformation
│   │   └── events.out.tfevents.1763897637.dd0ba72e8b53.56478.0.v2
│   └── Geometric-Intensity
│   └── events.out.tfevents.1763892690.dd0ba72e8b53.28022.0.v2
├── README.md                               
├── requirements.txt                        
├── Results                                 
│   ├── Baseline
│   │   ├── Base_Loss.png                   # loss and dice score plot over each epochs.
│   │   ├── Metrics.csv                     # metric results for each individua test sample
│   │   └── Predictions                     # predictions made by the trained model on test samples
│   ├── Elastic Deformation
│   │   ├── Elastic_Deformation_Loss.png    # loss and dice score plot over each epochs.
│   │   ├── Examples                        # visualizations of the augmented image
│   │   └── Predictions                     # predictions made by the trained model on test samples
│   │   ├── Metrics.csv                     # metric results for each individua test sample
│   └── Geometric–intensity
│   │   ├── Intensity_Loss.png              # loss and dice score plot over each epochs.
│   │   ├── Examples                        # visualizations of the augmented image
│   │   └── Predictions                     # predictions made by the trained model on test samples
│   │   ├── Metrics.csv                     # metric results for each individua test sample
│   ├── Comparison.csv                      # mean results across all test patients for all experiments
└── src                                     
    ├── augmentation.py                     # data augmentation pipelines
    ├── config.py                           # configuration settings
    ├── data_loader.py                      # data loading, preprocessing
    ├── evaluate.py                         # evaluating the trained models
    ├── model.py                            # 3D U-Net model architecture
    ├── train.py                            # training the models
    └── utils.py                            # Helper functions
```

## Dataset Download (Google Colab)
To download the dataset directly inside Google Colab, run:
```python
!gdown --id 1uzBJTwJzpE_cIYLoHRfBnvDmcuzz07t2 -O dataset.zip
```
Then unzip the dataset:
```python
!unzip dataset.zip -d dataset/
```
After unzipping, the dataset directory on Google Colab will look like this:
```
/content/dataset/heart_dataset
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```
The dataset contains 17 3D volumes for training and 3 independent 3D volumes for testing. The training set is used to develop the model, while the test set is reserved exclusively for final performance evaluation.

## Installation in Google Colab
After downloading the dataset in Google Colab, clone the repository using 
```bash
!git clone https://github.com/daniyalbinasif/3D_segmentation.git
%cd 3D_segmentation
```
and then install dependencies using
```bash
!pip install -r requirements.txt
```
## Training in Google Colab
Baseline (No Augmentation):
```bash
!python /content/3D_segmentation/src/train.py baseline
```

Geometric–Intensity Augmentation:
```bash
!python /content/3D_segmentation/src/train.py intensity
```

Elastic Deformation Augmentation:
```bash
!python /content/3D_segmentation/src/train.py elastic
```
## Evaluation
After training any model, run:
```bash
!python /content/3D_segmentation/src/evaluate.py
```
This produces:
- Dice, Hausdorff distance, Precision, Sensitivity per test case
- Mean metrics for each experiment
- Visualizations of training and validation loss over each epoch
- Visualizations of predictions generated by trained models
- Mean results across all test patients for all experiments




