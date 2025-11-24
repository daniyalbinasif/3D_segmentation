# 3D medical image segmentation of the heart
This project implements and compares augmentation techniques for medical image segmentation using a standardized 3D U-Net. The goal is to examine how different augmentation techniques affect model generalization on small volumetric datasets.

The repository contains:

- Full code for data loading, preprocessing, augmentation, training, evaluation, and inference
- Generated visualizations (augmented samples, loss curves, predicted segmentations)
- README.md with instructions
- requirements.txt for environment setup
- Experimental results for baseline, geometric–intensity augmentation, and elastic deformation augmentation


```
`
/content/3D_segmentation
├── Logs                                    # Stores TensorBoard event files for tracking training progress and metrics.
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
│   │   ├── Base_Loss.png
│   │   ├── Metrics.csv
│   │   └── Predictions
│   ├── Elastic Deformation
│   │   ├── Elastic_Deformation_Loss.png
│   │   ├── Examples
│   │   └── Predictions
│   │   ├── Metrics.csv
│   └── Geometric–intensity
│   │   ├── Intensity_Loss.png
│   │   ├── Examples
│   │   └── Predictions
│   │   ├── Metrics.csv
│   ├── Comparison.csv
└── src                                     # Source code directory, containing all Python scripts:
    ├── **augmentation.py**                 # data augmentation pipelines.
    ├── **config.py**                       # configuration settings.
    ├── **data_loader.py**                  # data loading, preprocessing, and batching.
    ├── **evaluate.py**                     # evaluating the trained models.
    ├── **model.py**                        # 3D U-Net model architecture.
    ├── **train.py**                        # training the models.
    └── **utils.py**                        # Helper functions
```
