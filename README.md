# 3D Segmentation
End-to-end pipeline for 3D medical image segmentation of the heart. Implements two augmentation strategies—geometric/intensity (flip, rotate, shift, jitter) and elastic deformation—trains 3D U-Net, and compares baseline vs augmented models using Dice, Hausdorff, sensitivity, and precision.

```
`
/content/3D-medical-image-segmentation-of-the-heart
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
