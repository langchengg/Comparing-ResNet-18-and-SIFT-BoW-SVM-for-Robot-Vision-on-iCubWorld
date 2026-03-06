# Comparing ResNet-18 and SIFT+BoW+SVM for Robot Vision on iCubWorld

This project provides a comparative study of deep learning (CNN) and traditional computer vision approaches for object recognition in robot vision scenarios using the **iCubWorld 1.0** dataset.

## Overview

The project compares two distinct approaches for image classification:

1. **Deep Learning Approach**: Fine-tuned ResNet-18 CNN with transfer learning
2. **Traditional CV Approach**: SIFT feature extraction + Bag-of-Words (BoW) representation + SVM/KNN classifiers

Both methods are evaluated on two domains from the iCubWorld dataset:
- **Human domain**: Objects captured by human operators
- **Robot domain**: Objects captured by the iCub robot

## Project Structure

```
├── code/
│   ├── cnn_resnet18.py           # ResNet-18 CNN implementation
│   ├── traditional_CV.py         # SIFT + BoW + SVM/KNN pipeline
│   ├── table.py                  # Results visualization scripts
│   ├── cnn_results_icub.csv      # CNN experiment results
│   ├── bow_sift_results_icub.csv # Traditional CV experiment results
│   ├── figures_cnn/              # CNN training curves, confusion matrices, error cases
│   └── figures_bow_sift/         # BoW+SVM confusion matrices and error cases
├── figures_bow/                  # Additional BoW experiment visualizations
├── LangCheng_report.pdf          # Detailed project report
└── README.md
```

## Methods

### ResNet-18 CNN Approach (`cnn_resnet18.py`)

- **Model**: Pre-trained ResNet-18 with frozen backbone and trainable classification head
- **Transfer Learning**: Uses ImageNet pre-trained weights
- **Classification Head**: Dropout (p=0.5) + Linear layer
- **Data Augmentation**: Random resized crop, horizontal flip, color jitter
- **Normalization**: ImageNet mean/std normalization

**Hyperparameter Grid Search:**
| Parameter | Values |
|-----------|--------|
| Batch Size | 16, 32 |
| Learning Rate | 1e-3, 5e-4 |
| Weight Decay | 0.0, 1e-4 |
| Epochs | 15 |

### Traditional CV Approach (`traditional_CV.py`)

- **Feature Extraction**: SIFT (Scale-Invariant Feature Transform)
- **Vocabulary Building**: MiniBatchKMeans clustering
- **Feature Encoding**: Bag-of-Words histogram with L2 normalization
- **Classifiers**: SVM (linear kernel) and KNN

**Hyperparameter Grid Search:**
| Parameter | Values |
|-----------|--------|
| Vocabulary Size | 50, 100 |
| Classifier | SVM, KNN |
| SVM C Parameter | 0.1, 1.0 |
| KNN Neighbors | 5, 10 |

## Results Summary

### CNN ResNet-18 Performance

| Domain | Best Test Accuracy | Hyperparameters |
|--------|-------------------|-----------------|
| Human | 97.86% | batch=16, lr=1e-3, wd=0 |
| Robot | 98.60% | batch=16, lr=1e-3, wd=0 |

### SIFT + BoW + SVM/KNN Performance

| Domain | Best Test Accuracy | Configuration |
|--------|-------------------|---------------|
| Human | 56.03% | vocab=100, SVM, C=1.0 |
| Robot | 55.74% | vocab=100, SVM, C=1.0 |

### Key Findings

- **CNN significantly outperforms traditional CV methods** on this dataset (~40% higher accuracy)
- **Transfer learning with ImageNet weights** provides excellent initialization for robot vision tasks
- **SVM generally outperforms KNN** in the BoW pipeline
- **Larger vocabulary sizes** (100 vs 50) improve BoW performance
- Both approaches perform similarly across human and robot domains

## Requirements

### Core Dependencies

```
torch
torchvision
numpy
scikit-learn
opencv-python
matplotlib
pandas
```

### Hardware Support

The CNN code automatically detects and uses:
- Apple Silicon MPS (Metal Performance Shaders)
- CUDA (NVIDIA GPUs)
- CPU fallback

## Usage

### Running the CNN Experiments

```bash
cd code
python cnn_resnet18.py
```

This will:
1. Train ResNet-18 models with different hyperparameter combinations
2. Generate training curves (loss/accuracy plots)
3. Save confusion matrices for each run
4. Save misclassified examples for error analysis
5. Export results to `cnn_results_icub.csv`

### Running the Traditional CV Experiments

```bash
cd code
python traditional_CV.py
```

This will:
1. Extract SIFT descriptors from all images
2. Build visual vocabularies using K-means clustering
3. Train SVM and KNN classifiers on BoW features
4. Save confusion matrices and error cases
5. Export results to `bow_sift_results_icub.csv`

### Generating Result Visualizations

```bash
cd code
python table.py
```

## Dataset

This project uses the **iCubWorld 1.0** dataset, which contains images of everyday objects captured by:
- Human operators (human domain)
- iCub humanoid robot (robot domain)

The dataset should be organized as:
```
DATA_ROOT/
├── human/
│   ├── train/
│   │   ├── class1/
│   │   ├── class2/
│   │   └── ...
│   └── test/
│       ├── class1/
│       ├── class2/
│       └── ...
└── robot/
    ├── train/
    └── test/
```

**Note**: Update the `DATA_ROOT` path in both Python scripts to point to your local dataset location.

## Output Files

### CSV Results
- `cnn_results_icub.csv`: Contains domain, run_id, hyperparameters, validation accuracy, and test accuracy for all CNN experiments
- `bow_sift_results_icub.csv`: Contains domain, run_id, vocabulary size, classifier type, parameters, and accuracies for all BoW experiments

### Figures
- **Training curves**: `{domain}_run{id}_loss.png`, `{domain}_run{id}_acc.png`
- **Confusion matrices**: `{domain}_run{id}_cm.png`
- **Error analysis**: `{domain}_run{id}_errors.png`
- **Hyperparameter analysis**: `{domain}_cnn_test_acc_per_run.png`, `{domain}_lr_vs_testacc_wd{wd}.png`

## Report

For detailed methodology, analysis, and discussion, see **`LangCheng_report.pdf`**.

## License

This project is for educational and research purposes.

## Acknowledgments

- iCubWorld dataset creators
- PyTorch and scikit-learn communities