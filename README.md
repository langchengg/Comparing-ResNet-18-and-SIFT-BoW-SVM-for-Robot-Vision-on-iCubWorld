# Comparing ResNet-18 and SIFT-BoW-SVM for Robot Vision on iCubWorld

This repository contains optimized implementations for comparing two computer vision approaches for robot object recognition on the iCubWorld dataset:

1. **Traditional Pipeline**: SIFT + Bag-of-Words + SVM
2. **Deep Learning Pipeline**: ResNet-18 CNN with frozen backbone

## Key Features

- **Performance Optimized**: All implementations include significant optimizations for speed and memory efficiency
- **Parallel Processing**: Multi-core CPU utilization for feature extraction
- **GPU Acceleration**: Mixed precision training for CNN pipeline
- **Comprehensive Evaluation**: Metrics, confusion matrices, and training curves

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Efficient data loading utilities
│   ├── sift_bow_svm.py     # Optimized SIFT+BoW+SVM pipeline
│   ├── resnet_cnn.py       # Optimized ResNet-18 CNN pipeline
│   └── utils.py            # Common utilities
├── requirements.txt        # Python dependencies
├── OPTIMIZATION_NOTES.md   # Detailed optimization documentation
└── LangCheng_report.pdf    # Research report
```

## Usage

### SIFT + BoW + SVM Pipeline

```python
from src.sift_bow_svm import SIFTBoWSVM, run_sift_bow_experiment
from src.data_loader import ICubWorldDataset, load_images_for_sift

# Load data
dataset = ICubWorldDataset('path/to/icubworld', split='train', domain='human')
images, labels, _ = load_images_for_sift(dataset)

# Train pipeline
pipeline = SIFTBoWSVM(vocab_size=100, classifier_type='svm', svm_c=1.0)
pipeline.fit(images, labels)

# Evaluate
test_images, test_labels, _ = load_images_for_sift(test_dataset)
metrics = pipeline.evaluate(test_images, test_labels)
```

### ResNet-18 CNN Pipeline

```python
from src.resnet_cnn import ResNet18Pipeline, run_resnet_experiment
from src.data_loader import create_data_loaders

# Create data loaders
loaders = create_data_loaders('path/to/icubworld', domain='human', batch_size=32)

# Train pipeline
pipeline = ResNet18Pipeline(num_classes=10, learning_rate=1e-3, freeze_backbone=True)
history = pipeline.fit(loaders['train'], loaders['val'], epochs=15)

# Evaluate
metrics = pipeline.evaluate(loaders['test'])
```

### Run Experiments

```bash
# SIFT+BoW+SVM experiments
python -m src.sift_bow_svm --data-root /path/to/icubworld --domain human

# ResNet-18 experiments
python -m src.resnet_cnn --data-root /path/to/icubworld --domain human --epochs 15
```

## Performance Optimizations

See [OPTIMIZATION_NOTES.md](OPTIMIZATION_NOTES.md) for detailed documentation of all performance improvements, including:

- **SIFT Pipeline**: Parallel extraction, Mini-Batch K-Means, FLANN matching
- **CNN Pipeline**: Mixed precision, frozen backbone, efficient data loading
- **Both**: Feature caching, lazy loading, optimized algorithms

## Results Summary

From the research report:

| Method | Human Domain | Robot Domain |
|--------|-------------|--------------|
| SIFT+BoW+SVM (best) | ~56% | ~56% |
| ResNet-18 CNN (best) | ~98% | ~99% |

## License

This project is for educational and research purposes.

## References

- iCubWorld Dataset: Pasquale et al., "Teaching iCub to recognize objects using deep Convolutional Neural Networks"
- ResNet: He et al., "Deep Residual Learning for Image Recognition"
- SIFT: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints"