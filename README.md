# Handwriting Writer Identification using Deep Learning

A deep learning project for **offline writer identification** from handwriting images using a custom Convolutional Neural Network (CNN) and transfer learning with InceptionV3.

Unlike handwriting recognition or OCR, this project does not focus on recognizing *what* is written. Instead, the objective is to identify **who wrote a handwriting sample** based on visual writing characteristics.

## Project Overview

This academic machine learning project compares two approaches for multi-class writer identification:

- **Custom CNN** trained from scratch as a baseline
- **InceptionV3** using transfer learning

The goal was not only to train a classifier, but also to compare how a custom architecture and a pretrained visual feature extractor perform on the same handwriting identification task.

## Dataset

The dataset contains:

- **64 writers**
- **100 handwriting samples per writer**
- **6,400 handwriting images** in total

The data was prepared for multi-class classification and evaluated using different train-test splits.

> **Dataset Availability**
>
> The handwriting dataset was collected and provided by the course instructor for academic use. Due to distribution restrictions, the original dataset and derived data files are not included in this repository.

## Experimental Workflow

The project follows this general workflow:

1. Audit and prepare the handwriting image dataset
2. Build class labels for the 64 writers
3. Create train-test dataset splits
4. Preprocess images for model training
5. Train a custom CNN baseline
6. Apply transfer learning with InceptionV3
7. Evaluate both approaches on held-out test data
8. Compare their classification performance

## Approaches

### Custom CNN

A convolutional neural network was developed and trained from scratch to learn writer-specific visual features directly from the handwriting images. This model served as the baseline for comparison.

### InceptionV3

Transfer learning with InceptionV3 was explored to test whether pretrained visual representations could improve writer classification performance compared with the custom CNN baseline.

## Results

| Model | Reported Accuracy |
| --- | ---: |
| Custom CNN | ~55.9% |
| InceptionV3 | ~66.9% |

The InceptionV3-based approach achieved approximately **11 percentage points higher reported accuracy** than the custom CNN in the documented experiment.

### Key Finding

The comparison suggests that pretrained visual representations transferred more effectively to this writer-identification task than the baseline CNN trained from scratch under the experimental configuration used in the notebook.

These results describe performance on writers represented in the dataset and should not be interpreted as identification performance for previously unseen writers.

## Technologies

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Pillow
- Google Colab

## Setup

Install the core Python dependencies with:

```bash
pip install -r requirements.txt
```

The full experiment still requires authorized access to the private handwriting dataset and the corresponding local/Google Drive paths used by the notebook.

## What I Learned

Through this project, I explored:

- Image preprocessing for deep learning
- Multi-class image classification
- Designing and training convolutional neural networks
- Transfer learning with InceptionV3
- Dataset splitting and experimental comparison
- Model evaluation and interpretation
- Comparing a custom architecture with a pretrained model
- Communicating experimental results without overgeneralizing model performance

## Repository Contents

- `writer_identification_cnn_inceptionv3.ipynb` — dataset preparation, model training, evaluation, and comparison
- `requirements.txt` — core Python dependencies used by the project
- `.gitignore` — repository exclusions for private data, generated outputs, and local files

## Reproducibility Notes

The notebook contains placeholder paths for private dataset locations. Because the original handwriting dataset is not publicly distributable, the repository cannot reproduce the full experiment without authorized access to the source data.

The dependency file is intentionally kept lightweight and unpinned because the original Google Colab environment versions were not preserved as part of the experiment. The notebook and reported outputs are retained to document the experimental process and model comparison.

## Project Context

This project was developed as part of an academic machine learning project focused on applying deep learning to image-based classification and comparing a custom CNN with transfer learning.
