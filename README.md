# Handwriting Writer Identification using Deep Learning
A deep learning project that explores **offline writer identification** from handwriting images using a custom Convolutional Neural Network (CNN) and transfer learning with InceptionV3.

Unlike handwriting recognition or OCR, this project does not focus on recognizing *what* is written. 
Instead, the objective is to identify **who wrote a handwriting sample** based on visual writing characteristics.

## Project Overview
The project compares two deep learning approaches:
- A custom CNN trained from scratch
- InceptionV3 using transfer learning
The models were trained and evaluated on a multi-class handwriting dataset containing samples from multiple writers.

## Dataset
The dataset used in this project contains:
- 64 writers
- 100 handwriting samples per writer
- 6,400 handwriting images in total
The data was prepared for multi-class writer identification and evaluated using different train-test splits.

> **Dataset Availability**
>
> The handwriting dataset was collected and provided by the course instructor for academic use. Due to distribution restrictions, the dataset and its derived data files are not included in this repository.

## Approaches
### Custom CNN
A convolutional neural network was developed as a baseline model to learn visual features directly from handwriting images.

### InceptionV3
Transfer learning with InceptionV3 was explored to investigate whether features learned from a large-scale image dataset could improve writer classification performance.

## Model Comparison
The experiments showed that the InceptionV3-based approach achieved stronger classification performance than the custom CNN.
| Model | Test Accuracy |
| --- | ---: |
| Custom CNN | ~55.9% |
| InceptionV3 | ~66.9% |

*Exact results correspond to the experimental configuration used in the notebook.*

## Technologies
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Matplotlib
- Pillow
- Google Colab

## What I Learned
Through this project, I explored:
- Image preprocessing for deep learning
- Multi-class image classification
- Designing and training convolutional neural networks
- Transfer learning with InceptionV3
- Dataset splitting and experimental comparison
- Model evaluation and interpretation
- Comparing a custom architecture with a pretrained model

## Repository Contents
- `writer_identification_cnn_inceptionv3.ipynb` — data preparation, model training, evaluation, and comparison
- `.gitignore` — repository exclusions

## Notes
The notebook contains placeholder paths for private dataset locations. 
The original dataset is intentionally excluded from this repository because it is not publicly distributable.

---

This project was developed as part of an academic machine learning project.
