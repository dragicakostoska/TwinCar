# TwinCar: Deep Learning for Automotive Classification

TwinCar is a deep learning project focused on vehicle classification and automotive attribute prediction. It explores and compares multiple state-of-the-art convolutional neural networks (CNNs) and Vision Transformer architectures using transfer learning techniques on public automotive datasets.

The project evaluates model performance across different datasets and training strategies, including fine-tuning and feature extraction, while providing a reproducible workflow for data preparation, training, evaluation, and inference.

## Features

* Comparison of multiple deep learning architectures:

  * EfficientNet-B0
  * ConvNeXt-Tiny
  * Vision Transformer (ViT-B/16)
  * Swin Transformer (Swin-T)
  * DeiT
* Support for multiple automotive datasets:

  * Stanford Cars
  * CompCars
* Transfer learning experiments with both fine-tuning and frozen-backbone approaches
* Comprehensive evaluation metrics and visualizations
* Batch inference demonstrations
* Structured and reproducible notebook-based workflow

---

## Project Structure

```text
TwinCar/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   │
│   ├── EfficientNet Experiments
│   ├── 03a_efficientnet_b0_v1.ipynb
│   ├── 03a_efficientnet_b0_v1_evaluation.ipynb
│   ├── 03b_efficientnet_b0_v2.ipynb
│   ├── 03b_efficientnet_b0_v2_evaluation.ipynb
│   │
│   ├── ConvNeXt Experiments
│   ├── 04_convnext.ipynb
│   ├── 04_convnext_tiny_evaluation.ipynb
│   │
│   ├── Stanford Cars Models
│   ├── ConvNeXt_Tiny - Stanford Cars.ipynb
│   ├── ConvNeXt_Tiny with freezing - Stanford Cars.ipynb
│   ├── ConvNeXt_Tiny with freezing v2 - Stanford Cars.ipynb
│   ├── EfficientNet_B0 - Stanford Cars.ipynb
│   ├── Swin_T - Stanford Cars.ipynb
│   ├── ViT_B_16_StanfordCars_model.ipynb
│   ├── deit_tiny_patch16_224 - Stanford Cars.ipynb
│   │
│   ├── CompCars Models
│   ├── 06_compcars_efficientnet_b0_make_model_year.ipynb
│   ├── EfficientNet_B0 - Comp Cars.ipynb
│   ├── ConvNeXt_Tiny - Comp Cars.ipynb
│   ├── CompCars_ViT_model.ipynb
│   │
│   ├── 05_batch_prediction_demo.ipynb
│   └── ml-final-project.ipynb
│
├── models/
├── reports/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Datasets

### Stanford Cars

A fine-grained vehicle classification dataset containing approximately 16,000 images across 196 vehicle categories. It is widely used for benchmarking car recognition models.

### CompCars

A large-scale automotive dataset containing over 30,000 images with detailed annotations, including vehicle make, model, and year. It is suitable for both classification and attribute prediction tasks.

---

## Model Architectures

### CNN-Based Models

#### EfficientNet-B0

EfficientNet uses compound scaling to balance network depth, width, and resolution. Multiple versions are included to evaluate the impact of training and optimization strategies.

#### ConvNeXt-Tiny

A modern CNN architecture inspired by Vision Transformers while retaining the efficiency and simplicity of convolutional networks. Experiments include both fully trainable and partially frozen variants.

### Transformer-Based Models

#### Vision Transformer (ViT-B/16)

A pure transformer architecture that processes images as sequences of patches for image classification.

#### Swin Transformer (Swin-T)

A hierarchical transformer architecture that uses shifted-window attention for efficient feature extraction.

#### DeiT

A data-efficient transformer model designed to achieve strong performance with reduced training requirements.

---

## Installation

### Prerequisites

* Python 3.8+
* CUDA-capable GPU (recommended)
* Jupyter Notebook

### Clone the Repository

```bash
git clone https://github.com/dragicakostoska/TwinCar.git
cd TwinCar
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

Core libraries used throughout the project include:

* PyTorch and TorchVision
* NumPy and Pandas
* Pillow
* Hugging Face Datasets
* Scikit-learn
* Matplotlib
* tqdm
* Jupyter Notebook and IPython Kernel

---

## Workflow

### 1. Data Exploration

`01_data_exploration.ipynb`

* Explore dataset characteristics
* Visualize class distributions
* Inspect image samples and dataset statistics

### 2. Data Preparation

`02_data_preparation.ipynb`

* Apply preprocessing and augmentation techniques
* Create training, validation, and test splits
* Build dataset loaders and transformations

### 3. Model Training

* Select the desired architecture notebook
* Configure hyperparameters
* Train using transfer learning or fine-tuning
* Monitor performance throughout training

### 4. Evaluation

* Analyze classification metrics
* Generate confusion matrices
* Visualize training and validation curves
* Compare model performance across architectures

### 5. Inference

`05_batch_prediction_demo.ipynb`

* Load trained models
* Run predictions on image batches
* Visualize outputs and confidence scores

---

## Training Strategies

The project investigates two common transfer learning approaches:

### Fine-Tuning

All network layers are trained starting from pretrained weights, allowing the model to adapt fully to the target dataset.

### Feature Extraction

Earlier layers are frozen while only the classification head is trained. This reduces training time and helps preserve pretrained feature representations.

---

## Evaluation Metrics

Each evaluation notebook provides:

* Top-1 and Top-5 Accuracy
* Precision, Recall, and F1 Score
* Per-class performance analysis
* Confusion matrices
* Training and validation loss curves
* Inference speed comparisons
* Prediction visualizations

---

## Key Observations

* EfficientNet-B0 v2 improves upon the baseline v1 configuration.
* ConvNeXt-Tiny achieves strong performance while maintaining computational efficiency.
* Transformer-based architectures provide competitive results and different representational advantages compared to CNNs.
* Transfer learning significantly reduces training requirements while maintaining strong classification accuracy.

---

## Customization

The project can be extended in several ways:

* Integrate additional automotive datasets
* Add new model architectures
* Experiment with alternative hyperparameters
* Explore multi-task learning objectives
* Implement custom data augmentation pipelines

---

## Future Work

* Convert notebook workflows into modular Python packages
* Implement model ensembling techniques
* Add advanced augmentation methods such as MixUp and RandAugment
* Explore knowledge distillation strategies
* Optimize deployment using ONNX or TensorRT
* Develop an inference API
* Create a unified benchmark report across all experiments

---

## References

* EfficientNet — *Scaling Convolutional Neural Networks Efficiently*
* ConvNeXt — *A ConvNet for the 2020s*
* Vision Transformer — *An Image is Worth 16×16 Words*
* Swin Transformer — *Hierarchical Vision Transformer Using Shifted Windows*
* DeiT — *Data-efficient Image Transformers*

---

## Contributing

Contributions, suggestions, and bug reports are welcome. Feel free to open an issue or submit a pull request.

---

## License

This project is provided for educational and research purposes.


---

**Last Updated:** June 2026
