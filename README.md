# TwinCar - Deep Learning for Automotive Classification

Deep learning for vehicle classification and fine-grained attribute prediction. TwinCar implements and compares CNN and Vision Transformer architectures with transfer learning to classify cars, predict make/model/year, and identify automotive attributes using Stanford Cars and CompCars datasets.

---

## Quick start

```bash
git clone https://github.com/dragicakostoska/TwinCar.git
cd TwinCar
pip install -r requirements.txt
```

Run the pipeline in order:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preparation.ipynb`
- Pick a training notebook (e.g. `notebooks/03b_efficientnet_b0_v2.ipynb`)
- Run the matching evaluation notebook
- `notebooks/05_batch_prediction_demo.ipynb` for batch inference

Prerequisites: Python 3.8+, Jupyter, and a CUDA GPU (recommended).

---

## What this project does

- Classifies vehicles into fine-grained classes (Stanford Cars: 196 classes)
- Predicts make, model, and year (CompCars: multi-task annotations)
- Evaluates multiple architectures with transfer learning (fine-tuning and frozen layers)
- Provides reproducible notebooks for data prep, training, evaluation, and inference
- Outputs metrics, confusion matrices, loss curves, and inference demos

---

## Key features

- Multi-architecture comparison: EfficientNet, ConvNeXt, ViT, Swin, DeiT
- Two datasets: Stanford Cars (~16K images) and CompCars (~30K images)
- Transfer learning strategies: full fine-tuning and partial layer freezing
- Detailed evaluation: accuracy, precision, recall, F1, confusion matrices, inference time
- Reproducible pipeline: structured notebooks for the full workflow

---

## Repository structure

```text
TwinCar/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03a_efficientnet_b0_v1.ipynb
│   ├── 03a_efficientnet_b0_v1_evaluation.ipynb
│   ├── 03b_efficientnet_b0_v2.ipynb
│   ├── 03b_efficientnet_b0_v2_evaluation.ipynb
│   ├── 04_convnext.ipynb
│   ├── 04_convnext_tiny_evaluation.ipynb
│   ├── ConvNeXt_Tiny - Stanford Cars.ipynb
│   ├── ConvNeXt_Tiny with freezing - Stanford Cars.ipynb
│   ├── ConvNeXt_Tiny with freezing v2 - Stanford Cars.ipynb
│   ├── EfficientNet_B0 - Stanford Cars.ipynb
│   ├── Swin_T - Stanford Cars.ipynb
│   ├── ViT_B_16_StanfordCars_model.ipynb
│   ├── deit_tiny_patch16_224 - Stanford Cars.ipynb
│   ├── 06_compcars_efficientnet_b0_make_model_year.ipynb
│   ├── EfficientNet_B0 - Comp Cars.ipynb
│   ├── ConvNeXt_Tiny - Comp Cars.ipynb
│   ├── CompCars_ViT_model.ipynb
│   ├── 05_batch_prediction_demo.ipynb
│   └── ml-final-project.ipynb
├── models/
├── reports/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Model architectures

**CNN models**
- EfficientNet B0 — compound-scaling CNN; v1 and v2 compared
- ConvNeXt Tiny — modern CNN inspired by transformers; includes frozen-layer variants

**Transformer models**
- ViT B-16 — pure transformer for image classification
- Swin-T — hierarchical transformer with shifted windows
- DeiT (tiny_patch16_224) — data-efficient transformer with distillation

---

## Datasets

| Dataset      | Size     | Classes / Attributes                     | Notes                               |
|--------------|----------|------------------------------------------|-------------------------------------|
| Stanford Cars| ~16K imgs| 196 fine-grained car classes             | Train/val/test splits in notebooks  |
| CompCars     | ~30K imgs| Make, model, year + attribute annotations| Multi-task learning experiments     |

---

## Workflow

1. **Data exploration** — visualize images, inspect class balance, compute statistics
2. **Data preparation** — augmentations, splits, data loaders
3. **Model training** — select architecture + dataset, configure hyperparameters, train
4. **Evaluation** — metrics (accuracy, precision, recall, F1), confusion matrices, loss curves, inference time
5. **Inference** — batch predictions with confidence scores and visualization

---

## Training strategies and findings

- **Transfer learning**: full fine-tuning vs. freezing early layers
- **Findings**:
  - EfficientNet B0 v2 improves over v1
  - ConvNeXt provides strong performance with efficient computation
  - Vision Transformers deliver competitive results with different inductive biases
- Each experiment includes per-class metrics, confusion matrices, and training/validation curves

---

## Customization

- **New datasets**: update data loading in `02_data_preparation.ipynb`
- **New architectures**: add notebooks following the naming convention
- **Hyperparameters**: adjust learning rate, batch size, epochs in training notebooks
- **Task changes**: adapt for other automotive tasks or multi-task setups

---

## Future improvements

- Convert notebooks to modular Python scripts
- Implement ensemble methods
- Add augmentations: Mixup, Cutout, RandAugment
- Explore knowledge distillation
- Efficient inference: ONNX, TensorRT
- Model serving API
- Comparative performance report

---

## References

- EfficientNet: https://arxiv.org/abs/1905.11946
- ConvNeXt: https://arxiv.org/abs/2201.03545
- Vision Transformer: https://arxiv.org/abs/2010.11929
- Swin Transformer: https://arxiv.org/abs/2103.14030
- DeiT: https://arxiv.org/abs/2012.12556

---

Last updated: June 2026  
Repository: https://github.com/dragicakostoska/TwinCar
