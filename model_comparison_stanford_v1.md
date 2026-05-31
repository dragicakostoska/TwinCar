# TwinCar Model Comparison: EfficientNet-B0 v1 vs EfficientNet-B0 v2 vs ConvNeXt-Tiny

## Main comparison table

| Model | Test images | Classes | Test loss | Top-1 accuracy | Top-5 accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Make Accuracy | Make + Model Accuracy | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 v1 | 8,000 | 195 | 0.8874 | 83.08% | 95.20% | 0.8391 | 0.8306 | 0.8304 | 0.8308 | 89.84% | 83.45% | 17.33 MB |
| EfficientNet-B0 v2 | 8,000 | 195 | 1.0310 | 81.64% | 94.50% | 0.8297 | 0.8163 | 0.8176 | 0.8184 | 88.81% | 82.19% | 17.33 MB |
| ConvNeXt-Tiny | 8,000 | 195 | 0.6942 | 87.08% | 96.81% | 0.8765 | 0.8696 | 0.8697 | 0.8705 | 93.03% | 87.58% | 111.95 MB |

---

## Ranking by main metrics

### 1. Best exact make/model classification

ConvNeXt-Tiny is the best model.

It achieved:

- **87.08% top-1 accuracy**
- **96.81% top-5 accuracy**
- **0.8705 weighted F1-score**
- **93.03% make accuracy**
- **87.58% make + model accuracy**

This means ConvNeXt-Tiny is the strongest model for the main TwinCar goal: recognizing the vehicle make and model from an image.

### 2. Best lightweight baseline

EfficientNet-B0 v1 is the best lightweight baseline.

It achieved:

- **83.08% top-1 accuracy**
- **95.20% top-5 accuracy**
- **0.8308 weighted F1-score**
- **17.33 MB model size**

EfficientNet-B0 v1 is much smaller than ConvNeXt-Tiny and still performs well. It is useful as a baseline and as a lighter deployment candidate.

### 3. EfficientNet-B0 v2 result

EfficientNet-B0 v2 did **not** improve over EfficientNet-B0 v1.

Even though v2 used a larger image size, stronger dropout and differential learning rates, its result was lower:

- v1 accuracy: **83.08%**
- v2 accuracy: **81.64%**
- v1 weighted F1: **0.8308**
- v2 weighted F1: **0.8184**

This is still a useful experiment because it shows that increasing image size and training complexity does not automatically improve the model.


## Why ConvNeXt-Tiny is better

TwinCar is a fine-grained image classification problem. The model does not only need to recognize that the object is a car. It needs to distinguish between visually similar car makes and models, for example:

- Audi S4 vs Audi S5
- BMW 3 Series vs BMW M3
- Chevrolet Silverado 1500 vs Chevrolet Silverado 2500HD

ConvNeXt-Tiny has a larger and stronger visual backbone than EfficientNet-B0. This helps it learn small visual differences such as headlights, grille shape, rear lights, body shape and other details that are important for vehicle make/model recognition.


## Important limitation

These results are measured on the Stanford Cars test split. Real TwinCar production images may come from drones or ground robots and may include different angles, shadows, reflections, occlusion and parking-lot backgrounds. Therefore, the reported accuracy should be interpreted as benchmark performance on the selected dataset, not guaranteed real-world production performance.

