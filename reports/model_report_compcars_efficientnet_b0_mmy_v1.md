# TwinCar CompCars Experiment Report: EfficientNet-B0 make + model + year

## Main experiment summary

| Model | Dataset | Test images | Classes | Test loss | Top-1 accuracy | Top-5 accuracy | Weighted F1 | Balanced accuracy | Make Accuracy | Make + Model Accuracy | Year Accuracy | Model size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | CompCars filtered | 9,633 | 1,637 | 1.3100 | 74.45% | 90.42% | 0.7455 | 74.46% | 86.93% | 79.64% | 79.43% | 24.72 MB |

---

## Dataset setup

This was a **CompCars-only experiment**.

The model was trained on a filtered version of the CompCars dataset using a fine-grained target:

```text
make + model + year
```

Example class label:

```text
Mercedes-Benz G Class AMG 2013
```

The filtered dataset was created from CompCars image folders and label files. Each leaf folder in the structure:

```text
image/<make_id>/<model_id>/<year>/
```

was treated as one fine-grained class.

### Dataset filtering

| Step | Result |
|---|---:|
| Initially indexed images | 69,099 |
| Raw make+model+year classes | 2,241 |
| Distinct makes | 41 |
| Distinct make+model combinations | 796 |
| Minimum images per class | 15 |
| Dropped classes | 604 |
| Dropped images | 4,883 |
| Final images | 64,216 |
| Final make+model+year classes | 1,637 |

The minimum class size filter was used because make+model+year classification creates many classes, and some year-level classes have very few images.

---

## Train / validation / test split

A reproducible stratified split was used.

| Split | Images |
|---|---:|
| Train | 44,950 |
| Validation | 9,633 |
| Test | 9,633 |

All 1,637 classes are represented in train, validation and test.

---

## Model

The experiment used **EfficientNet-B0 pretrained on ImageNet**.

| Setting | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Pretraining | ImageNet |
| Number of output classes | 1,637 |
| Trainable parameters | 6,104,545 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss with label smoothing |
| Scheduler | ReduceLROnPlateau |
| Epochs | 10 |
| Best checkpoint selection | Lowest validation loss |

EfficientNet-B0 was selected for this experiment because it is lightweight, fast enough to train on Kaggle/Colab GPU resources, and already proved useful as a baseline in the Stanford Cars experiments.

---

## Training results

The model improved throughout training and reached its best validation result in epoch 10.

| Epoch | Train Loss | Val Loss | Train Top-1 | Val Top-1 | Val Top-5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.8733 | 4.6829 | 15.1% | 24.0% | 45.0% |
| 2 | 3.0646 | 3.3841 | 58.0% | 48.2% | 72.3% |
| 3 | 2.1877 | 2.9245 | 78.3% | 58.4% | 80.5% |
| 4 | 1.8564 | 2.6369 | 86.6% | 64.3% | 85.3% |
| 5 | 1.6906 | 2.5482 | 90.8% | 66.6% | 86.5% |
| 6 | 1.5864 | 2.4262 | 93.4% | 69.9% | 87.8% |
| 7 | 1.5210 | 2.3629 | 94.9% | 70.9% | 88.7% |
| 8 | 1.4691 | 2.2838 | 95.9% | 73.4% | 89.5% |
| 9 | 1.4394 | 2.2697 | 96.3% | 73.9% | 89.5% |
| 10 | 1.4093 | 2.2455 | 96.9% | 74.8% | 89.8% |

### Interpretation

The training accuracy increased faster than the validation accuracy. This suggests that the model started to overfit, which is expected because the task is difficult and contains 1,637 fine-grained classes.

However, the validation accuracy continued improving until the final epoch, so the model was still learning useful patterns.

---

## Test set evaluation

The best checkpoint was evaluated on the held-out test set.

| Metric | Value |
|---|---:|
| Test loss | 1.3100 |
| Exact Top-1 accuracy | 74.45% |
| Exact Top-5 accuracy | 90.42% |
| Make accuracy | 86.93% |
| Make + Model accuracy | 79.64% |
| Year accuracy | 79.43% |
| Weighted F1-score | 0.7455 |
| Balanced accuracy | 74.46% |

---

## Error analysis

The model performed better at make-level classification than exact make+model+year classification.

This is expected because many CompCars classes differ only by year, body style, or small visual details.

### Top confused make+model+year pairs

| True class | Predicted class | Count |
|---|---|---:|
| Mercedes-Benz G Class AMG 2013 | Mercedes-Benz GL Class AMG 2013 | 8 |
| Ford New Focus hatchback 2012 | Ford New Focus sedan 2012 | 7 |
| Aston Martin Rapide 2013 | Aston Martin Rapide 2014 | 6 |
| Mazda Axela hatchback 2014 | Mazda Axela sedan 2014 | 6 |
| Mercedes-Benz V Class 2015 | Mercedes-Benz V Class 2014 | 6 |
| Toyota Sequoia 2010 | Toyota Sequoia 2011 | 6 |
| BMW Active Tourer 2015 | BMW 2 Series Active Tourer 2014 | 5 |
| McLaren 650S 2014 | McLaren 650S 2015 | 5 |
| Mercedes-Benz GL Class AMG 2013 | Mercedes-Benz G Class AMG 2013 | 5 |
| Mercedes-Benz CLS Class AMG 2015 | Mercedes-Benz CLS Class AMG 2014 | 5 |

Many of the most common mistakes are reasonable mistakes because the confused classes are visually similar or differ mainly by year.

---

## Comparison with Stanford Cars results

The CompCars experiment is not directly comparable to the Stanford Cars experiments because the datasets and class definitions are different.

| Experiment | Dataset | Model | Test images | Classes | Top-1 | Top-5 | Make Accuracy | Make + Model Accuracy |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Stanford final model | Stanford Cars | ConvNeXt-Tiny | 8,000 | 195 | 87.08% | 96.81% | 93.03% | 87.58% |
| CompCars experiment | CompCars filtered | EfficientNet-B0 | 9,633 | 1,637 | 74.45% | 90.42% | 86.93% | 79.64% |

The CompCars task is harder because it has many more classes and fewer images per class. It also uses make+model+year as the exact target, which makes the classification problem more fine-grained.

---

## Key conclusion

The CompCars EfficientNet-B0 experiment is useful because it tests the TwinCar idea on a larger and more diverse vehicle dataset.

The result is strong considering the difficulty of the task:

- 1,637 fine-grained classes
- many visually similar vehicles
- year-level classification
- relatively small number of images per class

---

## Important limitation

This experiment uses a filtered CompCars dataset and a custom train/validation/test split. The results should be interpreted as benchmark performance on that filtered setup, not as guaranteed performance on real TwinCar drone or robot images.

Real TwinCar production images may include different camera angles, shadows, reflections, occlusion, and parking-lot backgrounds.
