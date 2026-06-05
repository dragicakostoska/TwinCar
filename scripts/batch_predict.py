"""
Batch prediction script for the TwinCar project.

This script loads a trained PyTorch image-classification model and predicts
vehicle make/model/year labels for all images in a folder.

Example usage in Colab:

python scripts/batch_predict.py \
  --image_dir /content/drive/MyDrive/twincar/sample_images \
  --model_path /content/drive/MyDrive/twincar/models/convnext_tiny/best_model.pt \
  --label_map /content/drive/MyDrive/twincar/models/convnext_tiny/idx_to_class.json \
  --config /content/drive/MyDrive/twincar/models/convnext_tiny/train_config.json \
  --output /content/drive/MyDrive/twincar/predictions.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Makes from Stanford Cars. This helps us split labels such as
# "BMW_X5_SUV_2007" into make="BMW", model="X5 SUV", year="2007".
KNOWN_MAKES = [
    "AM General", "Aston Martin", "Land Rover", "Mercedes-Benz", "Rolls-Royce",
    "Acura", "Audi", "BMW", "Bentley", "Bugatti", "Buick", "Cadillac",
    "Chevrolet", "Chrysler", "Daewoo", "Dodge", "Eagle", "FIAT", "Ferrari",
    "Fisker", "Ford", "GMC", "Geo", "HUMMER", "Honda", "Hyundai", "Infiniti",
    "Isuzu", "Jaguar", "Jeep", "Lamborghini", "Lincoln", "MINI", "Maybach",
    "Mazda", "McLaren", "Mitsubishi", "Nissan", "Plymouth", "Porsche", "Ram",
    "Scion", "Spyker", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo", "smart",
]
KNOWN_MAKES = sorted(KNOWN_MAKES, key=len, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TwinCar batch prediction on a folder of vehicle images."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Folder containing images to predict."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained .pt model file, usually best_model.pt."
    )
    parser.add_argument(
        "--label_map",
        type=str,
        required=True,
        help="Path to idx_to_class.json saved during training."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to train_config.json. Used for img_size and normalization."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Where to save the prediction CSV file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="convnext_tiny",
        choices=["convnext_tiny", "efficientnet_b0"],
        help="Model architecture used during training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images processed at once. Lower this if you run out of memory."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to save for each image."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for images inside subfolders too."
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return load_json(path)


def load_idx_to_class(label_map_path: str) -> Dict[int, str]:
    path = Path(label_map_path)
    if not path.exists():
        raise FileNotFoundError(f"Label map not found: {path}")

    raw = load_json(path)
    idx_to_class = {int(k): str(v) for k, v in raw.items()}

    expected_indices = set(range(len(idx_to_class)))
    actual_indices = set(idx_to_class.keys())
    if actual_indices != expected_indices:
        raise ValueError(
            "Label map indices are not continuous from 0 to num_classes - 1. "
            f"Found min={min(actual_indices)}, max={max(actual_indices)}, count={len(actual_indices)}"
        )

    return idx_to_class


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def clean_state_dict_keys(state_dict: dict) -> dict:
    """Remove common prefixes that can appear after DataParallel or wrappers."""
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "", 1)
        if new_key.startswith("model."):
            new_key = new_key.replace("model.", "", 1)
        cleaned[new_key] = value
    return cleaned


def load_model(model_name: str, model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = build_model(model_name=model_name, num_classes=num_classes)

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def build_inference_transform(config: dict) -> transforms.Compose:
    img_size = int(config.get("img_size", 224))
    resize_size = int(config.get("resize_size", 256 if img_size == 224 else img_size * 1.15))
    imagenet_mean = config.get("imagenet_mean", [0.485, 0.456, 0.406])
    imagenet_std = config.get("imagenet_std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def collect_image_paths(image_dir: str, recursive: bool = False) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"image_dir must be a folder: {root}")

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) == 0:
        raise ValueError(f"No supported images found in: {root}")

    return image_paths


class ImagePredictionDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, str(image_path)


def clean_class_name(class_name: str) -> str:
    return str(class_name).replace("_", " ").strip()


def parse_vehicle_label(class_name: str) -> dict:
    """
    Convert a Stanford-style class label into make/model/year parts.

    Example:
    "BMW_X5_SUV_2007" -> make="BMW", model="X5 SUV", year="2007".
    """
    label = clean_class_name(class_name)

    year_match = re.search(r"\b(19|20)\d{2}\b$", label)
    year = year_match.group(0) if year_match else ""
    without_year = re.sub(r"\s+\b(19|20)\d{2}\b$", "", label).strip()

    make = None
    for candidate in KNOWN_MAKES:
        if without_year == candidate or without_year.startswith(candidate + " "):
            make = candidate
            break

    if make is None:
        parts = without_year.split()
        make = parts[0] if parts else "unknown"

    model_part = without_year[len(make):].strip()
    if not model_part:
        model_part = "unknown_model"

    return {
        "full_label": label,
        "make": make,
        "model": model_part,
        "make_model": f"{make} {model_part}".strip(),
        "year": year,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    idx_to_class: Dict[int, str],
    device: torch.device,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    top_k = min(top_k, len(idx_to_class))

    for images, image_paths in dataloader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

        for path, indices, probs in zip(image_paths, top_indices.cpu(), top_probs.cpu()):
            pred_idx = int(indices[0].item())
            pred_label_raw = idx_to_class[pred_idx]
            parsed = parse_vehicle_label(pred_label_raw)

            top_labels = [clean_class_name(idx_to_class[int(i.item())]) for i in indices]
            top_prob_values = [round(float(p.item()), 6) for p in probs]

            rows.append({
                "image_path": path,
                "predicted_idx": pred_idx,
                "predicted_class": parsed["full_label"],
                "predicted_make": parsed["make"],
                "predicted_model": parsed["model"],
                "predicted_make_model": parsed["make_model"],
                "predicted_year": parsed["year"],
                "confidence": round(float(probs[0].item()), 6),
                "top_k_predictions": json.dumps(top_labels, ensure_ascii=False),
                "top_k_probabilities": json.dumps(top_prob_values),
            })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    idx_to_class = load_idx_to_class(args.label_map)
    num_classes = len(idx_to_class)

    print("=" * 70)
    print("TwinCar batch prediction")
    print("=" * 70)
    print(f"Device       : {device}")
    print(f"Model name   : {args.model_name}")
    print(f"Model path   : {args.model_path}")
    print(f"Label map    : {args.label_map}")
    print(f"Classes      : {num_classes}")

    image_paths = collect_image_paths(args.image_dir, recursive=args.recursive)
    print(f"Images found : {len(image_paths)}")

    transform = build_inference_transform(config)
    dataset = ImagePredictionDataset(image_paths=image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(
        model_name=args.model_name,
        model_path=args.model_path,
        num_classes=num_classes,
        device=device,
    )

    predictions_df = predict(
        model=model,
        dataloader=dataloader,
        idx_to_class=idx_to_class,
        device=device,
        top_k=args.top_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print()
    print(predictions_df.head(min(10, len(predictions_df))).to_string(index=False))


if __name__ == "__main__":
    main()
