"""PyTorch Dataset/DataLoader for the prepared cargo classification dataset.

Images are decoded as RGB, resized to 224 x 224 and normalized with ImageNet
channel statistics. Random crop/rotation/color changes are train-only. The
implementation uses PIL and torch directly, so torchvision is not required.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset

CLASSES = ("barrel", "box", "sack")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    tensor = torch.from_numpy(arr)
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class CargoImageDataset(Dataset):
    def __init__(self, split_dir: str | Path, train: bool = False, image_size: int = 224):
        self.root = Path(split_dir)
        self.classes = list(CLASSES)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.train = train
        self.image_size = int(image_size)
        if self.image_size < 32:
            raise ValueError("image_size должен быть не меньше 32")
        self.samples = []
        for cls in self.classes:
            class_dir = self.root / cls
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Не найдена папка класса: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[cls]))
        if not self.samples:
            raise ValueError(f"В {self.root} не найдены изображения JPG/JPEG/PNG")

    def __len__(self):
        return len(self.samples)

    def _augment(self, image: Image.Image) -> Image.Image:
        # RandomResizedCrop(scale=0.85..1.0), rotation and color jitter, train only.
        w, h = image.size
        scale = random.uniform(0.85, 1.0)
        crop_w, crop_h = max(1, int(w * scale)), max(1, int(h * scale))
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image = image.rotate(random.uniform(-8, 8), resample=Image.Resampling.BILINEAR)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return image

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.train:
            image = self._augment(image)
        else:
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return _to_tensor(image), label


def make_loaders(dataset_root: str | Path, batch_size: int = 32, workers: int = 0,
                 image_size: int = 224):
    """Create train/valid/test loaders from the existing split/class layout."""
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    root = Path(dataset_root)
    loaders = {}
    for split in ("train", "valid", "test"):
        dataset = CargoImageDataset(root / split, train=(split == "train"), image_size=image_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
