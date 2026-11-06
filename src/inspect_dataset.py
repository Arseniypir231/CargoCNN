"""Inspect split structure and create a report-ready data pipeline figure."""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

CLASSES = ("barrel", "box", "sack")
SPLITS = ("train", "valid", "test")


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def inspect(data_root: str | Path, output_dir: str | Path):
    root, out = Path(data_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts, broken, dimensions = {}, [], Counter()
    samples = {}
    for split in SPLITS:
        counts[split] = {}
        for cls in CLASSES:
            paths = list_images(root / split / cls)
            counts[split][cls] = len(paths)
            for p in paths:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        dimensions[f"{im.width}x{im.height}"] += 1
                        if cls not in samples and split == "train":
                            samples[cls] = (p, im.convert("RGB").copy())
                except Exception as e:
                    broken.append({"path": str(p), "error": str(e)})

    summary = {
        "dataset_root": str(root.resolve()),
        "split_counts": counts,
        "totals_by_class": {c: sum(counts[s][c] for s in SPLITS) for c in CLASSES},
        "total_images": sum(sum(counts[s].values()) for s in SPLITS),
        "verified_image_files": sum(sum(counts[s].values()) for s in SPLITS) - len(broken),
        "broken_images": broken,
        "most_common_dimensions": dimensions.most_common(8),
        "split_ratio": "approximately 70/15/15; group-based split prepared upstream",
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _pipeline_figure(counts, samples, out / "dataset_pipeline.png")
    return summary


def _pipeline_figure(counts, samples, dest):
    fig = plt.figure(figsize=(14, 8), dpi=160)
    grid = fig.add_gridspec(2, 1, height_ratios=[0.75, 1.3], hspace=0.38)
    ax = fig.add_subplot(grid[0])
    x = np.arange(len(CLASSES)); width = 0.23
    colors = ("#4472C4", "#70AD47", "#ED7D31")
    for i, split in enumerate(SPLITS):
        vals = [counts[split][c] for c in CLASSES]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=split, color=colors[i])
        ax.bar_label(rects, padding=2, fontsize=9)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Количество изображений")
    ax.set_title("Подготовленный датасет: распределение train / valid / test")
    ax.legend(ncols=3, frameon=False)
    ax.grid(axis="y", alpha=.2)

    sub = grid[1].subgridspec(4, 4, wspace=.12, hspace=.2)
    headers = ["Исходное фото", "Resize 224×224 + RGB", "Аугментация: поворот", "Аугментация: яркость/контраст"]
    for j, text in enumerate(headers):
        a = fig.add_subplot(sub[0, j]); a.axis("off"); a.text(.5,.5,text,ha="center",va="center",fontsize=10,weight="bold")
    for row, cls in enumerate(CLASSES, start=1):
        p, original = samples[cls]
        resized = original.resize((224,224), Image.Resampling.BILINEAR)
        rotated = resized.rotate(7, resample=Image.Resampling.BILINEAR, fillcolor=(120,120,120))
        bright = ImageEnhance.Contrast(ImageEnhance.Brightness(resized).enhance(1.16)).enhance(1.12)
        variants = [original, resized, rotated, bright]
        for col, im in enumerate(variants):
            a=fig.add_subplot(sub[row,col]); a.imshow(im); a.axis("off")
            if col == 0: a.set_ylabel(cls, rotation=0, labelpad=38, va="center", weight="bold")
    fig.suptitle("Предобработка данных и пример аугментации обучающего изображения", fontsize=15, weight="bold", y=.99)
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prepared_cargo_cnn/classification_images")
    p.add_argument("--output", default="outputs")
    args=p.parse_args()
    result=inspect(args.data,args.output)
    print(json.dumps({k:result[k] for k in ("total_images","verified_image_files","broken_images","totals_by_class","split_counts")},ensure_ascii=False,indent=2))
