"""Real HOG + linear SVM baseline for the three cargo classes."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import joblib

CLASSES = ("barrel", "box", "sack")


def image_paths(split_dir: Path):
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = split_dir / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Не найдена папка класса: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(p)
                labels.append(label)
    return paths, np.asarray(labels, dtype=np.int64)


def hog_descriptor(path: Path, size: int = 96, cell: int = 8, bins: int = 9):
    """Compact unsigned-gradient HOG: 8x8 cells, 2x2-cell L2-Hys blocks."""
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    # Central differences with edge replication.
    gx = np.empty_like(gray); gy = np.empty_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gx[:, 0] = gray[:, 1] - gray[:, 0]; gx[:, -1] = gray[:, -1] - gray[:, -2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gy[0, :] = gray[1, :] - gray[0, :]; gy[-1, :] = gray[-1, :] - gray[-2, :]
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0) * (bins / 180.0)
    low = np.floor(angle).astype(np.int32) % bins
    frac = angle - np.floor(angle)
    h, w = gray.shape
    cy = np.arange(h)[:, None] // cell
    cx = np.arange(w)[None, :] // cell
    cell_index = np.broadcast_to(cy * (w // cell) + cx, gray.shape).ravel()
    hist = np.zeros(((h // cell) * (w // cell), bins), dtype=np.float32)
    np.add.at(hist.ravel(), cell_index * bins + low.ravel(), (magnitude * (1.0 - frac)).ravel())
    high = (low + 1) % bins
    np.add.at(hist.ravel(), cell_index * bins + high.ravel(), (magnitude * frac).ravel())
    hist = hist.reshape(h // cell, w // cell, bins)
    blocks = []
    eps = 1e-5
    for y in range(hist.shape[0] - 1):
        for x in range(hist.shape[1] - 1):
            block = hist[y:y+2, x:x+2].reshape(-1)
            block = block / np.sqrt(np.dot(block, block) + eps * eps)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.dot(block, block) + eps * eps)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def features(paths, cache: Path, size=64):
    cache.parent.mkdir(parents=True, exist_ok=True)
    key = str(size) + "\n" + "\n".join(f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths)
    import hashlib
    digest = hashlib.sha256(key.encode()).hexdigest()
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        if str(saved["key"].item()) == digest:
            return saved["X"]
    X = np.vstack([hog_descriptor(p, size=size) for p in paths])
    np.savez_compressed(cache, X=X, key=np.array(digest))
    return X


def run(dataset_root: str | Path, output_dir: str | Path, size: int = 64, c_value: float = 1.0, max_train_per_class: int = 2000):
    root, out = Path(dataset_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: image_paths(root / s) for s in ("train", "valid", "test")}
    train_paths_all, y_train_all = splits["train"]
    rng = np.random.default_rng(42)
    selected = []
    for label in range(len(CLASSES)):
        group = np.flatnonzero(y_train_all == label)
        if max_train_per_class and len(group) > max_train_per_class:
            group = np.sort(rng.choice(group, size=max_train_per_class, replace=False))
        selected.extend(group.tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    train_paths = [train_paths_all[i] for i in selected]
    y_train = y_train_all[selected]
    t0 = time.perf_counter()
    x_train = features(train_paths, out / "cache_train.npz", size)
    x_valid = features(splits["valid"][0], out / "cache_valid.npz", size)
    y_valid = splits["valid"][1]
    x_test = features(splits["test"][0], out / "cache_test.npz", size)
    y_test = splits["test"][1]

    # Select one baseline C on validation; test is kept for final evaluation.
    best = None
    for c in (0.1, 1.0, 10.0):
        model = make_pipeline(StandardScaler(), LinearSVC(C=c, class_weight="balanced", max_iter=5000, tol=1e-3, random_state=42))
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        score = f1_score(y_valid, pred, average="macro", zero_division=0)
        if best is None or score > best[0]:
            best = (score, c)

    # Keep the final baseline fit on train only; validation selects C and test remains untouched.
    final_model = make_pipeline(StandardScaler(), LinearSVC(C=best[1], class_weight="balanced", max_iter=5000, tol=1e-3, random_state=42))
    final_model.fit(x_train, y_train)
    pred = final_model.predict(x_test)
    cm = confusion_matrix(y_test, pred, labels=np.arange(len(CLASSES)))
    metrics = {
        "model": "HOG + StandardScaler + LinearSVC",
        "descriptor": {"image_size": [size, size], "cell_size": 8, "orientation_bins": 9, "block": "2x2 cells, L2-Hys"},
        "selected_C_by_validation_macro_f1": best[1],
        "validation_macro_f1_at_selected_C": best[0],
        "train_images": len(y_train), "train_images_available": len(y_train_all), "train_images_used": len(y_train),
        "max_train_per_class": max_train_per_class,
        "train_class_counts_used": {CLASSES[i]: int((y_train == i).sum()) for i in range(len(CLASSES))},
        "valid_images": len(y_valid), "test_images": len(y_test),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_macro_precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "class_names": list(CLASSES),
        "test_confusion_matrix": cm.tolist(),
        "test_support": {CLASSES[i]: int((y_test == i).sum()) for i in range(len(CLASSES))},
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "test_classification_report": classification_report(y_test, pred, target_names=CLASSES, output_dict=True, zero_division=0),
    }
    (out / "hog_svm_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(final_model, out / "hog_svm_model.joblib")
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Предсказанный класс", ylabel="Истинный класс",
           title="Матрица ошибок: HOG + SVM (test)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > threshold else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    _make_evidence_figure(metrics, cm, out / "week2_evidence.png")
    return metrics


def _make_evidence_figure(metrics, cm, destination):
    fig = plt.figure(figsize=(12, 7), dpi=160)
    gs = fig.add_gridspec(2, 2, height_ratios=(0.55, 1.45), hspace=.4, wspace=.35)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(.5,.72,"Фактический результат базовой модели — неделя 2",ha="center",va="center",fontsize=17,weight="bold")
    line=(f"HOG + Linear SVM   |   test accuracy: {metrics['test_accuracy']:.3f}   |   "
          f"macro precision: {metrics['test_macro_precision']:.3f}   |   "
          f"macro recall: {metrics['test_macro_recall']:.3f}   |   macro F1: {metrics['test_macro_f1']:.3f}")
    ax.text(.5,.25,line,ha="center",va="center",fontsize=12)
    ax2=fig.add_subplot(gs[1,0]); im=ax2.imshow(cm,cmap="Blues")
    ax2.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,
            xlabel="Предсказанный класс",ylabel="Истинный класс",title="Матрица ошибок на test")
    for i in range(3):
        for j in range(3): ax2.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax2,shrink=.78)
    ax3=fig.add_subplot(gs[1,1]); ax3.axis("off")
    rows=[]
    for name in CLASSES:
        row=metrics["test_classification_report"][name]
        rows.append([name,f"{row['precision']:.3f}",f"{row['recall']:.3f}",f"{row['f1-score']:.3f}",str(int(row['support']))])
    table=ax3.table(cellText=rows,colLabels=["Класс","Precision","Recall","F1","N test"],loc="center",cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1,1.8)
    ax3.set_title("Метрики по классам",pad=18)
    fig.savefig(destination,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prepared_cargo_cnn/classification_images")
    p.add_argument("--output", default="outputs")
    p.add_argument("--size", type=int, default=64)
    args = p.parse_args()
    result = run(args.data, args.output, args.size)
    print(json.dumps({k: result[k] for k in ("selected_C_by_validation_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1", "elapsed_seconds")}, indent=2, ensure_ascii=False))
