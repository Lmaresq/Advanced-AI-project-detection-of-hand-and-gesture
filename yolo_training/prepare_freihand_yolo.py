"""
Prepare the FreiHAND v2 dataset for YOLO hand detection and (optionally) train a model.

This script:
1) Extracts the FreiHAND zip (only the training split and annotations).
2) Projects 3D joints to 2D using the provided intrinsics to derive hand boxes.
3) Writes YOLO-format labels and splits train/val.
4) Launches a YOLO training run unless --skip-train is set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from ultralytics import YOLO

IMG_SIZE = (224, 224)  # FreiHAND RGB resolution (width, height)
CLASS_ID = 0  # single class: hand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FreiHAND YOLO preparation + training.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("FreiHAND_pub_v2.zip"),
        help="Path to FreiHAND_pub_v2.zip",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("freihand_raw"),
        help="Where to extract the raw dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("freihand_yolo"),
        help="Output directory for YOLO-ready dataset",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--max-samples", type=int, default=None, help="Subset for quick tests")
    parser.add_argument("--task", type=str, choices=["detect", "pose"], default="pose", help="Train detection or pose (squelette)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640, help="YOLO training image size")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt", help="Base model to fine-tune (pose recommended)")
    parser.add_argument("--device", type=str, default=None, help="Device string for YOLO (e.g., cpu, cuda:0)")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Directory where YOLO run artifacts are stored (default: out-dir/runs)",
    )
    parser.add_argument("--run-name", type=str, default="hand", help="YOLO run name")
    parser.add_argument("--hand-pt", type=Path, default=Path("hand.pt"), help="Chemin de sortie pour le poids final hand.pt")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--skip-train", action="store_true", help="Only prepare data, do not train")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recreate freihand_yolo even if dataset.yaml already exists",
    )
    return parser.parse_args()


def extract_training(zip_path: Path, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    rgb_dir = workdir / "training" / "rgb"
    if rgb_dir.exists():
        print(f"[extract] Raw data already present at {rgb_dir}")
        return workdir

    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m
            for m in zf.namelist()
            if m.startswith("training/rgb/") or m.startswith("training_")
        ]
        print(f"[extract] Extracting {len(members)} files to {workdir} (this can take a while)...")
        zf.extractall(path=workdir, members=members)

    return workdir


def project_bbox(
    xyz: List[List[float]], K: List[List[float]], scale: float, img_size: Tuple[int, int], pad_px: float = 5.0
) -> Tuple[float, float, float, float] | None:
    pts = np.asarray(xyz, dtype=np.float32) * float(scale)
    K = np.asarray(K, dtype=np.float32)

    uvw = K @ pts.T  # shape (3, 21)
    z = uvw[2]
    if np.any(z == 0):
        return None

    u = uvw[0] / z
    v = uvw[1] / z

    x1, x2 = float(u.min() - pad_px), float(u.max() + pad_px)
    y1, y2 = float(v.min() - pad_px), float(v.max() + pad_px)

    img_w, img_h = img_size
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

    cx = (x1 + x2) / (2 * img_w)
    cy = (y1 + y2) / (2 * img_h)
    w /= img_w
    h /= img_h
    return cx, cy, w, h


def project_keypoints(
    xyz: List[List[float]], K: List[List[float]], scale: float, img_size: Tuple[int, int]
) -> List[Tuple[float, float]] | None:
    pts = np.asarray(xyz, dtype=np.float32) * float(scale)
    K = np.asarray(K, dtype=np.float32)

    uvw = K @ pts.T  # shape (3, 21)
    z = uvw[2]
    if np.any(z == 0):
        return None

    u = uvw[0] / z
    v = uvw[1] / z

    img_w, img_h = img_size
    kpts = []
    for uu, vv in zip(u, v):
        x = min(max(uu, 0.0), img_w - 1)
        y = min(max(vv, 0.0), img_h - 1)
        kpts.append((x / img_w, y / img_h))
    return kpts


def load_annotations(workdir: Path) -> Tuple[List, List, List]:
    def load_json(name: str):
        with open(workdir / name, "r", encoding="utf-8") as f:
            return json.load(f)

    xyz = load_json("training_xyz.json")
    K = load_json("training_K.json")
    scales = load_json("training_scale.json")
    return xyz, K, scales


def build_yolo_labels(
    raw_dir: Path,
    out_dir: Path,
    xyz: List,
    K: List,
    scales: List,
    val_split: float,
    seed: int,
    max_samples: int | None,
    task: str,
) -> Path:
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    indices = list(range(len(xyz)))
    if max_samples:
        indices = indices[:max_samples]
        print(f"[prep] Using first {len(indices)} samples for a quick run.")

    # deterministic split
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = int(len(indices) * val_split)
    val_ids = set(indices[:val_count])

    stats = {"train": 0, "val": 0, "skipped": 0}
    for idx in indices:
        bbox = project_bbox(xyz[idx], K[idx], scales[idx], IMG_SIZE)
        if bbox is None:
            stats["skipped"] += 1
            continue

        keypoints = None
        if task == "pose":
            keypoints = project_keypoints(xyz[idx], K[idx], scales[idx], IMG_SIZE)
            if keypoints is None:
                stats["skipped"] += 1
                continue

        split = "val" if idx in val_ids else "train"
        cx, cy, w, h = bbox

        img_name = f"{idx:08d}.jpg"
        src_img = raw_dir / "training" / "rgb" / img_name
        dst_img = out_images / split / img_name
        dst_lbl = out_labels / split / img_name.replace(".jpg", ".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_img, dst_img)
        if task == "pose" and keypoints:
            flat_kpts = [coord for kp in keypoints for coord in kp]
            label_parts = [str(CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            label_parts.extend(f"{v:.6f}" for v in flat_kpts)
            dst_lbl.write_text(" ".join(label_parts) + "\n", encoding="utf-8")
        else:
            dst_lbl.write_text(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
        stats[split] += 1

    print(f"[prep] train: {stats['train']} | val: {stats['val']} | skipped: {stats['skipped']}")
    yaml_path = out_dir / "dataset.yaml"
    yaml_lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "nc: 1",
        "names: [hand]",
    ]
    if task == "pose":
        yaml_lines.append("kpt_shape: [21, 2]")
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    return yaml_path


def run_training(dataset_yaml: Path, args: argparse.Namespace, hand_out: Path) -> Path:
    project = args.project or (dataset_yaml.parent / "runs")
    project.mkdir(parents=True, exist_ok=True)
    print(f"[train] Starting YOLO training with {dataset_yaml}")
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.run_name,
        task=args.task,
    )
    weights_dir = Path(project) / args.run_name / "weights"
    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = weights_dir / "last.pt"
    if best_weights.exists():
        hand_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, hand_out)
        print(f"[train] hand.pt disponible dans {hand_out}")
    else:
        print("[train] Aucun poids best/last trouve, hand.pt non copie.")
    return best_weights


def resolve_path(base_dir: Path, user_path: Path) -> Path:
    return (base_dir / user_path).resolve() if not user_path.is_absolute() else user_path


def parse_names(raw_names) -> List[str]:
    if isinstance(raw_names, dict):
        ordered = []
        for key in sorted(raw_names.keys(), key=lambda k: int(k)):
            ordered.append(raw_names[key])
        return ordered
    if isinstance(raw_names, list):
        return raw_names
    return [str(raw_names)]


def count_images(path: Path) -> int:
    exts = ("*.jpg", "*.jpeg", "*.png")
    return sum(len(list(path.rglob(pat))) for pat in exts) if path.exists() else 0


def ensure_dataset_yaml(dataset_yaml: Path, out_dir: Path) -> None:
    if not dataset_yaml.exists():
        return

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    expected_path = str(out_dir.resolve())
    changed = False
    if data.get("path") != expected_path:
        data["path"] = expected_path
        changed = True

    if "names" not in data or data["names"] is None:
        data["names"] = ["hand"]
        changed = True

    if "nc" not in data:
        data["nc"] = len(parse_names(data["names"]))
        changed = True

    if changed:
        with open(dataset_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"[dataset] dataset.yaml mis a jour avec path={expected_path}")


def infer_dataset_task(data: Dict[str, object]) -> str:
    return "pose" if "kpt_shape" in data else "detect"


def summarize_dataset(dataset_yaml: Path) -> Dict[str, object]:
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", dataset_yaml.parent))
    train_path = Path(data.get("train", "images/train"))
    val_path = Path(data.get("val", "images/val"))
    if not train_path.is_absolute():
        train_path = root / train_path
    if not val_path.is_absolute():
        val_path = root / val_path

    names = parse_names(data.get("names", []))
    dataset_task = infer_dataset_task(data)
    kpt_shape = data.get("kpt_shape")
    train_imgs = count_images(train_path)
    val_imgs = count_images(val_path)

    print(f"[dataset] root: {root}")
    print(f"[dataset] train images: {train_imgs} | val images: {val_imgs}")
    print(f"[dataset] classes: {names}")
    print(f"[dataset] task: {dataset_task}" + (f" | kpt_shape: {kpt_shape}" if kpt_shape else ""))

    return {
        "names": names,
        "root": root,
        "train_images": train_imgs,
        "val_images": val_imgs,
        "task": dataset_task,
        "kpt_shape": kpt_shape,
    }


def print_training_plan(dataset_yaml: Path, dataset_info: Dict[str, object], args: argparse.Namespace, project: Path) -> None:
    steps = [
        f"Check du dataset deja prepare dans `{dataset_yaml}` (train {dataset_info['train_images']}, val {dataset_info['val_images']}), task={args.task}.",
        f"Chargement des poids de base YOLO `{args.model}` (doit etre un modele pose pour le squelette).",
        f"Entrainement sur `freihand_yolo` (imgsz={args.img_size}, batch={args.batch}, epochs={args.epochs}, workers={args.workers}, device={args.device or 'auto'}, task={args.task}).",
        f"Sauvegarde des artefacts dans `{project / args.run_name}` et copie du meilleur poids en `{args.hand_pt}` (hand.pt).",
        "Lecture des classes depuis le dataset puis (si dispo) depuis les poids entraines.",
    ]
    print("[plan] Etapes claires de l'entrainement YOLO sur freihand_yolo :")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step}")


def prepare_or_reuse_dataset(
    base_dir: Path, zip_path: Path, workdir: Path, out_dir: Path, args: argparse.Namespace
) -> Path:
    dataset_yaml = out_dir / "dataset.yaml"
    if dataset_yaml.exists() and not args.force_rebuild:
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        existing_task = infer_dataset_task(data)
        if existing_task == args.task:
            ensure_dataset_yaml(dataset_yaml, out_dir)
            print(f"[dataset] Dataset deja pret dans {out_dir}, utilisation directe.")
            return dataset_yaml
        else:
            print(f"[dataset] Dataset existant est {existing_task}, reconstruction pour task={args.task}.")

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path} (requis car pas de dataset pret)")

    raw_dir = extract_training(zip_path, workdir)
    xyz, K, scales = load_annotations(raw_dir)
    dataset_yaml = build_yolo_labels(
        raw_dir=raw_dir,
        out_dir=out_dir,
        xyz=xyz,
        K=K,
        scales=scales,
        val_split=args.val_split,
        seed=args.seed,
        max_samples=args.max_samples,
        task=args.task,
    )
    ensure_dataset_yaml(dataset_yaml, out_dir)
    return dataset_yaml


def report_classes(dataset_yaml: Path, dataset_info: Dict[str, object], trained_weights: Path | None) -> None:
    print(f"[classes] Noms issus du dataset: {dataset_info['names']}")
    if trained_weights is None:
        return
    if not trained_weights.exists():
        print(f"[classes] Poids entraines introuvables ({trained_weights}), saut du check modele.")
        return
    model = YOLO(str(trained_weights))
    print(f"[classes] Noms integres aux poids entraines: {model.names}")


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    zip_path = resolve_path(base_dir, args.zip_path)
    workdir = resolve_path(base_dir, args.workdir)
    out_dir = resolve_path(base_dir, args.out_dir)
    hand_out = resolve_path(base_dir, args.hand_pt)
    args.hand_pt = hand_out

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    if model_path.exists():
        args.model = str(model_path)
    else:
        print(f"[warn] Modele de base introuvable localement ({model_path}). Assurez-vous d'avoir un checkpoint {'pose' if args.task=='pose' else 'detect'} accessible sous ce nom.")

    dataset_yaml = prepare_or_reuse_dataset(base_dir, zip_path, workdir, out_dir, args)
    dataset_info = summarize_dataset(dataset_yaml)
    project_dir = Path(args.project) if args.project else (dataset_yaml.parent / "runs")
    print_training_plan(dataset_yaml, dataset_info, args, project_dir)

    if args.skip_train:
        print("[done] Jeu de donnees pret. Entrainement saute (--skip-train).")
        report_classes(dataset_yaml, dataset_info, trained_weights=None)
        return

    best_weights = run_training(dataset_yaml, args, hand_out)
    report_classes(dataset_yaml, dataset_info, trained_weights=best_weights)


if __name__ == "__main__":
    main()
