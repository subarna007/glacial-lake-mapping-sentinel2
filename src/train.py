"""Train a U-Net (ResNet-34 encoder, ImageNet weights) to segment glacial lakes.

    python -m src.train --chips data/chips --out results/unet
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .data import band_stats, load_tile, normalise, tile_table


class Tiles(Dataset):
    def __init__(self, chip_dir, files, mean, std, augment=False):
        self.dir, self.files, self.mean, self.std, self.augment = Path(chip_dir), list(files), mean, std, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x, y = load_tile(self.dir / self.files[i])
        valid = np.isfinite(x).all(0)
        x = normalise(x, self.mean, self.std)
        y = y.astype("float32")
        if self.augment:  # lakes have no preferred orientation
            # torch's generator is seeded per worker, numpy's is copied into every worker
            k = int(torch.randint(4, (1,)))
            x, y, valid = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(valid, k)
            if torch.rand(1).item() < 0.5:
                x, y, valid = x[:, :, ::-1], y[:, ::-1], valid[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(y))[None],
                torch.from_numpy(np.ascontiguousarray(valid.astype("float32")))[None])


def build_model():
    return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=len(config.BANDS), classes=1)


def masked_loss(logits, y, valid):
    """Binary cross-entropy plus soft Dice, both computed on pixels with data only."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (y * valid).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels, valids = [], [], []
    for x, y, v in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device))).float().cpu()
        probs.append(p[:, 0].numpy())
        labels.append(y[:, 0].numpy())
        valids.append(v[:, 0].numpy().astype(bool))
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(valids)


def iou_at(probs, labels, valids, t):
    pred = probs > t
    lab = labels.astype(bool)
    tp = (pred & lab & valids).sum()
    return tp / max(((pred | lab) & valids).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="data/chips")
    ap.add_argument("--out", default="results/unet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiles = tile_table(args.chips)
    train_files = tiles[tiles.split == "train"].file.tolist()
    val_files = tiles[tiles.split == "val"].file.tolist()
    print(f"train {len(train_files)} tiles, validation {len(val_files)} tiles, device {device}")
    mean, std = band_stats(args.chips, train_files)

    train_dl = DataLoader(Tiles(args.chips, train_files, mean, std, augment=True), args.batch,
                          shuffle=True, num_workers=2, drop_last=True)
    val_dl = DataLoader(Tiles(args.chips, val_files, mean, std), args.batch, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y, v in train_dl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            opt.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(x)
            loss = masked_loss(logits.float(), y, v)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        probs, labels, valids = predict_probs(model, val_dl, device)
        val_iou = float(iou_at(probs, labels, valids, 0.5))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": val_iou})
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val IoU {val_iou:.4f}  ({time.time() - t0:.0f}s)")
        if val_iou > best:
            best = val_iou
            torch.save(model.state_dict(), out / "model.pt")

    # pick the probability threshold on validation tiles, with the best checkpoint
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    probs, labels, valids = predict_probs(model, val_dl, device)
    grid = np.round(np.arange(0.2, 0.81, 0.05), 2)
    ious = [float(iou_at(probs, labels, valids, t)) for t in grid]
    threshold = float(grid[int(np.argmax(ious))])
    best_epoch = max(history, key=lambda h: h["val_iou"])["epoch"]
    meta = {"mean": mean.tolist(), "std": std.tolist(), "threshold": threshold,
            "val_iou_at_threshold": max(ious), "best_epoch": best_epoch,
            "train_tiles": len(train_files), "val_tiles": len(val_files), "bands": config.BANDS}
    (out / "model.json").write_text(json.dumps(meta, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best epoch {best_epoch}, threshold {threshold}, val IoU {max(ious):.4f}")


if __name__ == "__main__":
    main()
