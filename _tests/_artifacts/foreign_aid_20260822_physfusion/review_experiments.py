#!/usr/bin/env python3
"""Independent, non-production experiment harness for fc_improve.py review."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "_code"))

import fc_decision as FD  # noqa: E402
import fc_improve as FI  # noqa: E402


OUT = Path(__file__).resolve().parent
DEFAULT = {"lr": 1e-3, "epochs": 300}


def cumulative_oil(y: np.ndarray) -> np.ndarray:
    return FI.cumoil(y)


def summarize(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    cp, ct = cumulative_oil(pred), cumulative_oil(true)
    err = np.abs(cp - ct) / np.maximum(np.abs(ct), 1e-9)
    r2 = 1.0 - np.square(cp - ct).sum() / np.square(ct - ct.mean()).sum()
    oil = pred[:, :, 0, :]
    material_negative = oil < -1e-3
    return {
        "relerr": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "r2": float(r2),
        "neg_frac_raw": float((oil < 0).mean()),
        "neg_frac_lt_minus_1e-3": float(material_negative.mean()),
        "negative_rate_mean_magnitude": float((-oil[oil < 0]).mean()) if (oil < 0).any() else 0.0,
    }


def prepare() -> dict[str, np.ndarray]:
    th, y, ia, fc = FD.load(0)
    n = len(th)
    idx = np.random.default_rng(0).permutation(n)
    test, train_pool = idx[: n // 5], idx[n // 5 :]
    val_order = np.random.default_rng(20260822).permutation(train_pool)
    val, train_inner = val_order[:200], val_order[200:]
    yf = y.reshape(n, -1)
    ym, ys = yf[train_pool].mean(0), yf[train_pool].std(0) + 1e-8
    pf = FI.phys_features(th)

    def norm(x: np.ndarray, fit: np.ndarray) -> np.ndarray:
        mean, std = x[fit].mean(0), x[fit].std(0) + 1e-8
        return ((x - mean) / std).astype(np.float32)

    # Exact-repeat inputs use the original 1,600-sample train statistics.
    x0 = norm(th, train_pool)
    x1 = norm(np.hstack([th, pf]), train_pool)
    # Tuning inputs are fit only on the inner training split.
    x0_inner = norm(th, train_inner)
    x1_inner = norm(np.hstack([th, pf]), train_inner)
    return {
        "th": th,
        "y": y,
        "yf": yf,
        "ym": ym,
        "ys": ys,
        "x0": x0,
        "x1": x1,
        "x0_inner": x0_inner,
        "x1_inner": x1_inner,
        "train_pool": train_pool,
        "train_inner": train_inner,
        "val": val,
        "test": test,
    }


def train_one(
    x: np.ndarray,
    data: dict[str, np.ndarray],
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    seed: int,
    lr: float,
    epochs: int,
    mono: bool = False,
) -> tuple[dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    device = train_one.device
    net = FI.TF(x.shape[1], FI.NW * FI.NPH, FI.NT).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    a = torch.tensor(x[fit_idx], device=device)
    yf = data["yf"]
    # Output normalization must be refit on the actual fit split in tuning.
    ym = yf[fit_idx].mean(0)
    ys = yf[fit_idx].std(0) + 1e-8
    b_np = ((yf[fit_idx] - ym) / ys).reshape(-1, FI.NW * FI.NPH, FI.NT).astype(np.float32)
    b = torch.tensor(b_np, device=device)
    ymt = torch.tensor(ym.reshape(FI.NW * FI.NPH, FI.NT), device=device)
    yst = torch.tensor(ys.reshape(FI.NW * FI.NPH, FI.NT), device=device)
    t0 = time.time()
    for _ in range(epochs):
        order = torch.randperm(len(a), device=device)
        for start in range(0, len(a), 64):
            batch = order[start : start + 64]
            optimizer.zero_grad()
            out = net(a[batch])
            loss = torch.nn.functional.mse_loss(out, b[batch])
            if mono:
                physical = out * yst + ymt
                oil = physical.view(-1, FI.NW, FI.NPH, FI.NT)[:, :, 0, :]
                loss = loss + 1e-3 * torch.relu(-oil).mean()
            loss.backward()
            optimizer.step()
        scheduler.step()
    net.eval()
    with torch.no_grad():
        train_mse = torch.nn.functional.mse_loss(net(a), b).item()
        raw = net(torch.tensor(x[eval_idx], device=device)).cpu().numpy().reshape(len(eval_idx), -1)
    pred = (raw * ys + ym).reshape(-1, FI.NW, FI.NPH, FI.NT)
    metrics = summarize(pred, data["y"][eval_idx])
    eval_norm = ((data["yf"][eval_idx] - ym) / ys).reshape(-1, FI.NW * FI.NPH, FI.NT)
    metrics.update(
        train_mse=float(train_mse),
        eval_mse=float(np.mean(np.square(raw.reshape(eval_norm.shape) - eval_norm))),
        seconds=round(time.time() - t0, 3),
        seed=seed,
        lr=lr,
        epochs=epochs,
        mono=mono,
        d_in=int(x.shape[1]),
    )
    return metrics, cumulative_oil(pred)


train_one.device = "cuda:6"


def run_repeat(data: dict[str, np.ndarray], seeds: list[int]) -> None:
    records: list[dict[str, float | str]] = []
    cumulative: dict[str, np.ndarray] = {"true": cumulative_oil(data["y"][data["test"]])}
    configs = [("baseline", data["x0"], False), ("phys", data["x1"], False), ("mono", data["x0"], True)]
    for name, x, mono in configs:
        rows = []
        preds = []
        for seed in seeds:
            metrics, cp = train_one(x, data, data["train_pool"], data["test"], seed=seed, mono=mono, **DEFAULT)
            metrics["config"] = name
            records.append(metrics)
            rows.append(metrics["relerr"])
            preds.append(cp)
            print(f"repeat {name:8s} seed={seed:2d} relerr={metrics['relerr']:.6%} r2={metrics['r2']:.6f} train_mse={metrics['train_mse']:.6f}", flush=True)
        cumulative[name] = np.stack(preds)
        print(f"repeat-summary {name}: {np.mean(rows):.6%} +/- {np.std(rows, ddof=1):.6%}", flush=True)
    (OUT / "repeat.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "repeat_cumoil.npz", **cumulative)


def run_tune(data: dict[str, np.ndarray], seeds: list[int]) -> None:
    grid = [
        {"lr": 3e-4, "epochs": 300},
        {"lr": 1e-3, "epochs": 300},
        {"lr": 3e-3, "epochs": 300},
        {"lr": 1e-3, "epochs": 600},
    ]
    records: list[dict[str, float | str]] = []
    selected: dict[str, dict[str, float | int]] = {}
    for name, x in [("baseline", data["x0_inner"]), ("phys", data["x1_inner"])]:
        cfg_scores = []
        for cfg in grid:
            vals = []
            for seed in seeds:
                metrics, _ = train_one(x, data, data["train_inner"], data["val"], seed=seed, mono=False, **cfg)
                metrics["config"] = name
                metrics["split"] = "validation"
                records.append(metrics)
                vals.append(metrics["relerr"])
                print(f"tune {name:8s} lr={cfg['lr']:.1e} ep={cfg['epochs']:3d} seed={seed} val={metrics['relerr']:.6%} train_mse={metrics['train_mse']:.6f}", flush=True)
            cfg_scores.append((float(np.mean(vals)), cfg))
        score, cfg = min(cfg_scores, key=lambda item: item[0])
        selected[name] = {**cfg, "mean_validation_relerr": score}
        print(f"selected {name}: {selected[name]}", flush=True)

    # Refit selected configuration on all 1,600 training cases and evaluate untouched test set.
    cumulative: dict[str, np.ndarray] = {"true": cumulative_oil(data["y"][data["test"]])}
    for name, x in [("baseline", data["x0"]), ("phys", data["x1"])]:
        preds = []
        cfg = {"lr": float(selected[name]["lr"]), "epochs": int(selected[name]["epochs"])}
        for seed in range(5):
            metrics, cp = train_one(x, data, data["train_pool"], data["test"], seed=seed, mono=False, **cfg)
            metrics["config"] = name
            metrics["split"] = "test_refit"
            records.append(metrics)
            preds.append(cp)
            print(f"refit {name:8s} seed={seed} test={metrics['relerr']:.6%} r2={metrics['r2']:.6f}", flush=True)
        cumulative[name] = np.stack(preds)
    (OUT / "tune.json").write_text(json.dumps({"selected": selected, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "tuned_cumoil.npz", **cumulative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["repeat", "tune"])
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for faithful timing/reproduction")
    train_one.device = f"cuda:{args.gpu}"
    data = prepare()
    if args.phase == "repeat":
        run_repeat(data, list(range(10)))
    else:
        run_tune(data, [0, 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
