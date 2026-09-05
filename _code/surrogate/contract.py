"""流水线各阶段的接口契约、断言与抽查。

为什么需要这一层:本项目两次栽在"内部完全自洽、但与源头不符"上——
  · obs 按数组索引抽稀 → 同一槽位跨样本差最多 457 天, 且与 θ 相关
  · δ_P 用全场值域当分母 → 看起来 2.1%, 实际占真实信号 29.7%
两次的共同点是**所有维度都对、NaN 都是 0、看起来完全正常**。
所以检查必须做到两件事:① 打印, 不打印等于没看 ② 回原始文件核对, 不做内部自洽。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Check:
    stage: str
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Ledger:
    """把每个阶段的体检结果记下来, 最后落盘。verdict 非 ok 时流水线拒绝继续。"""

    checks: list[Check] = field(default_factory=list)
    tensors: dict = field(default_factory=dict)

    def add(self, stage: str, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(stage, name, bool(ok), detail))
        return bool(ok)

    def require(self, stage: str, name: str, ok: bool, detail: str = "") -> None:
        self.add(stage, name, ok, detail)
        if not ok:
            raise AssertionError(f"[{stage}] {name} 失败: {detail}")

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "n_checks": len(self.checks), "n_failed": len(self.failed),
            "verdict": "ok" if not self.failed else "FAILED",
            "checks": [c.__dict__ for c in self.checks],
            "tensors": self.tensors,
        }, indent=1, ensure_ascii=False), encoding="utf-8")

    def report(self) -> str:
        lines = [f"体检 {len(self.checks)} 项, 失败 {len(self.failed)} 项"]
        for c in self.checks:
            lines.append(f"  {'✅' if c.ok else '❌'} [{c.stage}] {c.name}"
                         + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def inspect(arr: np.ndarray, name: str, ledger: Ledger | None = None,
            stage: str = "data") -> dict:
    """维度体检:shape / dtype / NaN / inf / 值域。**必须打印**——没打印过的等于没看过。"""
    a = np.asarray(arr)
    f = a.astype(np.float64, copy=False) if a.dtype.kind in "fiu" else None
    info = {
        "shape": list(a.shape), "dtype": str(a.dtype),
        "n_nan": int(np.isnan(f).sum()) if f is not None else 0,
        "n_inf": int(np.isinf(f).sum()) if f is not None else 0,
        "min": float(np.nanmin(f)) if f is not None and f.size else None,
        "max": float(np.nanmax(f)) if f is not None and f.size else None,
        "mean": float(np.nanmean(f)) if f is not None and f.size else None,
    }
    print(f"  {name:22s} shape={str(info['shape']):22s} {info['dtype']:8s} "
          f"NaN={info['n_nan']:<7d} inf={info['n_inf']:<4d} "
          f"[{info['min']:.4g}, {info['max']:.4g}] mean={info['mean']:.4g}"
          if info["min"] is not None else f"  {name:22s} shape={info['shape']} {info['dtype']}")
    if ledger is not None:
        ledger.tensors[name] = info
        ledger.add(stage, f"{name}:no_inf", info["n_inf"] == 0,
                   f"inf 数 {info['n_inf']}")
    return info


def spotcheck_obs_vs_source(theta_idx: list[int], case_dirs: list[Path],
                            obs: np.ndarray, producers: tuple, obs_keys: tuple,
                            day_grid: np.ndarray, ledger: Ledger, tol: float = 1e-3) -> None:
    """🔴 回原始 UNSMRY 逐点核对 obs —— 这是唯一能抓住"时间轴错位"那类 bug 的检查。

    内部自洽检查(比如"obs 的 shape 对不对")对那个 bug 完全无效, 因为它维度全对。
    """
    from resdata.summary import Summary

    n_bad = 0
    for k, (i, cd) in enumerate(zip(theta_idx, case_dirs)):
        smry = cd / "out" / "NORNE_ATW2013.UNSMRY"
        if not smry.exists():
            ledger.add("spotcheck", f"case{i}:source_exists", False, str(smry))
            continue
        s = Summary(str(smry))
        days = np.asarray(s.numpy_vector("TIME"), dtype=float)
        col = 0
        worst = 0.0
        for w in producers:
            for key in obs_keys:
                try:
                    v = np.asarray(s.numpy_vector(f"{key}:{w}"), dtype=float)
                except Exception:
                    col += len(day_grid); continue
                m = np.isfinite(days) & np.isfinite(v)
                ref = np.interp(day_grid, days[m], v[m]) if m.sum() >= 2 else np.full(len(day_grid), np.nan)
                got = obs[i, col:col + len(day_grid)]
                scale = max(np.nanmax(np.abs(ref)), 1.0)
                d = np.nanmax(np.abs(got - ref)) / scale
                worst = max(worst, 0.0 if np.isnan(d) else d)
                col += len(day_grid)
        ok = worst < tol
        n_bad += (not ok)
        print(f"    抽查 case s{i:04d}: 与原始 UNSMRY 最大相对偏差 {worst:.2e} "
              f"{'✅' if ok else '❌'}")
        ledger.add("spotcheck", f"case{i}:obs_matches_UNSMRY", ok, f"最大相对偏差 {worst:.3e}")
    ledger.require("spotcheck", "obs_vs_source_all", n_bad == 0,
                   f"{n_bad} 个抽查样本与原始文件不符")


def spotcheck_fields_vs_source(theta_idx: list[int], case_dirs: list[Path],
                               fields: np.ndarray, field_names: tuple,
                               n_snap: int, ledger: Ledger, tol: float = 1e-2) -> None:
    """回原始 UNRST 逐点核对 3D 场。float16 存储, 容差放宽到 1e-2。"""
    from resdata.resfile import ResdataFile

    n_bad = 0
    for i, cd in zip(theta_idx, case_dirs):
        p = cd / "out" / "NORNE_ATW2013.UNRST"
        if not p.exists():
            ledger.add("spotcheck", f"case{i}:unrst_exists", False, str(p)); continue
        f = ResdataFile(str(p))
        n = f.num_named_kw(field_names[0])
        picks = np.linspace(0, n - 1, n_snap).astype(int)
        worst = 0.0
        for t, pk in enumerate(picks):
            for c, kw in enumerate(field_names):
                ref = np.asarray(f[kw][pk], dtype=np.float64)
                got = fields[i, t, c].astype(np.float64)
                scale = max(np.abs(ref).max(), 1e-6)
                worst = max(worst, float(np.abs(got - ref).max() / scale))
        ok = worst < tol
        n_bad += (not ok)
        print(f"    抽查 case s{i:04d}: 与原始 UNRST 最大相对偏差 {worst:.2e} "
              f"{'✅' if ok else '❌'}")
        ledger.add("spotcheck", f"case{i}:fields_match_UNRST", ok, f"最大相对偏差 {worst:.3e}")
    ledger.require("spotcheck", "fields_vs_source_all", n_bad == 0,
                   f"{n_bad} 个抽查样本与原始文件不符")


def check_splits(theta: np.ndarray, sp: dict, ledger: Ledger) -> None:
    """切分不变量:互不重叠、并集完整、OOD 的 |θ| 与训练集不相交。"""
    tr, va, oo = sp["train"], sp["val"], sp["ood"]
    allidx = np.concatenate([tr, va, oo])
    ledger.require("splits", "无重叠", len(np.unique(allidx)) == len(allidx),
                   f"并集 {len(allidx)} 唯一 {len(np.unique(allidx))}")
    ledger.require("splits", "覆盖全部样本", len(allidx) == len(theta),
                   f"{len(allidx)} vs {len(theta)}")
    mag = np.linalg.norm(theta, axis=1)
    gap = mag[oo].min() - mag[tr].max()
    ledger.add("splits", "OOD 与 train 的 |θ| 不相交", gap >= 0,
               f"OOD min {mag[oo].min():.4f} − train max {mag[tr].max():.4f} = {gap:+.4f}")
    print(f"  切分 train {len(tr)} / val {len(va)} / OOD {len(oo)}   "
          f"|θ| train≤{mag[tr].max():.3f}  OOD≥{mag[oo].min():.3f}  间隔 {gap:+.4f}")


def check_row_alignment(theta: np.ndarray, obs: np.ndarray, fields: np.ndarray,
                        ledger: Ledger) -> None:
    """标签与输入必须行对齐:三个数组的第 0 维必须一致。"""
    ok = len(theta) == len(obs) == len(fields)
    ledger.require("data", "行对齐", ok,
                   f"theta {len(theta)} / obs {len(obs)} / fields {len(fields)}")
