#!/usr/bin/env python3
"""把 NORNE-CF 打包成可公开下载的发布包。

分三档，因为数据体量差三个数量级：

    code    ~20 MB    代码 + 文档 + 图 + Volve 小数据集。看懂和复现方法用这个。
    sample  ~350 MB   code + 干预臂的**固定子集**(默认 500 样本) + 校验清单。
                      足够重跑分析、复现三张图。
    full    ~3 GB     code + 干预臂全部分片。做代理模型训练才需要。

🔴 许可证边界(打包时必须遵守, 别图省事全塞进去):
  · Norne deck 来自 OPM/opm-data, **ODbL 1.0**(开放数据库许可, 带 share-alike)。
    派生数据库必须同样以 ODbL 发布并署名 —— 所以本包的模拟产物用 ODbL。
  · 本项目自己的**代码**不受数据库许可约束, 用 MIT。
  · Elsevier 全文(_refs/_fulltext)是订阅内容, **任何档位都不打包**, 也不打包 _refs 整个目录
    (里面的笔记摘自订阅全文)。
  · 不打包 _legacy(旧物)、.git、__pycache__、任何 .env/凭证。

🔴 干预臂分片在后台持续生成, 是个移动靶。本脚本会**冻结**所选样本清单并写 SHA256,
   保证别人下到的包与清单一致、可复核。

用法:
    python make_release.py --tier code
    python make_release.py --tier sample --n-samples 500
    python make_release.py --tier full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne/shards")
GRID_CACHE = Path("/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_grid_cache.npz")
STAGE = ROOT / "_tmp" / "release"

# 打进包的项目内容。_refs / _legacy / _sandbox / _tmp 一律不进 —— 见文件头许可证边界。
INCLUDE = ["_code", "_tests", "_meta", "_wiki-methodology", "_figures",
           "_pipelines", "_codemap.md", "README.md"]
INCLUDE_DATA = ["_data/volve_causal_v0.2"]
EXCLUDE_GLOB = ["__pycache__", "*.pyc", ".ipynb_checkpoints", "*.orig", ".DS_Store"]

# 发布前扫一遍, 命中就中止。宁可误报也不能把密钥发出去。
SECRET_PAT = re.compile(
    r"(api[_-]?key|insttoken|secret|passwd|password|token)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}", re.I)

MIT = """MIT License

Copyright (c) {y} NORNE-CF contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""".format(y=date.today().year)


def copy_tree(src: Path, dst: Path) -> None:
    ig = shutil.ignore_patterns(*EXCLUDE_GLOB)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=ig, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def scan_secrets(root: Path) -> list[str]:
    """遍历文本文件找疑似凭证。命中即中止发布。"""
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix in {".npy", ".npz", ".png", ".pdf", ".xlsx", ".gz"}:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:                                    # noqa: BLE001
            continue
        for m in SECRET_PAT.finditer(txt):
            hits.append(f"{p.relative_to(root)}: …{m.group(0)[:28]}…")
    return hits


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_docs(stage: Path, tier: str, shards: list[Path], n_total: int) -> None:
    (stage / "LICENSE-CODE-MIT.txt").write_text(MIT, encoding="utf-8")
    for name in ("odbl-10.txt", "dbcl-10.txt"):
        src = Path("/mnt/data/yongan-admin-2/datasets/petro/opm-data") / name
        if src.exists():
            shutil.copy2(src, stage / f"LICENSE-DATA-{name}")

    (stage / "README-RELEASE.md").write_text(f"""# NORNE-CF — 带因果真值的油藏基准（{tier} 档）

打包日期 {date.today().isoformat()}

## 这是什么

用开源油藏模拟器 OPM Flow 在 Equinor 公开的 **Norne** 油田模型上跑出的两臂数据集：

| 臂 | 参数 θ 怎么定 | 用途 |
|---|---|---|
| 干预臂 | **独立随机化** | 因果效应有**真值**，天然无混杂 |
| 观测臂 | θ = π(油藏状态) | 带混杂的观测数据 |

两臂来自同一个模拟器、同一套参数化，所以真值已知。可以用它量化朴素回归
（例如 CRM 那一派的做法）到底错多少、双重机器学习能不能救回来。

与现有因果基准（IHDP / ACIC / Twins）的区别：那些的混杂是**作者手写的
propensity 函数**，审稿人长期批评这一点。这里的混杂来自**真实偏微分方程**，
未观测混杂就是油藏状态本身。

## 包里有什么

- `_code/` 全部代码。生成器、分析、绘图、测试。
- `_tests/` 回归测试（{'含' if tier != 'code' else ''}34 项）。`pytest _tests/` 可直接跑。
- `_meta/_registry.yml` 代码与数据登记表，每个脚本的用途和运行命令。
- `_figures/` 三维效应场图（cutaway / layer / section）。
- `_wiki-methodology/` 方法论笔记与任务计划。
- `_data/volve_causal_v0.2/` Volve 日度生产数据（另一个 Equinor 公开数据集）。
{"- `data/shards/` 干预臂分片，见下面「数据格式」。" if tier != "code" else "- **本档不含模拟分片**。要数据请取 sample 或 full 档。"}

## 数据格式{'' if tier != 'code' else '（本档未包含，仅作说明）'}

每个 `.npz` 分片是一个模拟样本：

| 键 | 形状 | 含义 |
|---|---|---|
| `theta` | (13,) | 参数。前 9 维是各注入井的 log10 注入目标率乘子，后 4 维是渗透率分区乘子 |
| `obs` | (2640,) | 22 口生产井 × 3 个量(WOPR/WWPR/WBHP) × 40 个时刻，已按**天数**插值到统一网格 |
| `fields` | (8, 2, 44431) | 8 个时刻 × (PRESSURE, SWAT) × 活动单元 |

网格 46×112×22，活动单元 44,431。模拟期 1997-11-06 起约 3312 天。

### ⚠️ 本批数据的已知限制（schema v1）

必须读，否则会得出错误结论：

1. **注入相态未分离。** θ 的前 9 维按井名匹配 `WCONINJE` 的 `RATE`，**没有区分水和气**。
   实测 9 口"注水井"里有 4 口是水气混注（C-1H、C-3H、C-4AH、C-4H），
   其中 C-3H 历史末期水率为 0、气率 143344 —— 它那时其实是注气井。
   **所以对这 4 口井，θ 的语义是「总注入目标乘子」，不能解释成注水率。**
   纯注水的 5 口是 C-2H、F-1H、F-2H、F-3H、F-4H，对它们无此问题。
   代码里的 schema v2 已修（θ 扩到 17 维，按「井×相态」分开），但本批数据是 v1。
2. **θ 是目标率不是实测率。** deck 里 1463 条注入记录带 600 bar 的井底压力上限，
   顶到上限后实际注入量会被截断。v1 分片未存实测注入率（WWIR/WGIR），无法事后核查。
   v2 已加。
3. **`fields` 存为 float16。** 量化会抹平小的 SWAT 变化：实测 15,742/44,431（35%）
   个单元在全部样本上 SWAT 完全相同、斜率恒为 0。v2 改用 float32。
4. **未强制 rc==0。** v1 只要模拟写出 ≥8 个重启步就收样本，没检查模拟器返回码。
5. 逐单元效应场约 16% 的单元 |t|<2，与 0 在统计上无法区分（用多元 OLS + HC1 稳健标准误）。
   图里这些单元画成中性灰。

## 怎么复现

```bash
python -m pytest _tests/ -q                       # 回归测试
python _code/causal_bench.py field --inj C-3H     # 逐单元效应场 + 标准误
xvfb-run -a python _code/plot_reservoir_3d.py --inj C-3H   # 三视图
```

需要 numpy / scipy / scikit-learn / pyvista / resdata / matplotlib。
重跑模拟需要 Docker 与 `openporousmedia/opmreleases` 镜像（单个样本约 250 秒）。

## 许可证与署名

- **代码**：MIT，见 `LICENSE-CODE-MIT.txt`。
- **数据**：**ODbL 1.0**（`LICENSE-DATA-odbl-10.txt`），与上游 OPM/opm-data 一致。
  ODbL 带 share-alike：再分发派生数据库须同样以 ODbL 发布并署名。
- 上游 Norne 模型来自 <https://github.com/OPM/opm-data>，
  原始基准案例见 <http://www.ipt.ntnu.no/~norne/wiki/doku.php?id=start>。
- 模拟器 OPM Flow：<https://opm-project.org/>。
- **本包不含任何订阅期刊全文**。

## 诚实边界

这里的"因果"是**模拟器内部**的因果，不是真实 Norne 油田的因果。
正确表述是"我们建立了带真值的基准并量化了朴素方法的偏差"，
**不是**"我们发现了 Norne 的真实井间连通性"。
""", encoding="utf-8")

    if shards:
        man = {"tier": tier, "schema": 1, "n_shards": len(shards),
               "n_available_at_pack_time": n_total,
               "selection": "按文件名排序后等距抽样, 保证可复核",
               "packed_on": date.today().isoformat(),
               "shards": [{"name": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                          for p in shards]}
        (stage / "data" / "MANIFEST.json").write_text(
            json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["code", "sample", "full"], default="sample")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stage = STAGE / f"norne-cf-{args.tier}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for rel in INCLUDE + INCLUDE_DATA:
        src = ROOT / rel
        if src.exists():
            copy_tree(src, stage / rel)
        else:
            print(f"  跳过(不存在) {rel}")

    all_sh = sorted(SHARDS.glob("*.npz"))
    picked: list[Path] = []
    if args.tier != "code":
        if not all_sh:
            print(f"🔴 {SHARDS} 下没有分片"); return 1
        if args.tier == "full":
            picked = all_sh
        else:                                    # 等距抽样, 不是取前 N —— 前 N 会偏向早期 seed
            import numpy as np
            idx = np.unique(np.linspace(0, len(all_sh) - 1, min(args.n_samples, len(all_sh))).astype(int))
            picked = [all_sh[i] for i in idx]
        d = stage / "data" / "shards"
        d.mkdir(parents=True)
        for p in picked:
            shutil.copy2(p, d / p.name)
        print(f"  分片 {len(picked)} 个 (打包时共有 {len(all_sh)} 个)")
        # 网格角点缓存:没有它就得让下游自己去拿 EGRID 才能画图。它同样是 deck 的派生物,
        # 一并按 ODbL 发布。paths.py 会自动在包内 data/ 下找到它。
        if GRID_CACHE.exists():
            shutil.copy2(GRID_CACHE, stage / "data" / GRID_CACHE.name)
            print(f"  网格缓存 {GRID_CACHE.stat().st_size/1e6:.1f} MB")

    write_docs(stage, args.tier, picked, len(all_sh))

    hits = scan_secrets(stage)
    if hits:
        print("🔴 疑似凭证, 中止发布:")
        for h in hits[:20]:
            print("   ", h)
        return 1
    print(f"  凭证扫描通过 ({sum(1 for _ in stage.rglob('*') if _.is_file())} 个文件)")

    out = Path(args.out) if args.out else ROOT / "_outputs" / f"norne-cf-{args.tier}-{date.today().isoformat()}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        tf.add(stage, arcname=stage.name)
    size = out.stat().st_size
    print(f"\n{out}  ({size/1e6:.1f} MB)")
    print(f"SHA256  {sha256(out)}")
    (out.parent / f"{out.name}.sha256").write_text(f"{sha256(out)}  {out.name}\n", encoding="utf-8")
    shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
