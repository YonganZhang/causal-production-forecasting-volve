"""统一的路径解析。别人下载发布包后，靠这里把绝对路径换成他自己的。

原来 9 个脚本里散着 17 处 `/mnt/data/yongan-admin-2/...` 硬编码。本机跑没问题，
但发布包到别人机器上，`norne_bulk` 在 **import 时**就会去 read_text 那个 deck，
直接 FileNotFoundError —— 包等于打不开。

解析顺序（每项都可单独用环境变量覆盖）：
    1. 环境变量
    2. 发布包内的相对路径（data/ 与 vendor/ 就在包里）
    3. 本机开发时的默认绝对路径

deck 找不到时**不在 import 期崩**：注入控制变量清单会退回读包内
`_meta/norne_controls.json`（打包时冻结）。只有真要构造 deck 跑模拟时才要求 deck 存在。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_DEFAULTS = {
    "NORNE_DECK": "/mnt/data/yongan-admin-2/datasets/petro/opm-data/norne",
    "NORNE_BULK": "/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne",
    "NORNE_BULK_V2": "/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_v2",
    "NORNE_OBS_V2": "/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_obs_v2",
    "NORNE_GRID_CACHE": "/mnt/data/yongan-admin-2/datasets/petro/_bulk/norne_grid_cache.npz",
}
# 发布包里的位置(相对包根)。存在就优先于本机默认值。
_IN_PACKAGE = {
    "NORNE_DECK": ROOT / "vendor" / "norne",
    "NORNE_BULK": ROOT / "data",
    "NORNE_GRID_CACHE": ROOT / "data" / "norne_grid_cache.npz",
}


def resolve(key: str) -> Path:
    if (env := os.environ.get(key)):
        return Path(env).expanduser()
    if (p := _IN_PACKAGE.get(key)) is not None and p.exists():
        return p
    return Path(_DEFAULTS[key])


NORNE_DECK = resolve("NORNE_DECK")
NORNE_BULK = resolve("NORNE_BULK")
NORNE_BULK_V2 = resolve("NORNE_BULK_V2")
NORNE_OBS_V2 = resolve("NORNE_OBS_V2")
NORNE_GRID_CACHE = resolve("NORNE_GRID_CACHE")

CONTROLS_JSON = ROOT / "_meta" / "norne_controls.json"


def require(p: Path, what: str, how: str) -> Path:
    """真要用某个资源时才检查存在性，并给出可操作的提示。"""
    if not p.exists():
        raise FileNotFoundError(
            f"{what} 不存在: {p}\n"
            f"  设环境变量指到你自己的路径，或按下面获取:\n  {how}")
    return p


DECK_HOWTO = ("git clone https://github.com/OPM/opm-data.git && "
              "export NORNE_DECK=$PWD/opm-data/norne")
