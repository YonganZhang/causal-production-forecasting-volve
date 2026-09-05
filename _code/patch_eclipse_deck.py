#!/usr/bin/env python3
"""把 Volve 官方 Eclipse deck 修成 OPM Flow 能解析的形式。

背景:官方 deck 是给商业 Eclipse 跑的。Eclipse 对若干写法宽容,OPM 严格,于是解析即失败。
本脚本只做**语法层**修补,不改任何物理数值。每处改动都写成 deck 注释,可追溯。

三类问题(均已实测):
  1. 表记录数超过维度声明。TABDIMS ntpvt=12 却给了 13 条 ROCK;EQLDIMS ntequl=12 却给了
     13 组 RSVD。实测 EQLNUM/PVTNUM 网格数组最大区号都是 12 => 第 13 组是废弃残留,截断。
  2. SCHEDULE 里手工插入的数据块漏写关键字。deck 里有注释 "-- put data manullay" 的地方,
     DATES 结束后直接跟井记录,没有 WCONHIST/WCONINJE 头,也没有终止符 /。
  3. 记录行尾挂了非注释的杂字符(如 "/ #13"),被当成新关键字。

用法:
    python patch_eclipse_deck.py <deck 目录>          # 原地修补, 自动备份 *.orig
    python patch_eclipse_deck.py <deck 目录> --check  # 只报告不修改
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

TAG = "-- [OPM patch]"
# 生产井记录 vs 注水井记录的判别(Volve 命名: P-* 生产, I-* 注水)
PROD_RE = re.compile(r"^\s*'P-[^']+'\s+'(OPEN|SHUT|STOP)'", re.I)
INJ_RE = re.compile(r"^\s*'I-[^']+'\s+'(WATER|GAS|OIL)'", re.I)
KEYWORD_RE = re.compile(r"^\s*([A-Z][A-Z0-9]{1,7})\s*(--.*)?$")


def _strip(line: str) -> str:
    return line.split("--")[0]


def patch_schedule(path: Path, check: bool) -> list[str]:
    """补上漏写的 WCONHIST / WCONINJE 关键字与记录块终止符。"""
    lines = path.read_text(errors="ignore").split("\n")
    out: list[str] = []
    notes: list[str] = []
    active: str | None = None      # 当前生效的关键字
    i = 0
    while i < len(lines):
        line = lines[i]
        body = _strip(line)
        kw = KEYWORD_RE.match(body)
        if kw:
            active = kw.group(1).upper()
            out.append(line)
            i += 1
            continue
        if active in (None, "DATES", "TSTEP") and (PROD_RE.match(body) or INJ_RE.match(body)):
            # 裸露的井记录块 —— 补头
            head = "WCONHIST" if PROD_RE.match(body) else "WCONINJE"
            notes.append(f"{path.name}:{i+1} 补 {head}")
            out.append(f"{head}   {TAG} 原 deck 此处漏写关键字")
            while i < len(lines) and (PROD_RE.match(_strip(lines[i])) or
                                      INJ_RE.match(_strip(lines[i])) or not lines[i].strip()):
                if lines[i].strip():
                    out.append(lines[i])
                i += 1
            out.append(f"/   {TAG} 补记录块终止符")
            active = None
            continue
        if body.strip() == "/":
            active = None
        out.append(line)
        i += 1
    if notes and not check:
        if not path.with_suffix(path.suffix + ".orig").exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".orig"))
        path.write_text("\n".join(out))
    return notes


def truncate_records(path: Path, keyword: str, keep: int, check: bool) -> list[str]:
    """删掉某关键字**超出 keep 条的多余记录**。

    ⚠️ 只删该关键字自己的记录块,遇到下一个关键字立刻停止。早期版本直接
    `lines[:end+1]` 把文件后面全砍了(把 VOLVE_2016.DATA 从 1037 行截到 620 行),
    这里必须严格限定作用域。
    """
    lines = path.read_text(errors="ignore").split("\n")
    try:
        start = next(i for i, l in enumerate(lines)
                     if _strip(l).strip().upper().startswith(keyword))
    except StopIteration:
        return []
    cnt, end = 0, None
    for i in range(start + 1, len(lines)):
        if "/" in _strip(lines[i]):
            cnt += 1
            if cnt == keep:
                end = i
                break
    if end is None:
        return []

    # 只在「下一个关键字出现之前」这段范围里找多余记录
    surplus: list[int] = []
    for i in range(end + 1, len(lines)):
        body = _strip(lines[i])
        if KEYWORD_RE.match(body):
            break                                   # 到下一个关键字, 停
        if body.strip():
            surplus.append(i)
    if not surplus:
        return []

    notes = [f"{path.name}: {keyword} 删除 {len(surplus)} 行多余记录"
             f"(声明 {keep} 区, 多出的是废弃残留)"]
    if not check:
        if not path.with_suffix(path.suffix + ".orig").exists():
            shutil.copy2(path, path.with_suffix(path.suffix + ".orig"))
        drop = set(surplus)
        kept = [l for i, l in enumerate(lines) if i not in drop]
        kept.insert(end + 1, f"{TAG} 已删除 {len(surplus)} 行超出 {keep} 区的残留记录")
        path.write_text("\n".join(kept))
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck_dir")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    d = Path(args.deck_dir)
    notes: list[str] = []

    main_deck = d / "VOLVE_2016.DATA"
    if main_deck.exists():
        notes += truncate_records(main_deck, "ROCK", 12, args.check)
    for rsvd in d.glob("RSVD_*"):
        if rsvd.suffix != ".orig":
            notes += truncate_records(rsvd, "RSVD", 12, args.check)
    for sch in sorted(d.glob("*.SCH")):
        notes += patch_schedule(sch, args.check)

    print(f"{'[check] 需要修补' if args.check else '已修补'} {len(notes)} 处:")
    for n in notes:
        print("  -", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
