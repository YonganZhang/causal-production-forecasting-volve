#!/usr/bin/env python3
"""把 Volve 官方生产 xlsx 加工成 analysis-ready 因果产能预测数据集。

输出(写到 _data/volve_causal_v0.2/):
  daily_production.csv   每井每天一行,列名与原 xlsx 一致,附因果角色注释见 README
  monthly_production.csv 月度表
  well_metadata.csv      7 个井眼的角色、生产/注水区间、角色切换日期(天然实验锚点)

v0.2 修正: role 判定改为按首次产油/首次注水的**时间先后**定方向。v0.1 只看
"是否同时有产油日和注水日", 把 F-5 误标成 producer_to_injector; 实测 F-5 在
2008-08-26 前 360 天全部 FLOW_KIND=injection 且产量与 ON_STREAM_HRS 全为 0,
真实方向是 injector_to_producer(2016-04-20 转生产)。

数据许可: Equinor Open Data Licence(见 license.txt),再分发须署名 Equinor。
"""
import csv
import datetime as dt
from pathlib import Path

import openpyxl

DATA_ROOT = Path(__file__).resolve().parent.parent / "_data"
BASE = DATA_ROOT / "volve_causal_v0.2"
XLSX = BASE / "Volve production data.xlsx"


def dump_sheet(ws, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            w.writerow(["" if v is None else v for v in row])


def classify_role(first_oil, first_wi):
    """按首次产油/首次注水的时间先后判角色方向。

    只看"有没有"会把先注后产的井判反(v0.1 的 F-5 bug), 所以这里比较日期。
    返回 (role, role_switch_date)。
    """
    if first_oil is None and first_wi is None:
        return "no_flow", None
    if first_wi is None:
        return "producer", None
    if first_oil is None:
        return "injector", None
    if first_oil < first_wi:
        return "producer_to_injector", first_wi
    if first_wi < first_oil:
        return "injector_to_producer", first_oil
    return "dual_same_day", first_oil


def build_well_metadata(ws):
    """逐井统计角色区间;角色方向由首次产油/首次注水日期先后决定。"""
    wells = {}
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    idx = {name: header.index(name) for name in (
        "DATEPRD", "NPD_WELL_BORE_NAME", "BORE_OIL_VOL", "BORE_WI_VOL", "FLOW_KIND", "WELL_TYPE")}
    for r in ws.iter_rows(min_row=2, values_only=True):
        name = r[idx["NPD_WELL_BORE_NAME"]]
        d = r[idx["DATEPRD"]]
        if isinstance(d, dt.datetime):
            d = d.date()
        s = wells.setdefault(name, {
            "first": d, "last": d, "days": 0, "oil_days": 0, "wi_days": 0, "types": set(),
            "first_oil": None, "last_oil": None, "first_wi": None, "last_wi": None})
        s["days"] += 1
        s["first"], s["last"] = min(s["first"], d), max(s["last"], d)
        s["types"].add(str(r[idx["WELL_TYPE"]]))
        if r[idx["BORE_OIL_VOL"]] not in (None, 0, "0"):
            s["oil_days"] += 1
            s["first_oil"] = min(s["first_oil"], d) if s["first_oil"] else d
            s["last_oil"] = max(s["last_oil"], d) if s["last_oil"] else d
        if r[idx["BORE_WI_VOL"]] not in (None, 0, "0"):
            s["wi_days"] += 1
            s["first_wi"] = min(s["first_wi"], d) if s["first_wi"] else d
            s["last_wi"] = max(s["last_wi"], d) if s["last_wi"] else d
    rows = []
    for name, s in sorted(wells.items()):
        role, switch = classify_role(s["first_oil"], s["first_wi"])
        rows.append({
            "well": name, "role": role,
            "role_switch_date": switch or "",
            "record_start": s["first"], "record_end": s["last"],
            "record_days": s["days"], "oil_producing_days": s["oil_days"],
            "injection_days": s["wi_days"],
            "first_oil_date": s["first_oil"] or "",
            "last_oil_date": s["last_oil"] or "",
            "first_injection_date": s["first_wi"] or "",
            "last_injection_date": s["last_wi"] or "",
            "well_types_seen": "|".join(sorted(s["types"])),
        })
    return rows


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    daily = wb["Daily Production Data"]
    dump_sheet(daily, BASE / "daily_production.csv")
    dump_sheet(wb["Monthly Production Data"], BASE / "monthly_production.csv")
    meta = build_well_metadata(wb["Daily Production Data"])
    with open(BASE / "well_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    print(f"daily rows={daily.max_row - 1}, wells={len(meta)}")
    for m in meta:
        print(f"  {m['well']:14s} {m['role']:22s} switch={m['role_switch_date'] or '-'}"
              f" oil={m['first_oil_date'] or '-'}..{m['last_oil_date'] or '-'}"
              f" wi={m['first_injection_date'] or '-'}..{m['last_injection_date'] or '-'}")


if __name__ == "__main__":
    main()
