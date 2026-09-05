# Tombstone: volve_causal_v0.1

- **归档日期**：2026-08-06
- **来源**：原 `_data/volve_causal_v0.1/`，本项目首个发布版数据集
- **归档原因**：`well_metadata.csv` 的 `role` 列有事实错误——把 F-5 标为 `producer_to_injector`（"2008-08-26 生产转注水"），实际方向相反（F-5 是 `injector_to_producer`，2016-04-20 转生产）。根因是 role 判定只看"是否同时有产油日和注水日"，未比较时间先后。详见 `_data/volve_causal_v0.2/README.md` 的「v0.2 变更」段。
- **替代物**：`_data/volve_causal_v0.2/`。`daily_production.csv` 与 `monthly_production.csv` 两版逐字节相同，只有 `well_metadata.csv` 和文档变了。
- **恢复方式**：目录原样保留，直接 `mv` 回 `_data/` 即可；或用当时的脚本版本 `git show 70f9c20:_code/build_causal_dataset.py` 重新生成。
- **对外影响**：v0.1 的 zip 已发布到 https://share.yongan.site/causal-production-volve/volve_causal_v0.1.zip ，含错误的 role 列。已由 v0.2 包替换。
