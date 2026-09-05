# 归档:A2 / E2 批次(被 F2 取代)

归档日期 2026-09-06。**这两批不是"跑得差的那几次",而是配置有已定位缺陷的版本。**

| 批次 | 配置 | 为什么被取代 |
|---|---|---|
| A2 | 对等化前 | 单 Agent 臂恰好保留了唯一的增产派角色,而团队臂是 1 增产 vs 6 减注 —— 等于给对照组发 buff。闭环 7 次里 6 次越跑越差(净 −139.6M)。 |
| E2 | 证据强度加权前 | 零本油藏实测数据的文献类角色与有实测数据的角色**等权**,主收益轴(tilt)被系统性压平:团队 +0.570 vs 单 Agent +0.865,而 tilt 与 ΔNPV 相关 +0.714。 |
| F2 | 现行 | ① 总工按证据强度裁决;② 文献类角色自限;③ 意见按证据强度排序。 |

E2 的结果(n=8/臂):Team +352.5M、单 Agent +320.9M、Team−单 p=0.329。
与 F2 结论方向一致(有 Agent ≫ 无 Agent;Team 与单 Agent 未检出差异),
故取代 A2/E2 **不改变任何结论**,只是去掉了已知有缺陷配置产生的噪音。

## 恢复

    mv _legacy/2026-09-06-batches-A2-E2-superseded/loop/*.json _pipelines/fc_team/
    mv _legacy/2026-09-06-batches-A2-E2-superseded/sim/*.npz  _pipelines/fc_decide/sim/

正式批次常量在 `_code/gaia.py:BATCH`。

## 同批归档:更早的三臂迭代

`B2 D2 S T U V W X Y` 前缀的 loop/sim 同属三臂对比的早期迭代,一并归档。
其中 **W / X / Y 三批已因数据泄露作废** —— 提示词里带过最优 θ 的实例表,
等于把答案发给了智能体(见 `_wiki-methodology/_top/_findings/2026-09-04-实例表数据泄露.md`)。
其余为 harness 缺陷修复前的版本。

**未归档、仍在用的**:`Rno* / A_full / B_noconn … H_nocons / N1-N4` 属于
**单角色消融**,是另一个实验,不是三臂对比的旧版本。
