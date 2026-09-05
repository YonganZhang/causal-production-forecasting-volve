"""诊断:把团队一轮里的全部原文落盘，看信息到底在哪一步丢了。

主环路只在日志里截 50 字，聚合数字看不出"角色说了什么、总工听没听"。
本脚本跑一轮 Readers + Synthesizer，把每个角色的完整 JSON、
喂给总工的完整消息串、总工的完整回应都存下来。
"""
from __future__ import annotations
import json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_gen as FG                                        # noqa: E402
from fc_team import ROLES, OUT                                   # noqa: E402
import fc_team_loop as _L
from fc_team_loop import (ASK_CHIEF, ASK_CRITIQUE, ASK_READER, DYN, FIXED, INJ,
                          N_STAGE, BASE_W, PERSPECTIVE, ask, slice_for)  # noqa: E402


LAST = None


def main() -> None:
    _L.SHARED_BG = (_L.tilt_prior() + _L.domain_rules()).strip()
    model = sys.argv[1] if len(sys.argv) > 1 else "sonnet"
    global LAST
    LAST = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    carto = json.loads((OUT / "cartography.json").read_text())
    active = [r for r in ROLES if r["id"] != "chief_engineer"]
    hist = "首轮，尚无历史反馈。"

    def one(role):
        data, prov = slice_for(role["id"], carto, LAST)
        p = (FIXED.format(role=role["id"], duty=role["duty"], ns=N_STAGE, ni=len(INJ),
                          inj=INJ, basew=BASE_W.round(0).tolist(), stages=FG.STAGE_AT,
                          data=data, cinj=2.0, cprod=1.0,
                          stance=PERSPECTIVE.get(role["id"], ""), shared=_L.SHARED_BG)
             + DYN.format(r=1, hist=hist) + ASK_READER)
        t0 = time.time()
        raw, ans = ask(p, model, 300)
        return {"role": role["id"], "prompt_chars": len(p), "sec": time.time() - t0,
                "data_slice": data, "answer": ans}

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, active))

    print("=" * 100)
    print("一、每个角色的完整输出")
    print("=" * 100)
    for r in res:
        a = r["answer"]
        print(f"\n【{r['role']}】 conf={a.get('confidence')}  "
              f"提示词 {r['prompt_chars']} 字  耗时 {r['sec']:.0f}s")
        print(f"  判断  : {a.get('assessment','')}")
        print(f"  建议  : {a.get('recommendation','')}")
        for c in a.get("concerns", [])[:3]:
            print(f"  风险  : {c}")

    peers = "\n".join(f"[{r['role']}] {r['answer'].get('assessment','')} "
                      f"建议: {r['answer'].get('recommendation','')}" for r in res)

    def crit(r):
        role = next(x for x in ROLES if x["id"] == r["role"])
        others = "\n".join(l for l in peers.split("\n")
                           if not l.startswith(f"[{r['role']}]"))
        p2 = (FIXED.format(role=role["id"], duty=role["duty"], ns=N_STAGE, ni=len(INJ),
                           inj=INJ, basew=BASE_W.round(0).tolist(), stages=FG.STAGE_AT,
                           data=r["data_slice"], cinj=2.0, cprod=1.0,
                           stance=PERSPECTIVE.get(role["id"], ""), shared=_L.SHARED_BG)
              + ASK_CRITIQUE.format(peers=others))
        try:
            _, a2 = ask(p2, model, 300); return r["role"], a2
        except Exception: return r["role"], None

    with ThreadPoolExecutor(max_workers=8) as ex:
        C = {k: v for k, v in ex.map(crit, res) if v}
    print("\n" + "=" * 100)
    print("一之二、角色互看后的质疑与修订")
    print("=" * 100)
    for k, v in C.items():
        print(f"\n【{k}】 conf={v.get('confidence')}")
        print(f"  反驳: {v.get('challenge','')}")
        print(f"  修订: {v.get('revision','')}")

    msg_txt = "\n".join(
        f"[{r['role']} | conf={r['answer'].get('confidence')} ] "
        f"{r['answer'].get('assessment','')} 建议: {r['answer'].get('recommendation','')}"
        + (f"\n    ↳ 质疑: {C[r['role']].get('challenge','')}"
           f"\n    ↳ 二轮: {C[r['role']].get('revision','')}" if r["role"] in C else "")
        for r in res)
    print("\n" + "=" * 100)
    print(f"二、总工实际收到的全部输入({len(msg_txt)} 字)")
    print("=" * 100)
    print(msg_txt)

    raw, syn = ask(ASK_CHIEF.format(msgs=msg_txt, facts=("\n\n".join(f"[{r['role']} 转呈]\n{r['answer'].get('evidence','')}" for r in res if str(r['answer'].get('evidence','')).strip()) or "(无证据转呈)"), incumbent="", truth="(诊断模式:无历史实测)", n=6, ni=len(INJ), ns=N_STAGE, wdev=25.0), model, 300)
    print("\n" + "=" * 100)
    print("三、总工的完整回应")
    print("=" * 100)
    print(f"conf={syn.get('confidence')}  候选数={len(syn.get('candidates',[]))}")
    for i, c in enumerate(syn.get("candidates", [])):
        th = np.asarray(c["theta"], float)
        print(f"\n  候选#{i}  θ范围 {th.min():+.2f}~{th.max():+.2f}  均值 {th.mean():+.3f}")
        print(f"    理由: {c.get('rationale','')}")
        print(f"    θ   : {np.round(th,2).tolist()}")

    # 候选之间到底有多不同?
    TH = np.stack([np.asarray(c["theta"], float).ravel() for c in syn["candidates"]])
    D = np.linalg.norm(TH[:, None] - TH[None], axis=-1)
    print(f"\n候选间 L2 距离: 均值 {D[np.triu_indices(len(TH),1)].mean():.3f}  "
          f"最小 {D[np.triu_indices(len(TH),1)].min():.3f}")

    (OUT / "diag_round4.json").write_text(json.dumps(
        {"readers": res, "chief_input_chars": len(msg_txt),
         "chief_input": msg_txt, "critique": C, "chief": syn},
        ensure_ascii=False, indent=1))
    print(f"\n→ {OUT/'diag_round4.json'}")


if __name__ == "__main__":
    main()
