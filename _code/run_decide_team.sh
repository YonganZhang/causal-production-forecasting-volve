#!/usr/bin/env bash
# 把「全 7 角色」与「去掉地质力学」两臂都补到 n=20,以裁定最终团队构成。
# n=10 时差 +35.2M 但 95% CI [-9,+80] 仍含 0;LLM 输出有波动,需要更多重复。
# 命令行与 run_coarse_queue.sh / run_ablate_queue.sh 完全一致,只差 --drop。
# 已完成的 tag 会跳过,可安全反复调用补齐(2026-09-08 曾因 LLM 额度耗尽中断 9 次)。
# 并发用 PAR 控制,额度紧张时降到 2。
set -u
cd "$(dirname "$0")"
PY=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python
L=../_pipelines/fc_team
FROM=${FROM:-11}; TO=${TO:-20}
JOBS=()
for k in $(seq "$FROM" "$TO"); do JOBS+=("full|_F2full7_$k" "geom|_G2nogeom_$k"); done
mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | shuf --random-source=<(yes 20260908b))
echo "定队列 ${#JOBS[@]} 个  重复 $FROM-$TO  开始 $(date '+%F %T')"
run_one() {
  IFS='|' read -r kind tag <<< "$1"
  [ -f "$L/loop$tag.json" ] && { echo "  跳过 $tag"; return; }
  if [ "$kind" = "geom" ]; then D=(--drop geomechanics_expert); else D=(); fi
  "$PY" -u fc_team_loop.py --model sonnet \
    --max-rounds 3 --n-cand 6 --n-sim 1 --threads 6 --device cpu \
    --c-inj 2.0 --c-prod 1.0 --w-max-dev 0.25 --tag "$tag" "${D[@]}" \
    > "$L/run$tag.log" 2>&1
  echo "  完成 $tag rc=$? $(date '+%T')"
}
export -f run_one; export PY L
printf '%s\n' "${JOBS[@]}" | xargs -P ${PAR:-2} -I{} bash -c 'run_one "$@"' _ {}
echo "定队列结束 $(date '+%F %T')"
