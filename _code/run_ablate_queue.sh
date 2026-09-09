#!/usr/bin/env bash
# 定稿(F2)配置下的留一角色消融 —— 用来定"最强的 Agent Team 到底是几个角色"。
#
# 为什么要重跑:早期批次(Rno*)的留一消融是在旧配置下跑的,同批全角色参照只有
# +204.3M,而定稿配置是 +361.3M。两批不可比,不能拿旧结论定最终模型。
#
# 与 run_coarse_queue.sh **完全相同**的命令行,只差 --drop,所以结果可直接与
# 已有的 F2full7(n=10, +361.3M)比较。
#
#   ROLES="geomechanics_expert" REPS=10 bash run_ablate_queue.sh
#
set -u
cd "$(dirname "$0")"
PY=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python
L=../_pipelines/fc_team
REPS=${REPS:-10}
ROLES=${ROLES:-geomechanics_expert}
JOBS=()
for r in $ROLES; do
  short=$(echo "$r" | cut -c1-4)   # geom=geomechanics, seis=seismic_4d …
  for k in $(seq 1 "$REPS"); do JOBS+=("$r|$short|$k"); done
done
mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | shuf --random-source=<(yes 20260908))
echo "消融队列 ${#JOBS[@]} 个  重复 $REPS/配置  角色: $ROLES  开始 $(date '+%F %T')"

run_one() {
  IFS='|' read -r role short k <<< "$1"
  tag="_G2no${short}_$k"
  [ -f "$L/loop$tag.json" ] && { echo "  跳过 $tag"; return; }
  "$PY" -u fc_team_loop.py --model sonnet \
    --max-rounds 3 --n-cand 6 --n-sim 1 --threads 6 --device cpu \
    --c-inj 2.0 --c-prod 1.0 --w-max-dev 0.25 --tag "$tag" --drop "$role" \
    > "$L/run$tag.log" 2>&1
  echo "  完成 $tag rc=$? $(date '+%T')"
}
export -f run_one; export PY L
printf '%s\n' "${JOBS[@]}" | xargs -P 3 -I{} bash -c 'run_one "$@"' _ {}
echo "消融队列结束 $(date '+%F %T')"
