#!/bin/bash
# 消融重复队列:7 个臂各再跑 5 次，把可分辨阈值从 71.8M$ 压到约 30M$。
# 受控并行，避免占满共享机器。
cd "$(dirname "$0")"
PY=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python
L=../_pipelines/fc_team
PAR=${PAR:-6}
ROLES=(connectivity_analyst economics_analyst geomechanics_expert
       reservoir_engineer production_surveillance seismic_4d_analyst constraint_auditor)
ABBR=(noconn noecon nogeo nores noprod noseis nocons)

JOBS=()
for k in 1 2 3 4 5; do
  for i in "${!ROLES[@]}"; do
    JOBS+=("${ABBR[$i]}|${ROLES[$i]}|$k")
  done
done
echo "队列长度 ${#JOBS[@]}  并行度 $PAR  开始 $(date '+%F %T')"

run_one() {
  IFS='|' read -r ab role k <<< "$1"
  tag="_R${ab}_$k"
  [ -f "$L/loop$tag.json" ] && { echo "  跳过(已存在) $tag"; return; }
  CUDA_VISIBLE_DEVICES=1 "$PY" -u fc_team_loop.py --model sonnet \
    --max-rounds 3 --n-cand 6 --n-sim 1 --threads 6 --device cuda:0 \
    --c-inj 2.0 --c-prod 1.0 --w-max-dev 0.25 --tag "$tag" --drop "$role" \
    > "$L/run$tag.log" 2>&1
  echo "  完成 $tag rc=$?  $(date '+%T')"
}
export -f run_one; export PY L

printf '%s\n' "${JOBS[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}
echo "队列结束 $(date '+%F %T')"
