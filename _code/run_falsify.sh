#!/usr/bin/env bash
# 🔴 这不是"再删一个角色看数字涨不涨"的贪心搜索,而是对**解释本身**的可证伪检验。
#
# 从地质力学那件事得到的解释是:
#   「没有本油田证据、却越界给出硬数字上限的角色,会拖累决策。」
#
# 4D 地震专家同为 literature 类、自报置信度更低(0.19),但**行为不同**:
# 它明说"只有标题与 DOI、无数值,不得编造",从不给硬上限。
#
# 于是解释给出一个可证伪的预测:
#   删掉 4D 地震  **不应该**带来提升。
#
#   若删了也变好 → 解释被证伪,真相只是"角色越少越好"
#   若删了没变化 → 解释成立,问题不在"有没有文献"而在"越不越界"
#
# 与 run_decide_team.sh 命令行完全一致,只差 --drop 的角色。
set -u
cd "$(dirname "$0")"
PY=/mnt/data/yongan-admin-2/envs/volve-chronos2/bin/python
L=../_pipelines/fc_team
FROM=${FROM:-1}; TO=${TO:-30}
JOBS=(); for k in $(seq "$FROM" "$TO"); do JOBS+=("_G2noseis_$k"); done
mapfile -t JOBS < <(printf '%s\n' "${JOBS[@]}" | shuf --random-source=<(yes 20260909))
echo "证伪队列 ${#JOBS[@]} 个  删 seismic_4d_analyst  开始 $(date '+%F %T')"
run_one() {
  tag="$1"
  [ -f "$L/loop$tag.json" ] && { echo "  跳过 $tag"; return; }
  "$PY" -u fc_team_loop.py --model sonnet \
    --max-rounds 3 --n-cand 6 --n-sim 1 --threads 6 --device cpu \
    --c-inj 2.0 --c-prod 1.0 --w-max-dev 0.25 --tag "$tag" --drop seismic_4d_analyst \
    > "$L/run$tag.log" 2>&1
  echo "  完成 $tag rc=$? $(date '+%T')"
}
export -f run_one; export PY L
printf '%s\n' "${JOBS[@]}" | xargs -P ${PAR:-2} -I{} bash -c 'run_one "$@"' _ {}
echo "证伪队列结束 $(date '+%F %T')"
