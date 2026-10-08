#!/bin/bash
# ============================================================================
# deploy.sh — 统一部署入口（DeepSeek-V4 四模型 × 海光 DCU）
#
# 用法：
#   ./deploy.sh list                                    # 列出 4 个模型
#   ./deploy.sh serve   --model flash0731 [选项]        # Slurm 多机多卡部署
#   ./deploy.sh smoke   --model flash0731               # 单机 8 卡冒烟（无 Slurm）
#   ./deploy.sh download --model probase                # 下载权重
#   ./deploy.sh install [--no-model]                    # 环境安装 + 打补丁
#
# 通用选项：
#   --model <name>     flash0731 | flashbase | pro0813 | probase
#   --dry-run          只打印将执行的命令，不真正提交/运行
#   --script <path>    逃生舱：跳过模型默认脚本，直接用指定脚本（serve/smoke 用）
#
# serve 选项（覆盖模型默认并行策略）：
#   --nodes N --gres N --tp N --pp N --dp N --maxlen N --partition <p>
#
# 所有路径走环境变量（WS/MODELS/DAS185/WORK），见各脚本头部。
# ============================================================================
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WS
MODELS="${MODELS:-$HOME/models}"
DAS185="${DAS185:-$HOME/das185}"
WORK="${WORK:-$HOME/work}"
export MODELS DAS185 WORK

# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------
model_dir() {
  case "$1" in
    flash0731) echo "DeepSeek-V4-Flash-0731" ;;
    flashbase) echo "DeepSeek-V4-Flash-Base" ;;
    pro0813)   echo "DeepSeek-V4-Pro-0813" ;;
    probase)   echo "DeepSeek-V4-Pro-Base" ;;
    *) echo "" ;;
  esac
}
model_weight() {
  case "$1" in
    flash0731) echo "~156 GB (FP4)" ;;
    flashbase) echo "~275 GB (FP8)" ;;
    pro0813)   echo "~832 GB (MXFP4/FP8)" ;;
    probase)   echo "~1.46 TB (FP8)" ;;
    *) echo "" ;;
  esac
}
# Slurm serve 脚本（pro0813 默认 4n32c/shca 版；20n80c/mlx5 版用 --script 指定）
model_serve() {
  case "$1" in
    flash0731) echo "$WS/scripts/deploy/slurm/flash0731_serve.sh" ;;
    flashbase) echo "$WS/scripts/deploy/slurm/flash_base_serve.sh" ;;
    pro0813)   echo "$WS/scripts/deploy/slurm/v4pro_4n32c_serve.sh" ;;
    *) echo "" ;;
  esac
}
model_smoke() {
  case "$1" in
    flash0731) echo "$WS/scripts/deploy/smoke/flash0731_8card_smoke.sh" ;;
    flashbase) echo "$WS/scripts/deploy/smoke/flash_base_8card_smoke.sh" ;;
    *) echo "" ;;
  esac
}
model_download() {
  case "$1" in
    flash0731) echo "$WS/scripts/download/dl_flash_0731_base.sh" ;;
    flashbase) echo "$WS/scripts/download/dl_flash_0731_base.sh" ;;
    probase)   echo "$WS/scripts/download/dl_v4pro_base.sh" ;;
    *) echo "" ;;
  esac
}
# 默认并行策略（与各 serve 脚本注释一致）
model_defaults() {
  case "$1" in
    flash0731) echo "4 4 4 4 1 8192" ;;   # nodes gres tp pp dp maxlen
    flashbase) echo "8 4 4 8 1 4096" ;;
    pro0813)   echo "4 8 8 2 2 8192" ;;   # TP8 PP2 DP2 EP16（4n32c）
    *) echo "" ;;
  esac
}

usage() {
  sed -n '2,25p' "$WS/deploy.sh" | sed 's/^# \{0,1\}//'
  exit 1
}

err() { echo "deploy.sh: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
cmd_list() {
  printf "%-10s %-14s %-22s %-10s %s\n" "model" "weight" "serve 默认" "smoke" "下载"
  printf "%s\n" "---------------------------------------------------------------------------------"
  for m in flash0731 flashbase pro0813 probase; do
    dir=$(model_dir "$m")
    [ -z "$dir" ] && continue
    w=$(model_weight "$m")
    s=$(model_serve "$m"); s=${s:+$(basename "$s")}
    k=$(model_smoke "$m"); k=${k:+✓}
    d=$(model_download "$m"); d=${d:+✓}
    printf "%-10s %-14s %-22s %-10s %s\n" "$m" "$w" "${s:-—}" "${k:--}" "${d:--}"
  done
  echo
  echo "状态（详见 docs/KNOWN_ISSUES.md）："
  echo "  flash0731 / flashbase : 环境/权重/RCCL 通，推理卡 MoE 算子（等厂商补丁）"
  echo "  pro0813               : 排障至最后一步，差 FAST_WOA=0 未复跑（最接近跑通）"
  echo "  probase               : 仅下载脚本，从未部署验证"
}

# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
cmd_serve() {
  local MODEL="" NODES="" GRES="" TP="" PP="" DP="" MAXLEN="" PARTITION="" SCRIPT="" DRY=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --nodes) NODES="$2"; shift 2 ;;
      --gres) GRES="$2"; shift 2 ;;
      --tp) TP="$2"; shift 2 ;;
      --pp) PP="$2"; shift 2 ;;
      --dp) DP="$2"; shift 2 ;;
      --maxlen) MAXLEN="$2"; shift 2 ;;
      --partition) PARTITION="$2"; shift 2 ;;
      --script) SCRIPT="$2"; shift 2 ;;
      --dry-run) DRY=1; shift ;;
      *) err "未知选项: $1" ;;
    esac
  done
  [ -n "$MODEL" ] || err "serve 需要 --model"
  local def
  def=$(model_defaults "$MODEL")
  [ -n "$def" ] || err "模型 '$MODEL' 无 serve 脚本（用 --script 指定自定义脚本）"
  read -r d_nodes d_gres d_tp d_pp d_dp d_maxlen <<<"$def"
  NODES=${NODES:-$d_nodes}; GRES=${GRES:-$d_gres}
  TP=${TP:-$d_tp}; PP=${PP:-$d_pp}; MAXLEN=${MAXLEN:-$d_maxlen}
  DP=${DP:-$d_dp}
  SCRIPT=${SCRIPT:-$(model_serve "$MODEL")}
  [ -f "$SCRIPT" ] || err "serve 脚本不存在: $SCRIPT"

  # sbatch 命令行 -N/--gres 覆盖脚本内 #SBATCH 指令；TP/PP/DP/MAXLEN 走 --export
  # 覆盖脚本里的 ${VAR:-默认}。
  local CMD=(sbatch -N "$NODES" --gres="dcu:$GRES" \
    --export="ALL,MODELS=$MODELS,DAS185=$DAS185,WORK=$WORK,TP=$TP,PP=$PP,DP=$DP,MAXLEN=$MAXLEN")
  [ -n "$PARTITION" ] && CMD+=(--partition "$PARTITION")
  CMD+=("$SCRIPT")

  echo ">> 模型: $MODEL ($(model_weight "$MODEL"))"
  echo ">> 并行: nodes=$NODES gres=$GRES tp=$TP pp=$PP dp=$DP maxlen=$MAXLEN"
  echo ">> 脚本: $SCRIPT"
  if [ "$DRY" = 1 ]; then
    echo ">> [dry-run] ${CMD[*]}"
  else
    "${CMD[@]}"
  fi
}

# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------
cmd_smoke() {
  local MODEL="" SCRIPT="" DRY=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --script) SCRIPT="$2"; shift 2 ;;
      --dry-run) DRY=1; shift ;;
      *) err "未知选项: $1" ;;
    esac
  done
  [ -n "$MODEL" ] || err "smoke 需要 --model"
  SCRIPT=${SCRIPT:-$(model_smoke "$MODEL")}
  [ -n "$SCRIPT" ] || err "模型 '$MODEL' 无单机冒烟脚本"
  [ -f "$SCRIPT" ] || err "冒烟脚本不存在: $SCRIPT"
  echo ">> smoke: $MODEL → $SCRIPT"
  if [ "$DRY" = 1 ]; then
    echo ">> [dry-run] bash $SCRIPT"
  else
    bash "$SCRIPT"
  fi
}

# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
cmd_download() {
  local MODEL="" SCRIPT="" DRY=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --script) SCRIPT="$2"; shift 2 ;;
      --dry-run) DRY=1; shift ;;
      *) err "未知选项: $1" ;;
    esac
  done
  [ -n "$MODEL" ] || err "download 需要 --model"
  SCRIPT=${SCRIPT:-$(model_download "$MODEL")}
  [ -n "$SCRIPT" ] || err "模型 '$MODEL' 无下载脚本（pro0813 请用 --script 指定，或参考 dl_v4pro_base.sh 改 model id）"
  [ -f "$SCRIPT" ] || err "下载脚本不存在: $SCRIPT"
  echo ">> download: $MODEL → $SCRIPT"
  if [ "$DRY" = 1 ]; then
    echo ">> [dry-run] bash $SCRIPT"
  else
    bash "$SCRIPT"
  fi
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
cmd_install() {
  local NO_MODEL=0 DRY=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --no-model) NO_MODEL=1; shift ;;
      --dry-run) DRY=1; shift ;;
      *) err "未知选项: $1" ;;
    esac
  done
  local SCRIPT="$WS/scripts/install/setup_all.sh"
  local CMD=(bash "$SCRIPT")
  [ "$NO_MODEL" = 1 ] && CMD+=(--no-model)
  echo ">> install → $SCRIPT"
  if [ "$DRY" = 1 ]; then
    echo ">> [dry-run] ${CMD[*]}"
  else
    "${CMD[@]}"
  fi
}

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
main() {
  [ $# -ge 1 ] || usage
  case "$1" in
    list) shift; cmd_list "$@" ;;
    serve) shift; cmd_serve "$@" ;;
    smoke) shift; cmd_smoke "$@" ;;
    download) shift; cmd_download "$@" ;;
    install) shift; cmd_install "$@" ;;
    -h|--help|help) usage ;;
    *) err "未知命令: $1（可用 list/serve/smoke/download/install）" ;;
  esac
}

main "$@"
