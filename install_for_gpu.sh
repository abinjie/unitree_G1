#!/usr/bin/env bash
# GPU-aware CUDA / PyTorch / mjlab installer.
# Detects the NVIDIA GPU and driver, picks the matching PyTorch CUDA wheel,
# and (optionally) installs mjlab or a standalone PyTorch.
#
# Focus: works on RTX 5090 (Blackwell, sm_120) now and Tesla T4 (Turing, sm_75)
# for later migration. T4 needs driver >= 525.60.13 to run the cu128 wheel via
# CUDA minor-version compatibility; older drivers fall back to CPU with a warning.
#
# Usage:
#   ./install_for_gpu.sh                 # detect + report only (no changes)
#   ./install_for_gpu.sh --mjlab         # uv sync --extra <cu128|cpu> in ./mjlab
#   ./install_for_gpu.sh --mjlab /path   # ...in /path
#   ./install_for_gpu.sh --pytorch       # uv pip install torch+torchvision
#   ./install_for_gpu.sh --dry-run --mjlab   # preview commands, install nothing
#
# Exit codes: 0 ok / 1 bad args / 2 no NVIDIA GPU / 3 unsupported GPU or driver

set -euo pipefail

# ─── config ──────────────────────────────────────────────────────────────────
DRY_RUN=false
VERBOSE=false
ACTION="detect"            # detect | mjlab | pytorch
MJLAB_DIR=""               # default: script's own dir (set in main)
# Large CUDA wheels (nvidia-*, torch+cu128) often exceed uv's 30s default when
# many packages download in parallel. Override only if the user has not set them.
: "${UV_HTTP_TIMEOUT:=300}"
: "${UV_CONCURRENT_DOWNLOADS:=4}"
export UV_HTTP_TIMEOUT UV_CONCURRENT_DOWNLOADS
# PyTorch CUDA wheel indexes (only cu128 ships with mjlab; cu121/cu124 for
# standalone PyTorch when the driver cannot do 12.x but can do 12.0/12.1).
declare -A PT_INDEX=(
  [cu128]="https://download.pytorch.org/whl/cu128"
  [cu124]="https://download.pytorch.org/whl/cu124"
  [cu121]="https://download.pytorch.org/whl/cu121"
  [cpu]="https://download.pytorch.org/whl/cpu"
)

# ─── helpers ─────────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m⚠\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; }
vlog() { $VERBOSE && printf '\033[2m  %s\033[0m\n' "$*" || true; }
run()  {
  if $DRY_RUN; then printf '\033[2m  [dry-run]\033[0m %s\n' "$*"; else vlog "→ $*"; "$@"; fi
}

# ─── arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mjlab)        ACTION="mjlab"; [[ $# -ge 2 && "$2" != -* ]] && { MJLAB_DIR="$2"; shift; } ;;
    --pytorch)      ACTION="pytorch" ;;
    --dry-run)      DRY_RUN=true ;;
    -v|--verbose)   VERBOSE=true ;;
    -h|--help)
      sed -n '2,16p' "$0"; exit 0 ;;
    *) err "unknown option: $1"; exit 1 ;;
  esac
  shift
done

# ─── GPU detection ───────────────────────────────────────────────────────────
require_nvidia() {
  command -v nvidia-smi >/dev/null || { err "nvidia-smi not found — no NVIDIA driver installed."; exit 2; }
}

# Echo "name<TAB>driver<TAB>cc" for the first GPU (multi-GPU: first wins, warn later).
read_gpu() {
  local name driver cc
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
  [[ -z "$name" ]] && { err "nvidia-smi returned no GPU."; exit 2; }
  # compute_cap field missing on very old drivers → fall back to name parsing.
  if [[ -z "$cc" ]]; then
    cc=$(cc_from_name "$name")
    vlog "compute_cap query empty; inferred cc=$cc from name"
  fi
  printf '%s\t%s\t%s\n' "$name" "$driver" "$cc"
}

# Fallback compute capability from GPU name (covers common datacenter / consumer cards).
cc_from_name() {
  case "$1" in
    *Blackwell*|*"RTX 5090"*|*"RTX 5080"*|*"GB200"*)   echo 12.0 ;;
    *"B200"*|*"B100"*)                                  echo 12.0 ;;
    *"H100"*|*"H200"*|*Hopper*)                         echo 9.0 ;;
    *"RTX 4090"*|*"RTX 4080"*|*"RTX 4070"*|*Ada*)       echo 8.9 ;;
    *"RTX 3090"*|*"RTX 3080"*|*"RTX 3070"*|*A40*|*A16*) echo 8.6 ;;
    *"A100"*|*Ampere*)                                  echo 8.0 ;;
    *"T4"*|*"Tesla T4"*|*"RTX 2080"*|*"RTX 2070"*|*"RTX 2060"*|*Turing*) echo 7.5 ;;
    *"V100"*|*Volta*)                                   echo 7.0 ;;
    *"P100"*|*"P40"*|*Pascal*)                          echo 6.0 ;;
    *) echo "" ;;
  esac
}

# Max CUDA version the installed driver supports. Primary source: the
# "CUDA Version: X.Y" line in the nvidia-smi header (most reliable). Fallback:
# driver-version → CUDA mapping for older drivers where the header differs.
driver_max_cuda() {
  local drv="$1"
  local hdr
  hdr=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -n1 | grep -oE '[0-9]+\.[0-9]+')
  if [[ -n "$hdr" ]]; then printf '%s\n' "$hdr"; return; fi
  local major=${drv%%.*}
  case "$major" in
    58*|57*|580*|570*) echo 12.8 ;;
    565|56*) echo 12.6 ;; 550|55*) echo 12.4 ;; 545) echo 12.3 ;;
    535|53*) echo 12.2 ;; 530) echo 12.1 ;; 525|52*) echo 12.0 ;;
    520) echo 11.8 ;; 510) echo 11.6 ;; 495) echo 11.5 ;; 470) echo 11.4 ;;
    460) echo 11.2 ;; 450) echo 11.0 ;; 440) echo 10.2 ;; 418) echo 10.1 ;;
    *) echo "" ;;
  esac
}

# ─── arch classification ─────────────────────────────────────────────────────
# Echo: arch<TAB>min_cu_for_arch<TAB>needs_cu128
classify() {
  local cc="$1" major minor
  major=${cc%%.*}; minor=${cc#*.}
  case "$major" in
    12) printf 'Blackwell\t12.8\t1\n' ;;   # sm_120+: cu128 mandatory
    9)  printf 'Hopper\t11.8\t0\n' ;;
    8)  if [[ "$minor" == 9 ]]; then printf 'Ada\t11.8\t0\n'
        else printf 'Ampere\t11.8\t0\n'; fi ;;
    7)  if [[ "$minor" == 5 ]]; then printf 'Turing\t10.0\t0\n'
        else printf 'Volta\t9.0\t0\n'; fi ;;
    6)  printf 'Pascal\t8.0\t0\n' ;;
    *)  printf 'unknown\t0.0\t0\n' ;;
  esac
}

# ─── wheel selection ─────────────────────────────────────────────────────────
# Returns the extra/wheel tag to install: cu128 | cu124 | cu121 | cpu.
# Logic:
#   • Blackwell (sm_120+) → cu128 (only wheel with sm_120 SASS). Driver MUST be ≥580.
#   • Others, driver supports CUDA 12.x (≥525) → cu128 (runs via minor-version
#     compatibility on T4/Ampere/Ada/etc.). This is the mjlab-supported path.
#   • Others, driver only supports CUDA 12.0/12.1 → cu121/cu124 (standalone PyTorch only;
#     mjlab cannot use these — its pyproject only ships cu128/cpu).
#   • Driver too old for any 12.x → cpu (with warning).
pick_wheel() {
  local arch="$1" needs_cu128="$2" drv_max="$3"
  if [[ "$needs_cu128" == 1 ]]; then
    if [[ -n "$drv_max" ]] && ver_ge "$drv_max" 12.8; then echo cu128; return; fi
    err "Blackwell GPU needs CUDA 12.8+ but driver supports only ${drv_max:-?}."
    exit 3
  fi
  # Non-Blackwell: cu128 works wherever the driver supports CUDA 12.0+ (minor-version compat).
  if [[ -n "$drv_max" ]] && ver_ge "$drv_max" 12.0; then echo cu128; return; fi
  # Driver can't do 12.x at all → only standalone PyTorch fallbacks apply.
  if [[ -n "$drv_max" ]] && ver_ge "$drv_max" 12.1; then echo cu124; return; fi
  if [[ -n "$drv_max" ]] && ver_ge "$drv_max" 12.0; then echo cu121; return; fi
  echo cpu
}

ver_ge() {  # ver_ge  X Y  → true if X >= Y
  awk -v a="$1" -v b="$2" 'BEGIN{ split(a,A,"."); split(b,B,".");
    for(i=1;i<=2;i++){ if(A[i]+0>B[i]+0) exit 0; if(A[i]+0<B[i]+0) exit 1 } exit 0 }'
}

# ─── T4-specific guidance ────────────────────────────────────────────────────
t4_report() {
  local drv_max="$1" wheel="$2" drv_ok="⚠ 当前驱动不满足 12.0, 需先升级驱动."
  if [[ -n "$drv_max" ]] && ver_ge "$drv_max" 12.0; then drv_ok="✓ 当前驱动满足."; fi
  cat <<EOF

  ── Tesla T4 迁移备忘 ─────────────────────────────────────────
  硬件: Turing, sm_75, 16GB VRAM. 训练 G1 velocity 建议 num-envs
        ≤1024 (T4 显存 + 算力远低于 5090, 且 fp32 吞吐有限).
  驱动: 当前支持 CUDA ${drv_max:-?}. cu128 wheel 在 T4 上靠 CUDA 12.x
        小版本兼容运行, 门槛是驱动 ≥ 525.60.13 (即支持 CUDA 12.0+).
        $drv_ok
  wheel: 选用 ${wheel}. 这是 mjlab 唯一支持的 CUDA extra.
  若驱动无法升级: 只能用 cpu extra (无法用 GPU 训练), 或换卡.
  ──────────────────────────────────────────────────────────────
EOF
}

# ─── installers ──────────────────────────────────────────────────────────────
install_mjlab() {
  local wheel="$1" dir="$2"
  [[ ! -d "$dir" ]] && { err "mjlab dir not found: $dir"; err "clone first: git clone https://github.com/mujocolab/mjlab.git $dir"; exit 1; }
  local extra=$wheel
  [[ "$wheel" == "cpu" ]] && extra=cpu
  # cu124/cu121 aren't mjlab extras — only cu128/cpu exist in its pyproject.
  if [[ "$wheel" != "cu128" && "$wheel" != "cpu" ]]; then
    err "mjlab only ships cu128/cpu extras, but this GPU/driver needs '$wheel'."
    err "options: (a) upgrade driver so cu128 works, (b) use --pytorch for standalone, (c) force cpu."
    exit 3
  fi
  log "Installing mjlab with --extra $extra in $dir"
  log "uv download: UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}s, concurrent=${UV_CONCURRENT_DOWNLOADS}"
  ( cd "$dir" && run uv sync --extra "$extra" )
  ok "mjlab ready. train with: cd $dir && uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 1024"
}

install_pytorch() {
  local wheel="$1"
  command -v uv >/dev/null || { err "uv not installed. install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  log "Installing PyTorch ($wheel) via uv pip"
  log "uv download: UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}s, concurrent=${UV_CONCURRENT_DOWNLOADS}"
  run uv pip install torch torchvision --index-url "${PT_INDEX[$wheel]}"
  ok "PyTorch ($wheel) installed. verify: python -c 'import torch; print(torch.cuda.is_available())'"
}

# ─── main ────────────────────────────────────────────────────────────────────
main() {
  require_nvidia
  [[ -z "$MJLAB_DIR" ]] && MJLAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local raw name driver cc arch min_cu needs_cu128 drv_max wheel
  raw=$(read_gpu); IFS=$'\t' read -r name driver cc <<<"$raw"
  drv_max=$(driver_max_cuda "$driver")
  raw=$(classify "$cc"); IFS=$'\t' read -r arch min_cu needs_cu128 <<<"$raw"
  wheel=$(pick_wheel "$arch" "$needs_cu128" "$drv_max")

  cat <<EOF
$(ok 'GPU 检测结果')
  名称          : $name
  驱动版本      : $driver
  Compute Cap  : $cc   (arch: $arch)
  驱动支持最高  : CUDA ${drv_max:-未知}
  ─────────────────────────────────
  推荐 wheel    : $wheel
  arch 最低 CUDA: $min_cu
  必须 cu128    : $([[ $needs_cu128 == 1 ]] && echo yes || echo no)
EOF

  [[ "$arch" == "unknown" ]] && { err "unrecognized GPU (cc=$cc). add it to cc_from_name()."; exit 3; }
  [[ "$wheel" == "cpu" && "$arch" != "unknown" ]] && warn "GPU present but driver too old for CUDA 12.x → falling back to CPU wheel."

  # T4 focus: always print T4 guidance when the detected GPU is a T4.
  if [[ "$name" == *"T4"* ]]; then t4_report "$drv_max" "$wheel"; fi

  case "$ACTION" in
    detect)  log "dry detection only. pass --mjlab or --pytorch to install." ;;
    mjlab)   install_mjlab "$wheel" "$MJLAB_DIR" ;;
    pytorch) install_pytorch "$wheel" ;;
  esac
}

main "$@"
