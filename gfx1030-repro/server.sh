#!/usr/bin/env bash
# Issue-27 repro server: the reporter's exact recipe, pinned to the 6900 XT
# (ROCm0 / HIP device 0), cores 0-15 so the :9931 server keeps 16-23.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
# HF hub cache (freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF, snapshot from refs/main); MODEL_DIR overrides
MODEL="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--freriadoc--Qwen3.8-Flash-Next-Avalon-GGUF/snapshots/372fe3a4e572c5ba75a90e579de0695633d083a8}"

export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export GGML_CUDA_MOE_CACHE_RESERVE_MB="${REPRO_RESERVE_MB:-1024}"
# fingerprint diagnostics (set in the environment before launching):
#   GGML_CUDA_MOE_HOSTED_STATS=N   per-layer hosted stats every N layers
#   GGML_CUDA_MOE_HOSTED_MAX_ROWS / GGML_MOE_GPU for the differential table
if [ -n "${GGML_CUDA_MOE_HOSTED_STATS:-}" ]; then export GGML_CUDA_MOE_HOSTED_STATS; fi
if [ -n "${GGML_MOE_HOSTED_MAX_ROWS:-}" ]; then export GGML_MOE_HOSTED_MAX_ROWS; fi
if [ -n "${GGML_MOE_GPU:-}" ]; then export GGML_MOE_GPU; fi
if [ -n "${GGML_CUDA_MOE_TIMELINE:-}" ]; then export GGML_CUDA_MOE_TIMELINE; fi

# MTP is in the reporter's recipe; REPRO_MTP=0 drops it (diagnostic: does spec decode hold the card
# up through the idle and hide the idle-floor power step)
MTP_ARGS=()
if [ "${REPRO_MTP:-1}" != "0" ]; then
    MTP_ARGS=(-md "$MODEL/MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf" \
        --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0)
fi

exec taskset -c 0-15 "$HERE/../build/bin/llama-server" \
    -m "$MODEL/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf" \
    -c 262144 -b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
    "${MTP_ARGS[@]}" \
    -t 16 -lv "${REPRO_LV:-3}" --host 127.0.0.1 --port 9932 2>&1 | tee "$HERE/server.log"
