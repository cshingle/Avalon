#!/usr/bin/env bash
# Same as server.sh, but the reporter's -b 9536 -ub 9536 build (their second reset
# path: a 30k-token prefill that resets without the hosted path).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HOME/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF"

export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export GGML_CUDA_MOE_CACHE_RESERVE_MB="${REPRO_RESERVE_MB:-1024}"
if [ -n "${GGML_CUDA_MOE_HOSTED_STATS:-}" ]; then export GGML_CUDA_MOE_HOSTED_STATS; fi
if [ -n "${GGML_MOE_HOSTED_MAX_ROWS:-}" ]; then export GGML_MOE_HOSTED_MAX_ROWS; fi
if [ -n "${GGML_MOE_GPU:-}" ]; then export GGML_MOE_GPU; fi
if [ -n "${GGML_CUDA_MOE_TIMELINE:-}" ]; then export GGML_CUDA_MOE_TIMELINE; fi

exec taskset -c 0-15 "$HERE/../build/bin/llama-server" \
    -m "$MODEL/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf" \
    -c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
    -md "$MODEL/MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf" \
    --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 \
    -t 16 -lv "${REPRO_LV:-3}" --host 127.0.0.1 --port 9932 2>&1 | tee "$HERE/server-9536.log"
