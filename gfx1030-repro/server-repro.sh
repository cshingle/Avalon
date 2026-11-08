#!/usr/bin/env bash
# Same as server.sh, but the reporter's build (commit 8c4e475, build-repro).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="$HOME/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF"

export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0
export GGML_CUDA_MOE_CACHE_RESERVE_MB="${REPRO_RESERVE_MB:-1024}"

exec taskset -c 0-15 "$HERE/../build-repro/bin/llama-server" \
    -m "$MODEL/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf" \
    -c 262144 -b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
    -md "$MODEL/MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf" \
    --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 \
    -t 16 -lv "${REPRO_LV:-3}" --host 127.0.0.1 --port 9932 2>&1 | tee "$HERE/server-repro.log"
