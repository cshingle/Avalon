#!/usr/bin/env bash
# 200 ms GPU + RAM sampler for the issue 27 reset trace (Avalon gfx1030).
# usage: sampler.sh [outfile] [duration_s] [pci_bdf]
#   pci_bdf: e.g. 0000:03:00.0 - default: the first amdgpu that reports gpu_busy_percent
#   duration 0 = until killed
# Records busy %, sclk, power, temp, MemAvailable and the PCIe AER fatal counter.
# Keep it running across the reset hunt; after reboot the CSV ends where the machine died.
set -u
OUT=${1:-sampler.csv}
DUR=${2:-0}
D=${3:-}
if [ -z "$D" ]; then
    for f in /sys/bus/pci/devices/*/gpu_busy_percent; do
        [ -e "$f" ] && D=$(dirname "$f") && break
    done
fi
[ -n "$D" ] || { echo "no amdgpu device with gpu_busy_percent found" >&2; exit 1; }
H=$(ls -d "$D"/drm/card*/device/hwmon/hwmon* 2>/dev/null | head -1)

rd() { cat "$1" 2>/dev/null || echo 0; }
aer_fatal() { awk '/TOTAL_ERR_FATAL/{print $2}' "$D/aer_dev_fatal" 2>/dev/null || echo na; }

echo "t_s,busy_pct,sclk_mhz,power_uw,temp_mc,memavail_kb,aer_fatal" > "$OUT"
T0=$(date +%s%N)
while :; do
    NOW=$(date +%s%N)
    T=$(( (NOW - T0) / 1000000 ))
    if [ "$DUR" != "0" ] && [ "$T" -ge $(( DUR * 1000 )) ]; then break; fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
        "$(awk -v t="$T" 'BEGIN{printf "%.1f", t/1000}')" \
        "$(rd "$D/gpu_busy_percent")" \
        "$(awk -v f="$(rd "$H/freq1_input")" 'BEGIN{printf "%.0f", f/1000000}')" \
        "$(rd "$H/power1_average")" \
        "$(rd "$H/temp1_input")" \
        "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
        "$(aer_fatal)" >> "$OUT"
    sleep 0.2
done
