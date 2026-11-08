# gfx1030 issue-27 repro harness

Replays the reporter's trigger pattern (https://github.com/freriadoc/Avalon/issues/27)
on our RX 6900 XT (gfx1030) to fingerprint the hosted-MoE launch burst on healthy
hardware. Safe to run alongside the :9931 server on the 7900 XT.

Layout:
- `server.sh`  - reporter's exact flags/env, pinned: HIP device 0 (6900 XT),
  cores 0-15, port 9932, log to `server.log`
- `sampler.sh` - 200 ms CSV of sclk / busy / power / temp / MemAvailable / AER,
  `./sampler.sh out.csv 0 [pci_bdf]` (bdf auto-detected when omitted)
- `replay.py`  - synthetic ~23k-token tool prompt, then unrelated short prompts
  (sizes land within ~1%, measured by the server's tokenizer)

Run:
    ./server.sh &                     # minutes to load; watch server.log
    ./sampler.sh run1.csv 0 &         # until killed
    ./replay.py replay                # big -> 5 s idle -> 535-token short
    ./replay.py sweep 166 266 346 535 # threshold sweep
    kill %2                           # stop the sampler

Notes:
- **Set `REPRO_LV=5`**: at default verbosity the ggml-side lines (`[moe-cache]`,
  `moe hosted:` stats, `ggml_cuda_init`) are suppressed and never reach the log.
  The hosted path can be running with zero visible evidence.
- Diagnostics env: `GGML_CUDA_MOE_HOSTED_STATS=8` (per-layer hosted stats),
  `GGML_MOE_HOSTED_MAX_ROWS=N`, `GGML_MOE_GPU=0` (hosted off).
- Each `replay.py` run calibrates token/char on the live tokenizer.
- Short prompts are freshly random each time: no shared prefix with the big one.
- The expert cache is warm after the first replay; to replicate the reporter's
  cold-cache resets, restart server.sh between replays.
- AER fatal counters on 0000:07:00.0 are in the CSV; baseline is 0.
- Watch `memavail_kb` in the CSV: the moe-cache pools pin host memory and
  ~58 GB headroom is shared with the :9931 server.
