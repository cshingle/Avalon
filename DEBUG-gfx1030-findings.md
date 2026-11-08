# Findings: Avalon#27 gfx1030 hosted-MoE reset

Running log for `DEBUG-gfx1030-reset-plan.md`. Bench = RX 6900 XT (gfx1030), PCI 0000:07:00.0.

## 2026-11-07: logging gotcha (invalidated the first three repro runs' conclusions)

At default `-lv 3` the ggml-side log lines (everything from libggml-hip/libggml-base:
`ggml_cuda_init`, `[moe-cache]`, `moe hosted:` stats) never reach llama-server's output -
the ggml log threshold is ERROR unless verbosity >= 5 (see the `common_fit_params(...
GGML_LOG_LEVEL_ERROR)` pattern in common/common.cpp:1302; the reporter obviously runs
with `-lv`). All "hosted never executes on the bench" conclusions from runs 3-6 were
wrong: hosted ran silently the whole time.

Rule for every repro run from now on: `REPRO_LV=5` (already wired into server.sh /
server-repro.sh).

## 2026-11-07: what actually runs (run7, main build 24b76099a, -lv 5)

- `find_hosted_moe` runs only for graphs on the sched that owns the moe-cache session;
  it marks all decode/small-graph MoE blocks hosted: `max_rows=8191`, cost model never
  fires (`cost_seen=0`), `hosted_set=144` (48 blocks x 3 tensors). Prefill graphs are
  rejected by row count (`rows_rej=47`, first rejected block: 20480 rows).
- The real row gate for the reporter's `MAX_ROWS` effect is the plan-time cap in
  moe-cache.cu, not the sched function.
- Hosted on our bench: ~106-206 CPU rows/layer during short requests, 7085 experts moved
  to GPU over the session. No `borrow`/`give back`/`loan plan` lines here at all (our
  budget/preload never lends; the reporter's card runs with those active - their reset's
  last words).

## 2026-11-07: differential fingerprints of the short request (plan Phase 2, rows 1/3/4)

Procedure each: fresh server (cold cache), 23k-token request, 5 s idle, ~600-token
unrelated request. Sampler 200 ms on the 6900 XT. Files: `gfx1030-repro/run{8,9,10}.csv`,
logs `run{8,9,10}-*.log`.

| config | ttft | trajectory during the short request |
|---|---|---|
| run8 hosted default (reporter: reset) | 8.9 s | 93% @2123 MHz 78 W -> fall to **3 s at 499 MHz, 13 W, ~40 % busy (poller phase)** -> re-ramp -> **99 % @2523-2540 MHz, 196 W peak** -> done. Step: 13 W/499 MHz to 196 W/2540 MHz in <1 s |
| run9 MAX_ROWS=256 (reporter: stable) | 16.4 s | single ramp, then 17 s sustained 99 % @2540 MHz, 84-113 W plateau. No poller phase, max 164 W |
| run10 MOE_GPU=0 (reporter: stable) | 16.8 s | same sustained-boost shape, 88-108 W; trailing 5 s at 499 MHz/43 W is cache fill, after completion |

**The delta**: hosted adds a deep-power (13 W) + low-clock (499 MHz) poller phase inside
the request, then a full-clock full-power burst starting from it, twice in one request
(the first partial spike at t+0 precedes the poller phase too). Both stable configs are
one monotonic ramp to a moderate sustained plateau. The unstable fingerprint contains a
180 W / 2000 MHz power-clock step with sub-second rise time, immediately after 3 s of
near-idle silicon. Matches the reporter's reset samples (45 W @2525 MHz busy / 5 W idle)
and their 0.4 s-into-request death point.

Our 6900 XT takes the same pattern at 2547 MHz and never resets (AER fatal 0).

## 2026-11-07: Phase 2 complete (rows 2/5/7 + forced lending, runs 11-14)

| config | our trajectory | reporter |
|---|---|---|
| run11 166-token follow-up, cold | clean single ramp from deep idle, 2.5 s burst to 151 W/2547 MHz, no poller pin | stable |
| run12 hosted batch while GPU busy (no idle gap) | sustained 99 % @2525-2540 MHz, 130-204 W, no poller phase, no clock fall | stable |
| run13 30k prefill with -b 9536 -ub 9536 (non-hosted reset) | sustained 99 % @2510-2530 MHz, 170-215 W mean, transient peaks 236-271 W for 100 s | reset |
| run14 hosted default + forced cache lending (reserve 3072 MiB, pool 4.5 GB; 19 borrow/give-back cycles incl. 795 MiB give back + 133/266 MiB borrows mid-request) | same two-step hosted pattern as run8 (poller 13 W/499 MHz -> 178 W/2530 MHz); lending adds no visible power steps at 200 ms resolution | n/a |

Reading (consistent across all rows):

- Steady full-clock load is NOT the killer per se: run9/run10/run12 sustain 99 %
  @2540 MHz, and the hosted merge burst at boost clock (run12) is harmless.
- Both reset configs share a fast transition from a low-power/low-voltage state to
  max boost + high current: hosted = 5 W idle -> first partial spike (123 W, 2547 MHz,
  0.2 s) -> 3 s poller at 499 MHz/13 W -> second step to 196 W/2540 MHz (<1 s);
  ub9536 prefill = idle -> sustained 190-215 W at 2530 MHz with 236-271 W peaks.
- The hosted 166-token stable case has the same first spike but no second step
  (poller too short to pin the clock low). The hosted stable configs eliminate the
  poller phase entirely. This isolates the **poller-phase -> merge-burst step** as the
  hosted-path discriminator.
- Their card's cache lending (give back/borrow/loan plan, their reset's last log words)
  reproduces here under budget pressure and does not measurably change the power
  fingerprint; it marks where they were in the pipeline, not an extra stressor.

Fork bug found on the way: with the cache pool squeezed to 486 MiB
(`GGML_CUDA_MOE_CACHE_RESERVE_MB=7168`), warmup aborts at
`ggml_cuda_graph_evaluate_and_capture: MUL_MAT_ID with host weights outside a hosted
MoE block (ffn_moe_gate-1)` (ggml-cuda.cu:6744) - should fall back instead of crashing.

## Fix implications (Phase 4 re-ranked from the table)

1. Kill the poller->burst step on gfx103x: (a) shorten/avoid the poller phase (overlap
   CPU compute so CTAs don't sit 3 s at 499 MHz), (b) ramp the clock before the merge
   launch (small warmup dispatch or grid staging), or (c) default
   `GGML_MOE_HOSTED_MAX_ROWS` low on Navi 21 so short requests take the stable
   prefetch path (run9 shape - already proven stable for the reporter).
2. For the ub9536 prefill reset: soft-start of the first ubatch after idle (split the
   first chunk), or a documented RDNA2 ubatch cap. Their 2048/2048 recipe never reset.
3. Reporter interim: `GGML_MOE_HOSTED_MAX_ROWS=256` + `-b 2048 -ub 2048` covers both
   reset paths with configs we measured as fingerprint-clean here.

## Open items

- Phase 2 rows still to capture: 166-token follow-up (their stable threshold), the
  tail-chunk case (hosted batch while GPU already busy), and the `-b 9536` 30k prefill
  reset path (not hosted; the :9931 daily server runs those flags, fingerprint it here).
- Their card runs with cache lending active (`borrow`/`loan plan` lines); ours never
  lends. If lending changes the fingerprint (pool copies during the poller phase), we
  must reproduce it here - tighten the cache budget to force lending.
- `GGML_MOE_HOST_DEBUG` prints in ggml-backend.cpp are TEMP; remove before any PR.
