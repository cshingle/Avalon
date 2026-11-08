# Debug plan: Avalon#27 - gfx1030 hard reset on first big hosted MoE batch

Branch: `fix-gfx1030-hosted-reset` (off `fix-gfx1030-stall`, on top of `origin/main`)
Issue: https://github.com/freriadoc/Avalon/issues/27

## Working hypothesis

A large hosted MoE batch (grid = one workgroup per routed row, `k_moe_hosted_merge`,
moe-cache.cu:11042) launched while the Navi 21 card ramps from idle to ~2500 MHz causes
a platform-level hard reset on marginal cards. Every non-resident row's workgroup
busy-polls a flag in coherent (`hipHostMallocCoherent`) mapped host memory
(moe-cache.cu:8244) for the whole CPU-compute duration, on every MoE layer: thousands of
CTAs issuing concurrent uncached PCIe reads at boost clock, starting from idle.

Already ruled out by the reporter: PSU/power draw, RAM, CPU boost, IOMMU (`iommu=pt`),
kernel version, ROCm version, cache lending, prefill-inplace, arena idle, cache mode off,
hosted off (`GGML_MOE_GPU=0`: stable), hosted limited to 256 rows (stable).
Clock cliff: resets >= 2400 MHz, stable <= 2200 MHz.

Still open: ROCr mman mapping (`HSA_ENABLE_MMAN_MODE=0` untested), PCIe ASPM
(`iommu=pt` does not cover it), coherent vs WC mapping policy, the exact launch transient,
and the separate `-b 9536` reset during 30k prefill (not the hosted path - a fix limited
to hosted launch may be incomplete).

## Strategy

The reset is a property of the reporter's marginal silicon; the software bug is the
pattern that marginal silicon cannot tolerate. The pattern is fully measurable on a
healthy card, so:

- The bench (stable 6900 XT) is the **pattern lab**: differential fingerprinting of
  stable-vs-unstable configurations, mitigation development, attenuation metrics.
- The reporter is a **sparingly used reset oracle**: rank-ordered decisive tests, one
  variable per build. Validating a working fix costs them zero reboots (stable runs);
  only a failed candidate costs one.
- The undervolt reset-hunt on our own card is a **contingency**, not a phase: escalate
  only after two failed candidates have burned two of their reboots.

## Bench

- GPU[0]: RX 6900 XT 16 GB (gfx1030) - free, reserved for this work. Pin with
  `HIP_VISIBLE_DEVICES=0` (verify the index -> card mapping first).
- GPU[1]: RX 7900 XT - runs the daily-driver llama-server on :9931. Coexists with every
  phase except the contingency; `pcie_aspm` kernel-param tests still need a whole-box
  reboot (schedule around :9931).
- CPU split for hosted timing fidelity: test server `taskset -c 0-15`, leave 16-23 for :9931.
- RAM: ~55 GB available, moe-cache pools are pinned and non-swappable. Watch
  `MemAvailable` during load; allocation failure/OOM is the risk, not a reset.
- Model: `~/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF` (same shards as the reporter),
  plus the MTP draft. Page-cache sharing with :9931 keeps the extra cost moderate.

## Phase 0 - replay harness (safe, coexists with :9931)

1. Verify `HIP_VISIBLE_DEVICES=0` selects the 6900 XT with the current build.
2. Server flags per the reporter's recipe:
   `-c 262144 -b 2048 -ub 2048 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1`
   `-md MTP/... --spec-type draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0`
   env: `HSA_ENABLE_SDMA=0 GGML_CUDA_MOE_CACHE_RESERVE_MB=1024`
3. Synthetic replay: ~23k-token chat request (system prompt + 20 tool schemas), then an
   unrelated request at each sweep size: 166 / 266 / 346 / 535 tokens.
4. Sampler: 200 ms sysfs/rocm-smi log of sclk, busy, power; AER counters before/after;
   `MemAvailable` in the same log.
5. Success: burst visible in the log at the short request's start (GPU idle -> ramp ->
   sustained busy). Expect no reset on our card; record its baseline behavior anyway.

## Phase 1 - instrument the hosted path (code change, small)

- Log behind an env knob (e.g. `GGML_CUDA_MOE_HOSTED_LOG=N`), per publication:
  layer, rows, resident rows vs CPU rows, merge grid size, CPU-compute duration,
  wait time seen by the merge kernel.
- Use `GGML_CUDA_MOE_TIMELINE` for the host<->GPU relay timings where needed.
- Output: a burst fingerprint per configuration (poller count x poll duration x layers,
  host read volume, clock trajectory during the window).

## Phase 2 - differential fingerprinting (the core of the hunt)

Measure the delta between the reporter's stable and unstable configurations, all
runnable here on healthy hardware. The delta IS the root-cause candidate and the
attenuation target for fixes.

| # | Configuration | Reporter result |
|---|---------------|-----------------|
| 1 | 535-token follow-up, hosted default, full clock | reset |
| 2 | ~166-token follow-up (~1660 rows), full clock | stable |
| 3 | hosted on, `GGML_MOE_HOSTED_MAX_ROWS=256` | stable |
| 4 | `GGML_MOE_GPU=0` (hosted off) | stable |
| 5 | same 2660-row batch with GPU already busy (tail-chunk case) | stable |
| 6 | same at 2200 MHz capped | stable |
| 7 | `-b 9536 -ub 9536` 30k prefill (non-hosted reset path) | reset |

For each pair, the question: what appears in the unstable fingerprint that is absent
from the stable one? Expect: concurrent UC host reads starting mid-clock-ramp. But
configuration 7 may fingerprint differently - keep it in scope.

## Phase 3 - elimination matrix

Split by where the test must run. Bench runs measure pattern + perf cost; reporter runs
are elimination tests at full clock, one variable per build.

- Bench only (no reset expected, safe): `GPU_MAX_HW_QUEUES=2`; queue-count audit of
  Avalon's streams (pool + fill + hosted + card + PR #15 fresh streams vs the
  "Runlist oversubscribed" warning here and on stock llama.cpp).
- Reporter elimination queue (ranked by information per reboot):
  1. Lead mitigation build (from Phase 4) - resolution attempt, 10 replays at 2459 MHz.
  2. `HSA_ENABLE_MMAN_MODE=0` - closes the ROCm mmap hypothesis.
  3. `pcie_aspm=off` / policy=performance - the gap their `iommu=pt` test did not cover.
  4. Coherent-vs-WC mailbox build (if the bench shows a meaningful traffic delta).
  5. `setpci` PCIe Gen1/Gen2 force at full sclk - separates link-PHY from everything else.
  6. `--spec-type none` - fills a gap in their matrix.

## Phase 4 - fix candidates (this branch)

Ranked by how directly they attack the fingerprint delta:

1. **Idle-aware hosted launch**: after an idle GPU period, warm up with a small dispatch
   (or stage the merge grid) before the first big hosted batch.
2. **Poller throttle**: cap concurrent host-flag pollers or add backoff in the merge
   kernel's wait loop so thousands of CTAs don't all hammer the mailbox at once.
3. **Mapping policy**: non-coherent (WC) mailbox/rows with explicit fences if the coherent
   UC pattern shows up as the traffic delta (moe-cache.cu:7378).
4. **Prefill path** (from config 7): if the big-ubatch fingerprint differs, a matching
   mitigation or a documented RDNA2 ubatch cap.
5. **Defaults/docs for RDNA2**: lower `GGML_MOE_HOSTED_MAX_ROWS` on gfx103x, README note
   for `GPU_MAX_HW_QUEUES=2`, queue-count reduction if the audit justifies it.
6. **Side bug**: the `ggml_cuda_moe_hosted_wait` deadlock (GPU 99% + host spin) - request
   the reporter's saved backtrace, reproduce here in a loop.

Bench gate before any candidate goes to the reporter:
- fingerprint attenuates measurably vs the Phase 2 baseline,
- prefill/decode t/s unchanged at full clock,
- output correctness, MTP path exercised.

## Phase 5 - reporter validation loop (reset oracle, used sparingly)

- One variable per build, written procedure, 10-replay pass at full clock per candidate.
- A pass = resolution signal; a fail = one reboot, then the next item on the Phase 3 queue.
- Escalation rule: after TWO failed candidates (two of their reboots spent), trigger the
  contingency below before sending more builds.

## Contingency (NOT a phase) - local reset hunt via voltage cliff

Only if the reporter loop stalls (2 failed candidates) or a fix shows partial effect and
needs fast bisection:
- Negative voltage offsets via `pp_od_clk_voltage` (-50 / -75 / -100 mV) at full sclk on
  the 6900 XT; reporter's card reset 1/1 at -50 mV, so the trick has precedent.
- Requires :9931 stopped; induced reset = whole-machine reboot; keep trees clean, work
  committed. Find our card's cliff, then iterate fixes locally at "just below boost".
- If no cliff by -100 mV: bench is confirmed "a stable sample"; stop, return to the
  differential + oracle loop.

## Phase 6 - resolve

- Comment drafts for #27: differential fingerprint data, test results, fix instructions
  (posted by us as humans, not AI-voiced).
- README/docs for RDNA2 mitigations.
- Upstream split: README note + queue-count tweak are upstream-shaped; hosted-path
  changes are fork-shaped. Upstream rules apply: human-owned design, no AI-written PR
  text, no automated submissions.
- Physical mechanism ("why the card dies") stays an evidence-backed inference; state it
  as such in the issue. That is enough to close it.

## Asks for the reporter

1. The full `ggml_cuda_moe_hosted_wait` backtrace they offered.
2. Whether stock llama.cpp also logs "Runlist is getting oversubscribed" on their box
   (isolates the queue warning to Avalon's stream count; no reboot needed for them).
3. Then the ranked elimination queue as builds arrive (Phase 3).

## Risks / constraints

- 55 GB RAM headroom vs pinned pools: monitor, prefer failure at allocation time.
- CPU contention distorts mailbox timing; always pin with taskset in Phase 0/1/2.
- Reporter's card is ground truth; our card may never reset - the plan assumes fixes are
  validated on their loop, pattern attenuation on ours.
- Their `-b 9536` prefill reset means hosted-only fixes may be incomplete; config 7
  stays in the matrix until shown otherwise.

## Milestones

| M | Deliverable | Est. |
|---|-------------|------|
| M0 | Replay harness + sampler, burst visible on bench | 0.5 d |
| M1 | Instrumented hosted path, per-config burst fingerprints | 1 d |
| M2 | Differential table complete (7 configs), delta identified | 1-2 d |
| M3 | Fix candidates + bench gate (attenuation + no perf cost) | 2-4 d |
| M4 | First reporter build: ranked elimination queue in motion | coordination |
| M5 | Confirmation at 2459 MHz, docs, issue close-out | coordination |
| (X) | Contingency voltage-cliff hunt, only if M4 stalls twice | 1-2 d, gated |
