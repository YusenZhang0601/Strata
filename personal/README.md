# Personal Strata

This branch carries an official Strata engine baseline plus a short list of personal patches tuned for one machine:
a shared RTX A6000 (48 GB, sm_86) workstation running Windows natively, on PCIe Gen3, with two EPYC sockets. Every
measurement below comes from that one machine and does not claim anything for other hardware. The repository is
`YusenZhang0601/Strata` and the release branch is `personal/stable`. The official baseline is recorded in
[`upstream.json`](upstream.json). Build, validation and release steps are in [RELEASE.md](RELEASE.md).

From p2 on, every personal patch is a separate commit, opt-in through an environment switch, and leaves upstream
behavior unchanged when the switch is unset. p1 predates this rule: its one commit `ccbab643` carries the PCIe
patch, a deep-read prototype and that prototype's bench. General improvements are offered back to
`Niko1221/Strata` from separate contribution branches.

## Branches

| Branch | Content | Moves when |
|---|---|---|
| `personal/stable` | Official baseline + personal patches + this `personal/` directory | A release passes every gate in RELEASE.md |
| `personal/decode-roofline` | Research history: prototypes, bench tools and measurement-only commits | Research work. Not built for production after p1; p1's `ccbab643` is on both branches |
| `drill/<tag>` (local) | The patch series replayed on a new baseline commit; `<tag>` only names the release | Rehearsing an upgrade; joins `personal/stable` only after the gates pass |
| `contrib/<area>-<topic>` | One upstream contribution, branched from official `main` | Preparing or updating one upstream issue/PR |
| `main` | Untouched mirror of official `main` | Only by syncing with upstream |

`personal/stable` is never force-pushed or rebased. A new baseline works in three steps:

1. Replay the code patches on the new baseline commit recorded in `upstream.json`. That is the BUILD commit of
   the official engine package, or the tag when no BUILD commit is known.
2. Regenerate `personal/` from the latest `personal/stable`.
3. Join the result to the previous `personal/stable` with a merge whose tree is the new series and whose second
   parent is the old branch.

The branch only fast-forwards, and every commit ever built for production stays reachable. The current series is
`git log --first-parent --no-merges <base_commit>..personal/stable`. To extract it for the next replay, cherry-pick
`git rev-list --reverse --first-parent --no-merges <base_commit>..personal/stable`, skipping the commits that only
touch `personal/`. A plain `git rebase`, or GitHub's commit list, also walks the older series through the merge's
second parent. The exact commands are in RELEASE.md. Each production build is tagged `personal-<engine>-p<N>` at
the exact commit it was built from.

Official releases are reviewed by hand rather than merged automatically, because the GPU gates (kvq bitwise
comparison and replay A/B) only run on the target workstation. The cadence is: read the release, decide which
changes are useful here, rehearse the replay, run the gates, then release.

## Baseline

Engine 0.1.56, base commit `06698c674673`, the BUILD commit of the official 0.1.56 CUDA 13 package. Tag `v0.2.10`
contains it 69 commits later; the CUDA `strata` target's sources are identical in that range. A clean sm_86 build
of the base commit was kvq-bitwise identical to the official 0.1.56 binary, which is consistent with a matching
toolchain.

## Patches

Numerics column: **bitwise** means the kvq teacher-forced comparison is identical on every line under fixed
conditions. kvq reads a code segment and a document segment of 12,288 tokens each, which gives 12,287 scored
positions per segment and 24,574 lines in total, with the top-64 log-probabilities at every position.
**Reorder-level** means the summation order changes, so tokens can diverge; such a patch needs a quality comparison
(top-1 agreement, KL, ΔNLL) and an explicit decision.

| Patch | Commit | Switch (default) | Numerics | Evidence (this A6000) | Production |
|---|---|---|---|---|---|
| PCIe share nodes off | `ccbab643` (expert_source.hpp/.cpp, verify.cpp) | `STRATA_PCIE_SHARE_NODES=0` (unset = upstream) | Bitwise relative to `--pcie-frac 0` | kvq 24,574/24,574 identical, run inside a 7-switch bitwise stack that includes both deployed switches | Yes, since 2026-11-13, with `--pcie-frac 0` |
| Deep hyper-connection read | `ccbab643` (fused_gr.hpp, fused_gr.cu) | `STRATA_HC_DEEP=1` (unset = upstream); shape `STRATA_HC_DEEP_WPB/_PD/_RPW` | Bitwise (self-test against the plain read at init) | Bench: no gain | No; retires on next baseline |
| `hc_deep_bench` | `ccbab643` (CMakeLists.txt, src/kernels/hc_deep_bench.cpp) | Separate bench executable, no switch | n/a, not part of `strata.exe` | Used for the measurement above | No; retires on next baseline |
| Batched QSA stream + pre75 slots + batch side mixer (K1) | `46763c79`, `9294ab7e`, `88dd097f` (qsa_decode_attn, qsa_select, kv_stream, verify.cpp) | `STRATA_BATCH_QSA_STREAM_OPT=1`, `STRATA_ATTN_PRE75_SLOTS=1`, `STRATA_BATCH_SIDE_MIXER=1` (all unset = upstream) | Kernel bitwise parity (`qsa_slots_parity` 16/16 bitwise identical across 4 KV formats × 1/2/4/8 slots; 24,574-line `kvq` not re-run) | 4×100K decode window −5.88 ms/win (−15.1%), 0 MiB extra VRAM; `aplus_accept` 10/10 | Yes, since `personal-0.1.56-p2` |
| Multi-slot batched MTP drafting (S1-L2) | `6e03e456`, `cd883219` (mtp.hpp/.cpp, generate.cpp) | `STRATA_BATCH_MTP_FUSED=1` (unset = upstream); optional `STRATA_BATCH_MTP_ATTN_SLOTS` (keep unset for automatic ≤32 MiB guard) | Per-row draft arithmetic parity (drafts only; target verifier decides all emitted tokens; `kvq` not re-run) | 4×100K `emit_ms` 2.92 → 1.85 ms (−36.6%), 0.53 MiB staging VRAM, zero `gpu_wait` spillover; `aplus_accept` 10/10 | Yes, since `personal-0.1.56-p2` |
| Conversation cache + retained reuse idle TTL (S3A) | `387b858c`, `3ef74876` (conversation_cache.hpp, conversation_cache_test.cpp, generate.cpp) | `STRATA_CONV_CACHE_TTL_S=<sec>` (unset/0 = upstream) | n/a (host cache lifecycle only) | `conversation_cache_test` 4,877 checks passed; live TTL drains both parked entries and retained `reuse_` K/V to 0 B | Yes, since `personal-0.1.56-p2` (`1800`) |
| Exact VRAM park staging + NUMA0 allocator + parallel host memcpy (S3B) | `7b9e341d`, `92168701`, `32cc9e3a` (memory, conversation_buffer/snapshot/state, CMakeLists.txt) | `STRATA_PARK_EXACT_STAGE_VRAM=1`, `STRATA_CONV_CACHE_NUMA_NODE=0`, `STRATA_PARK_HOST_MEMCPY=1` (all unset = upstream; optional `STRATA_PARK_COPY_THREADS`, `STRATA_PARK_NUMA0_MIN_FREE_MIB`) | Continuation bitwise parity (100K cold park → restore continuation 100% SHA-256 identical; `kvq` not re-run) | `VirtualAllocExNuma` self-check `on_node=256 off_node=0`; 100K cold park 1,120 → 186 ms (6.0×), warm park 126.5 → 15.7 ms (8.1×), restore 298 → 184 ms; fixes 9× VRAM over-reservation (+306 MiB with async park/ckpt vs ~1.6 GiB) | Yes, since `personal-0.1.56-p2` (with `STRATA_PARK_ASYNC=1`, `STRATA_CKPT_ASYNC=1`) |
| Fast stdin IPC + differential token IPC + PUA regex coalescing (S2A-1 + S2A-2) | `ea8c8e6d`, `25ee3800`, `1a85d83c`, `085058b7` (generate.cpp, serve/frontend.py, serve/server.py, serve/test_ipc_delta.py) | `S2A-1` engine stdin/BGEN parser always active (lossless format-compatible); `STRATA_IPC_DELTA=1` (unset = upstream) | Exact token-ID parity (verified by 64-bit CRC32+Adler32 prefix digest + `REF_MISS` auto-recovery; `kvq` not re-run) | 167K host overhead 135.06 → 62.92 ms (−53.4%; S2A-1 −27.4 ms alone + S2A-2 −38.1 ms on top); `test_ipc_delta.py` 5/5 + real pipe `REF_MISS` recovery PASS | Yes, engine `S2A-1` since `personal-0.1.56-p2`; `STRATA_IPC_DELTA=1` when matched serve layer is deployed |
| Selection-first sparse on-demand KV + small-chunk dedicated prefill workspace (K3) | `90f75738`, `d09afc71`, `94ef9135`, `8b6047bb`, `1e798b2a`, `a1c2c904` (qsa_prompt_attn.cu, kv_stream, prefill.hpp/.cpp, generate.cpp) | `STRATA_PREFILL_SPARSE_KV=1` (default `STRATA_PREFILL_SPARSE_MAX_T=256`), `STRATA_PREFILL_WIN_SMALL_OWN=1` (all unset = upstream; diagnostic `STRATA_PREFILL_SPARSE_CHECK=0`) | Kernel/continuation bitwise parity against full staging on matched selections (100K parity + `res4096` `ovf=0` + `aplus_accept` 10/10; 24,574-line `kvq` not re-run) | 100K hot incremental prefill `T=155` 941 → 727 ms (−22.8%), 0 cache slots lent for chunks ≤ 768 tok (`T_cross ≈ 350–380 tok`); full-stack `mem_free = 4,703 MiB` | Yes, since `personal-0.1.56-p2` |

### Rejected / excluded experiment

- **Async commit overlap (`STRATA_BATCH_COMMIT_ASYNC`, S1-L1):** tested in isolation on this A6000 and excluded from `personal/stable`. Overlapping `commit_bm_` with 4× independent per-slot MTP passes caused GDDR6 bandwidth/TDP contention and WDDM tail spillover (`commit` dropped −1.75 ms, `emit` rose +1.33 to +1.75 ms, and `gpu_wait` regressed +0.54 to +0.75 ms). Replaced by `STRATA_BATCH_MTP_FUSED=1` (S1-L2) with synchronous `commit_bm_`.

### Operational guardrails for `personal-0.1.56-p2`

1. **Leave `STRATA_BATCH_MTP_ATTN_SLOTS` unset in production** (uses the automatic `<= 32 MiB` staging guard; forcing `=1` with `window == 0` at 262K context would allocate 387–580 MiB).
2. **Keep `STRATA_PREFILL_SPARSE_CHECK=0` (unset) in production** (`=1` adds ~480 per-layer host syncs for overflow debugging only).
3. **Pair `STRATA_CONV_CACHE_NUMA_NODE=0` with `STRATA_CONV_CACHE_TTL_S=1800` and bounded `--conversation-cache-slots` / `--conversation-cache-mib`** so parked 100K conversations keep comfortable NUMA0 headroom.

**PCIe share nodes off.** With `STRATA_PCIE_SHARE_NODES=0` the host plan never assigns a PCIe share: every expert
miss goes to the CPU pool, as with `--pcie-frac 0`, whatever the request, the automatic share or
`STRATA_PCIE_BALANCE` say. The verify window then leaves out the share's five empty launches per layer: the flag-B
wait, blob fetch, pointer rebase, and the share's grouped gate/up and down. The window keeps them when the device
plans the group, when PCIe balancing is on, or with remote experts. Compared with a run that does assign a PCIe
share, the rare positions that share would have served are computed on the CPU instead. In the measurement before
the change, that was 7 of 313,920 expert selections.

Side effects and limits:

- With the switch set, a request's `pcie_frac`, a `--calibrate` share, `STRATA_PCIE_BALANCE` and (from engine 0.1.57
  on) `STRATA_ADAPT_FETCH` have no effect, and no log line says so. Use the switch only together with
  `--pcie-frac 0`.
- Upstream once skipped the same block for the whole session at `--pcie-frac 0` (PR #109, `4a50d4ce`). Merge
  `1977f595` left that out, because a request can still ask for a PCIe share and the window graph is captured once.
  This patch closes the share at the host plan instead, whatever the request asks.
- Known gap in p1: the host plan reads residency twice, once to count misses and once to assign them. If another
  thread marked an expert non-resident between the two reads, one share could still be assigned while the window
  skips its launches. No such writer was found on the production path; the pipelined concurrent prefill
  (`STRATA_PIPED_CONCURRENT`) is off. A planned hardening adds an explicit `!pcie_share_nodes_off()` to the
  assignment condition, so a late miss stays on the CPU. With the switch unset that path is unchanged from
  upstream.

**Deep hyper-connection read.** A prototype with a deeper weight prefetch and several down rows per warp. It is
checked bit for bit against the plain read at start-up and was no faster on this A6000. It is compiled into p1/p2
only because both share the 0.1.56 series commit `ccbab643`. From the next baseline it stays on
`personal/decode-roofline`, where the negative result remains reproducible.

### Upstream switches validated with these patches

These are upstream options, not patches. The release gates were run with them, and they affect the measured result.

| Setting | Numerics | Measured on this A6000 |
|---|---|---|
| `STRATA_TOPK_SPLIT=1` | Bitwise (top-k self-test identical; kvq inside the same 7-switch stack) | −2.7% / −3.4% per decode window at 100K / 167K context |
| `--pcie-frac 0` | Not compared by kvq, which always runs at `--pcie-frac 0`. Against the default share, the rare PCIe-served experts move to the CPU and may round differently | No decode change on its own. Cold 100K-token read about 1% slower in the combined replay (attributed to this setting, not isolated) |

`STRATA_HC_PERSIST=1` (upstream) would save about 1.8% more at 100K (1.2% at 167K, within noise). It is
reorder-level: document-segment top-1 agreement 90.6–93.6%, mean KL 0.04–0.19 (max up to 19), ΔNLL ≤ 0. It is
not enabled.

## Releases

| Release | Engine | Built from | `strata.exe` SHA-256 | `strata_kq.dll` SHA-256 | Gates | Deployed |
|---|---|---|---|---|---|---|
| `personal-0.1.56-p1` | 0.1.56 | `ccbab643d051` | `ae0beb3d1999207482b7767d88c724254322aba0c3dc476162d83f79232c3254` | `cf0b25f7df4401d5f01aebd328569b80934266b5f719588b37aeb645998326ff` | See below | 2026-11-13 |
| `personal-0.1.56-p2` | 0.1.56 | `a1c2c904139d` | `9ca1348a2ea895294b80cf0ee882a50273f31c16c5feaffd903d5c5c502baa17` | `a95954c3a626122c3590b3748c467e268edf752375b344fccaee73657f9b62f8` | See below | 2026-11-22 |

p2's gates (built from code commit `a1c2c904139d5c6a028750141d7a75f35253b76a`; the subsequent docs-only commit changes only `personal/` so the compiled binary SHA-256 is unchanged; the 24,574-line `kvq` harness was not re-run for p2—numerics were gated via kernel/continuation bitwise parity checks and the 10/10 acceptance suite below):

- **Native unit & parity tests:** `qsa_slots_parity.exe` (`STRATA_ATTN_DAILY=1 STRATA_ATTN_PRE75_SLOTS=1`) 16/16 bitwise PASS; `conversation_cache_test.exe` (`STRATA_CONV_CACHE_NUMA_NODE=0`) `on_node=256 off_node=0 fell_back=0`, 4,877 checks PASS; 100K cold park → restore continuation SHA-256 bitwise match; 100K WDDM sparse-vs-staging continuation byte-exact parity + `res4096` `ovf=0`; `serve/test_ipc_delta.py` 5/5 PASS plus real Windows pipe `REF_MISS` auto-recovery PASS.
- **Full-stack D1–D6 ABAB evaluation (ALL OFF vs ALL 6 ON on RTX A6000, 0 thermal throttle, 0 CPU interference):**
  - **D1 (4×100K concurrent decode, matched 918 tok / 5.94–6.00 rows):** decode window `41.92 → 34.32 ms` (−7.60 ms/win, −18.1%; `gpu_wait` `32.96 → 27.20 ms`, `emit` `2.90 → 1.92 ms`, `commit` unchanged `1.82 vs 1.83 ms`, slope `5.90 → 4.66 ms/row`); end-to-end 4×100K burst wall clock (admission + park/restore + incremental prefill + IPC + decode) `20.55 → 11.92 s` (−42.0%, 1.72×).
  - **D2 (single-lane 100K decode):** zero regression (`3.85 vs 3.90 ms/row`).
  - **D3 (100K hot incremental prefill):** `T=155` `941.0 → 726.6 ms` (−22.8%), 0 cache slots lent across all tested ≤ 768 tok chunks (`T_cross ≈ 350–380 tok`, default `STRATA_PREFILL_SPARSE_MAX_T=256`).
  - **D4 (100K cold/warm park & restore):** cold park `1,056 → 185 ms` (−82.5%, 5.7×), `resize` `700 → 0.28–0.52 ms`, restore `273 → 238 ms`.
  - **D5 (host overhead, S2A-1 + S2A-2):** 167K `135.06 → 62.92 ms` (−72.14 ms, −53.4%); 100K `88.18 → 54.90 ms` (−37.7%).
  - **D6 (VRAM floor & acceptance):** steady-state `mem_free = 4,703 / 4,723 MiB` (≥ 2,048 MiB floor); acceptance suite **10/10 PASS**.

p1's gates:

- **kvq:** bitwise, inside the 7-switch stack.
- **Replay:** two ABAB pairs against the official 0.1.56 binary with its default PCIe share. This build ran with
  `STRATA_PCIE_SHARE_NODES=0`, `STRATA_TOPK_SPLIT=1` and `--pcie-frac 0` together.
  - Decode window −4.7% (pairs −4.3% / −5.1%).
  - Reads, parks and the four-way burst within noise.
  - Cold 100K-token read +1%, beyond noise. This was accepted as the cost of `--pcie-frac 0`.
- **Acceptance:** suite 10/10.

The p1 build ran on the working tree at `06698c674673` with the patch applied, not on a checkout of `ccbab643`. Its
build receipt records `sha256(git diff HEAD)` =
`34036332483bd527ba24bd5b0e12fc8afd20f47f3d7b0c0a3e589cb268189793`. Running
`git diff --abbrev=7 06698c67 ccbab643` over the six tracked files that build saw hashes to the same value: CMakeLists.txt
plus five sources. `hc_deep_bench.cpp` was untracked in that tree and is not part of `strata.exe`.

## Next baseline

Engine 0.1.59: BUILD `712fec6b4297` of the official Windows CUDA 13 package; tag `v0.2.13` adds only release tooling.
A rehearsal replays the PCIe share patch alone, and it applies without conflicts.

- Every launch the patch skips is unchanged in 0.1.59.
- The PCIe share's only host writer is still gated by the switch.
- The deep hyper-connection read and its bench stay on `personal/decode-roofline`. Upstream reshaped `fused_gr.cu`
  nine times in that range, and the prototype had no gain.

Not yet built or measured. Production keeps p2 (0.1.56) until a 0.1.59 build passes every gate in RELEASE.md, with the pins
listed there.

## Maintenance rules

- A new personal patch is its own commit, opt-in, with upstream behavior unchanged when the switch is unset, and
  its row in the table above names the switch, default, numerics class and evidence before it is released.
- When an official release provides an equivalent change, retire the personal patch at that baseline.
- Do not start contribution branches from `personal/stable`; branch them from official `main`, one problem each,
  following the upstream CONTRIBUTING.md.
- Research and measurement-only commits stay on `personal/decode-roofline` or another research branch.
