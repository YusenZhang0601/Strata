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
| Deep hyper-connection read | `ccbab643` (fused_gr.hpp, fused_gr.cu) | `STRATA_HC_DEEP=1` (unset = upstream); shape `STRATA_HC_DEEP_WPB/_PD/_RPW` | Bitwise (self-test against the plain read at init) | Bench: no gain | No; retires from this branch at p2 |
| `hc_deep_bench` | `ccbab643` (CMakeLists.txt, src/kernels/hc_deep_bench.cpp) | Separate bench executable, no switch | n/a, not part of `strata.exe` | Used for the measurement above | No; retires from this branch at p2 |

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
checked bit for bit against the plain read at start-up and was no faster on this A6000. It is compiled into p1 only
because it shares the commit that was built for production. From p2 it stays on `personal/decode-roofline`, where
the negative result remains reproducible.

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

Not yet built or measured. Production keeps p1 until a 0.1.59 build passes every gate in RELEASE.md, with the pins
listed there.

## Maintenance rules

- A new personal patch is its own commit, opt-in, with upstream behavior unchanged when the switch is unset, and
  its row in the table above names the switch, default, numerics class and evidence before it is released.
- When an official release provides an equivalent change, retire the personal patch at that baseline.
- Do not start contribution branches from `personal/stable`; branch them from official `main`, one problem each,
  following the upstream CONTRIBUTING.md.
- Research and measurement-only commits stay on `personal/decode-roofline` or another research branch.
