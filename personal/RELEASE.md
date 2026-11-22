# Building and releasing personal Strata

A release is one `strata.exe`, with the `strata_kq.dll` from the same build, that has passed every gate below on the
target workstation. Nothing reaches production on a successful compile alone. Steps run in order.

The workstation has one GPU, which production uses. The toolchain-parity, kvq and replay gates (§2–§4) therefore
run in one stop window:

- Before stopping, check that `/v1/status` reports `in_flight=0`.
- Stop only the service's own process tree, identified by PID, command line and start time.
- End the window the same way whether the gates pass or fail. Start the current production release again, wait
  until it has loaded, and send a reference request. Then remove the window's temporary tasks and per-arm copies,
  keeping the raw results.

A failed gate in §2–§4 stops the release. Nothing has been deployed at that point, so production simply restarts on
its current release. §6 runs after deployment, so a failure there means [Rollback](#rollback).

## 0. Before building

- Read the official release notes and the engine diff since the current baseline. Record:
  - which upstream changes matter here;
  - which personal patches retire;
  - which defaults changed.

  Update `personal/upstream.json` in the same series.
- Build at the official engine package's BUILD commit, not at the tag. Read it from the release feed
  (`strata-feed.json`, `engine.packages[].build_id`). For 0.1.59 it is `712fec6b4297`; tag `v0.2.13` is four
  release-tooling commits later. `v0.2.13.1` changes no engine source: it is a Control fix, docs and release
  tooling, plus the CMake version string (0.1.59.1).
- Pin each default that changed between the two engines until it has been measured here. The release notes name
  them. For 0.1.56 → 0.1.59 there are three:
  - 0.1.57 raised the cap of a bare `--prefill auto` on NVIDIA to 32768. This certainly changes this configuration,
    so pin `--prefill auto:8192`.
  - 0.1.57 turned on K-quant MMQ on cc ≥ 8.0. Pin `STRATA_MMQ_KQ=0`.
  - 0.1.59 reads Q8_0 hyper-connection projections by default when the GGUF carries them. Pin `STRATA_HC_Q8=0`.

  Whether the last two do anything depends on the pack's tensor types: packs without K-quant experts never read
  the MMQ switch. Keep them pinned anyway, together with `STRATA_NV_CHUNK=0` as a second guard on the prefill cap.
  The start-up log must not say `NVIDIA: --prefill auto may take chunks up to`.

## 1. Build

| Item | Value used for p1 |
|---|---|
| Host | The target workstation itself (Windows, native) |
| CUDA | 13.0 (`nvcc` from a pinned toolkit directory, `CUDA_PATH` pointing at it) |
| Host compiler | MSVC 19.37 (Visual Studio 2022 `vcvars64.bat`) |
| Generator | Ninja |
| ggml | llama.cpp checkout at the commit in `third_party/ggml/VERSION.txt` (`3cf03257f219`), passed as `STRATA_GGML_DIR` |
| Target | `strata` (produces `strata.exe` and `strata_kq.dll`) |

From 0.1.59 the tree also vendors `third_party/strata-ggml`, which becomes the default when `STRATA_GGML_DIR` is not
given. Upstream keeps the override for two releases. Use the same ggml source as the official build of that engine;
the toolchain-parity gate (§2) is what proves the choice.

Move the source to the build host as a `git bundle`. On the build host, before building:

- run `git bundle verify` and compare the bundle's SHA-256 with the one computed where it was made;
- check `git rev-parse HEAD` against the intended full SHA;
- require `git status --porcelain --untracked-files=all` to be empty.

Configure and build:

```text
cmake -S <src> -B <build> -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
      -DCUDAToolkit_ROOT=<cuda13> -DCMAKE_CUDA_COMPILER=<cuda13>/bin/nvcc.exe -DCMAKE_CUDA_ARCHITECTURES=86
      -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=<llama.cpp checkout>
cmake --build <build> --target strata --config Release -j8
```

The workstation is shared. Build at below-normal priority on a CPU subset, with `CUDA_VISIBLE_DEVICES=-1` so the
build never touches the GPU that serves production. Compiling does not need a stop window.

Write a build receipt next to the outputs with:

- the full source SHA and the empty status output;
- full toolchain versions (`nvcc --version`, the `cl` banner, CMake, Ninja);
- the configure line;
- the ggml source and commit;
- each output's size and SHA-256.

## 2. Toolchain parity (once per new baseline)

Build the clean official base commit with the same toolchain. Run kvq on it and on the official binary of that
engine version. They must be bitwise identical. If they are not, stop: later gates cannot tell patch effects from
toolchain effects. For 0.1.56 the clean build and the official binary were identical on all 24,574 lines.

## 3. kvq numerics gate

kvq is a teacher-forced comparison built on the upstream `STRATA_LOGPOS` / `STRATA_LOGPOS_TOPK` hooks. It reads a
code segment and a document segment of 12,288 tokens each, which gives 12,287 scored positions per segment and
24,574 lines in total. Every line records the top-64 log-probabilities. Runs use fixed conditions:

- adaptation frozen;
- the same expert profile and cache size;
- `--pcie-frac 0` and `--prefill 512`;
- int8 KV;
- the production environment plus the switches under test.

The harness is a local adaptation of upstream `tools/quant_quality.py` and is not yet in this repository.

- **Bitwise patches**: run the candidate with exactly the deployed switch set and compare it with the clean base
  build. Pass: 24,574/24,574 identical lines. p1 ran a superset of the deployed switches.
- **Reorder-level patches**: report top-1 agreement, mean and max KL, and ΔNLL per context bucket. The reference for
  "acceptable" is the int8-KV-vs-fp16 difference on the same segments. Enabling one needs an explicit decision.
- **New baselines**: also compare the candidate against the current production release. This is not expected to be
  bitwise, because upstream numerics change, for example through correctness fixes. Report the drift and attribute
  it, so it is not mistaken for a patch effect.

## 4. Replay A/B gate

Replay the recorded real-load scenario against copies of the production configuration on a test port, in the order
old, new, old, new. The scenario has:

- two long cold reads (about 100K and 64K tokens);
- same-conversation continuation, a conversation switch, a short side conversation and a switch back;
- a four-way burst.

Compare:

- decode window time (`gen_ms / (generated − accepted)`), grouped by window size where possible, because MTP
  acceptance varies with the generated text;
- measured-segment read time;
- park count and time;
- four-way burst time;
- the cold 100K-token read.

Pass when no item regresses beyond the noise of the old-vs-old pair, and the change claimed for the release
reproduces in both pairs. A regression beyond noise needs an explicit, recorded decision to accept it. p1 accepted
a 1% slower cold 100K-token read as the cost of `--pcie-frac 0`.

Noise on this machine: about ±0.2 ms per window at 100K, ±0.3–0.5 ms at 167K. Tokens per second cannot separate
configurations here, because the generated text differs between runs.

## 5. Deploy

1. Copy the outputs, verified by SHA-256, into a new directory named `engine-<engine>-personal-<short commit>`. Never
   overwrite a directory that production has used.
2. Decide which `strata-vision.exe` and which CUDA runtime DLLs (cuBLAS, cudart) production loads with the new
   engine. Record each one's path and SHA-256. Compare them with the official package of the same engine version,
   whose `BUILD.json` lists what it was built with.
3. Write `PROVENANCE.json` in the engine directory (schema below).
4. Back up the production configuration to a dated recovery directory. Then change `exe`, plus any switches and
   pins.
5. Stop the service as above, start it, wait for it to load (about 140 s), and send one reference request.

`PROVENANCE.json`, build receipts and binaries stay on the workstation. They are not committed to this repository or
attached to a GitHub release.

## 6. Acceptance gate

Run the acceptance suite against production:

- status;
- chat, messages and responses tool calls, plus `previous_response_id`;
- history tools and deferred-loading tools;
- closing tags inside file contents;
- vision;
- four-way concurrency.

Pass: 10/10. On Windows set `PYTHONIOENCODING=utf-8`, so a console code page cannot crash the run.

## 7. Release: tag, join and push

Never `git checkout` in a clone that other sessions share; do this in a separate worktree. `B` is the full SHA that
was built, and `S` is the current `personal/stable`.

```sh
B=<full sha of the built commit>; S=$(git rev-parse personal/stable)
git worktree add ../strata-join "$B" --detach && cd ../strata-join
git checkout "$S" -- personal/                     # start from the latest notes on personal/stable
# edit personal/README.md (release row, patch table) and personal/upstream.json for this release
git commit -m "personal: notes for personal-<engine>-p<N>"
D=$(git rev-parse HEAD)
git diff --exit-code "$B" "$D" -- . ':(exclude)personal'    # the notes commit changes no code
M=$(git commit-tree "$D^{tree}" -p "$D" -p "$S" -m "personal: join the <engine> series (tree = new series; previous personal/stable is the second parent)")
git merge-base --is-ancestor "$S" "$M" && git diff --exit-code "$D" "$M"
git update-ref refs/heads/personal/stable "$M" "$S"   # fails if personal/stable moved meanwhile
git tag -a personal-<engine>-p<N> "$B" -F <message file>
cd - && git worktree remove ../strata-join
```

For a release on the same baseline, such as a new patch on `personal/stable`, skip the join. Commit on top of
`personal/stable` in the worktree and fast-forward it.

Before pushing, read the full diff and every new commit and tag message:

- no local paths, host names, account names, addresses or credentials;
- the GitHub account in use is the fork owner's.

Then push without `-f` and read the remote back:

```sh
git push fork personal/stable personal-<engine>-p<N>
git ls-remote fork refs/heads/personal/stable 'refs/tags/personal-<engine>-p<N>*'   # must equal $M and the tag
```

To extract the series for the next baseline, list the commits that change code:

```sh
for c in $(git rev-list --reverse --first-parent --no-merges <base_commit>..personal/stable); do
  git diff-tree --no-commit-id --name-only -r "$c" | grep -qv '^personal/' && echo "$c"
done
```

Cherry-pick those onto the new baseline commit in a fresh `drill/<tag>` worktree. The `personal/` notes are
regenerated at the next join.

## PROVENANCE.json

```json
{
  "schema": 2,
  "release": "personal-0.1.59-p2",
  "engine_version": "0.1.59 + personal",
  "repository": "https://github.com/YusenZhang0601/Strata",
  "built_on": "drill/v0.2.13",
  "released_on": "personal/stable",
  "commit": "<full sha of the built commit>",
  "upstream": {"repository": "Niko1221/Strata", "base_commit": "<full sha>", "tag": "<tag>"},
  "build": {"cuda": "<nvcc full version>", "msvc": "<cl version>", "arch": "sm_86", "generator": "Ninja",
            "flags": "-DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF", "ggml": "<source and commit>",
            "receipt": "<build receipt file>"},
  "artifacts": {"strata.exe": "<sha256>", "strata_kq.dll": "<sha256>"},
  "vision": {"path": "<strata-vision.exe used>", "sha256": "<sha256>", "origin": "<official package or build>"},
  "runtime_dlls": [{"name": "<dll>", "path": "<where it loads from>", "sha256": "<sha256>"}],
  "switches": [{"name": "STRATA_PCIE_SHARE_NODES", "value": "0", "origin": "personal"},
               {"name": "STRATA_TOPK_SPLIT", "value": "1", "origin": "upstream"}],
  "pins": ["--prefill auto:8192", "STRATA_NV_CHUNK=0", "STRATA_MMQ_KQ=0", "STRATA_HC_Q8=0"],
  "validation": {"toolchain_parity": "<result>", "kvq": "<result>", "replay_ab": "<result>",
                 "acceptance": "<n>/10", "accepted_regressions": "<none or what was accepted>", "dates": "<when>"},
  "rollback": {"config_backup": "<path>", "previous_engine": "<directory>"}
}
```

The p1 file predates this schema. Its `branch` and "not pushed" fields are stale: the source was pushed after the
deployment, and the release now lives on `personal/stable`. Its commit, hashes and toolchain are correct. From
`personal-0.1.56-p2` onward, releases write schema 2 and keep all new optimization switches opt-in (default OFF) while
enforcing the release guardrails listed in `personal/README.md` (leave `STRATA_BATCH_MTP_ATTN_SLOTS` unset, keep
`STRATA_PREFILL_SPARSE_CHECK=0`, and pair `STRATA_CONV_CACHE_NUMA_NODE=0` with `STRATA_CONV_CACHE_TTL_S` and bounded
conversation cache slots).

## Rollback

Every release keeps its predecessor deployable. Restore the backed-up configuration, which points `exe` at the
previous directory and restores its switches and pins. Then stop the service process tree as above, start it, and
run the reference request and the acceptance suite. Keep at least one verified rollback point when cleaning up old
engine directories.
