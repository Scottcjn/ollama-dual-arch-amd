# Running one model across an MI50 (gfx906) **and** a modern RDNA4 card (gfx1200) — combined VRAM, one runtime

**TL;DR:** AMD dropped `gfx906` (MI50 / MI60 / Radeon VII) from ROCm after 6.3, and ollama dropped it after 0.12.3. Existing community fixes get a *lone* MI50 running again — but they hard-code `gfx906` only and force `HSA_OVERRIDE_GFX_VERSION=9.0.6`, which **excludes any modern card in the same box**. This repo generalizes that to **multi-arch**: build one ollama runtime that speaks `gfx906` *and* `gfx1200` at once, so an MI50 (16 GB HBM2) and an RX 9060 XT (16 GB GDDR6) act as **~32 GB of combined VRAM** and split a single 32B model across both — **verified at 16.6 tok/s, 100% on GPU** (see [Benchmark](#benchmark)).

Built on the shoulders of [xxDoman](https://github.com/xxDoman) (ROCm-7.1-runtime + ROCm-6.3-Tensile approach) and [MTLoser / Dawson Reed](https://github.com/MTLoser/ollama-mi50-rocm71-build) (the `v0.18.2` source build + SOLVE_TRI patch). The change here is small and additive — see [The multi-arch diff](#the-multi-arch-diff). `build_dual_arch.sh` runs the whole thing.

---

## Why this is non-obvious (the two walls)

**Wall 1 — the kernel won't even see the RDNA4 card.** On Ubuntu 24.04's stock kernel (6.8), `amdgpu` has no device entry for Navi 44 (RX 9060 XT, `1002:7590`). `lspci -k` shows **"NO DRIVER BOUND"** on the 9060 while the MI50 binds fine. Fix: install AMD's **`amdgpu-dkms` from ROCm 6.4.2** (kernel module only — `--usecase=dkms`, *not* the userspace, so your gfx906 stack is untouched) and reboot. That module (`6.12.12`) knows Navi 44 and binds **both** cards.

```bash
wget https://repo.radeon.com/amdgpu-install/6.4.2/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt install -y ./amdgpu-install_6.4.60402-1_all.deb && sudo apt update
sudo amdgpu-install --usecase=dkms -y     # kernel module ONLY
sudo reboot
# after: both cards show in `rocm-smi`, both bound to amdgpu in `lspci -k`
```

> **Multi-GPU boot flag: `iommu=pt`.** AMD's ROCm system-requirements docs say systems with multiple GPUs *may* need `iommu=pt` on the kernel command line to avoid application hangs. A mixed-arch box is at least as exposed as a homogeneous one. If you see hangs once both cards are in play, add `iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, `update-grub`, reboot, and confirm with `grep -o 'iommu=pt' /proc/cmdline`. Our test box ran without it; we have not yet checked whether it was set (see [Measurements still owed](#measurements-still-owed)).

**Wall 2 — stock ollama can't run both archs in one process.**
- `ollama 0.12.3` bundles a ROCm with native `gfx906` → the MI50 computes clean, but it has **no** `gfx1200`, so the 9060 is invisible.
- `ollama 0.32.5` bundles ROCm 7.2 with native `gfx1200` → the 9060 works, but `gfx906` was removed. Grafting ROCm-6.3 `gfx906` Tensile files into it fixes *enumeration* but **segfaults on the first gfx906 GEMM** — ROCm 7's HIP compiler generates broken device code for one kernel (`SOLVE_TRI`) on gfx906.

So there is **no stock ollama** that drives both. You must build from source, and the build must target **both** archs.

---

## The multi-arch diff

Starting from MTLoser's `build-ollama.sh` and runtime `Dockerfile`, the edits that turn a gfx906-only build into a gfx906 + gfx1200 build:

```diff
# build-ollama.sh — compile for BOTH architectures
-    -DAMDGPU_TARGETS="gfx906" \
+    -DAMDGPU_TARGETS="gfx906;gfx1200" \
```
```diff
# Dockerfile (runtime) — DROP the global gfx906 override (it would force the 9060 -> 906),
# and spread layers across both GPUs by default
-    HSA_OVERRIDE_GFX_VERSION=9.0.6
+    OLLAMA_SCHED_SPREAD=1
```

The **SOLVE_TRI patch stays** (it forces that one op to CPU fallback for *all* archs — harmless, rarely hit at inference; at 8K ctx generation is 100% GPU):
```bash
sed -i '/case GGML_OP_SOLVE_TRI:/s/$/\n            return false; \/\/ fall back to CPU/' \
    ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu
```

The ROCm 7.1 runtime already ships `gfx1200` Tensile; you overlay the ROCm-6.3 `gfx906` Tensile on top (`COPY 906/ /opt/rocm/lib/rocblas/library/`). Result: one `rocblas` with **both** archs.

> Gotcha found in the wild: MTLoser's `Dockerfile.build` COPYs `amdgpu-install_6.3.70100-1_all.deb`, which no longer exists upstream. The real ROCm 7.1 installer is `amdgpu-install_7.1.70100-1_all.deb` (same `70100` build) — alias it to the expected name, or fix the COPY line. `build_dual_arch.sh` handles this.

---

## Build & run

```bash
./build_dual_arch.sh          # clones MTLoser base, applies multi-arch patches, builds image `ollama-dual`
# other pairs: ARCHS="gfx906;gfx1100" ./build_dual_arch.sh   (untested beyond the default, see below)

docker run -d --name ollama-dual \
    --device=/dev/kfd --device-cgroup-rule='c 226:* rmw' -v /dev/dri:/dev/dri \
    -v ollama_models:/models -e OLLAMA_MODELS=/models \
    -e OLLAMA_SCHED_SPREAD=1 \
    -p 11434:11434 ollama-dual
```

`ollama` reports both: `compute=gfx906 (MI50, 16 GiB)` **and** `compute=gfx1200 (RX 9060 XT, 15.9 GiB)`, and `ollama ps` shows the model split across both GPUs. `SCHED_SPREAD` (not `HSA_OVERRIDE`) is the key — each card uses its native arch.

---

## Benchmark

Real numbers from ollama's own eval counters (not synthetic). **qwen2.5:32B Q4_K_M (~20 GB)**, split across MI50 (gfx906) + RX 9060 XT (gfx1200), 8K context, **100% on GPU**:

| Metric | Result |
|---|---|
| **Generation** | **16.6 tok/s** (128 tok / 7.69 s) |
| **Prompt eval** | **174.5 tok/s** (48 tok / 0.28 s) |
| On GPU | **100%** — all 65 layers, no CPU spill |
| VRAM per card | 11.7 GB (9060) + 11.7 GB (MI50) — balanced to the byte |
| Model load | 17.6 s (one-time) |
| **2 concurrent requests** | **33.2 tok/s aggregate** (16.6 + 16.6; wall 14.8 s vs 23.6 s serial) |

Context for the number: a single 24 GB card runs 32B Q4 at ~30–40 t/s; a V100-32 ~20–25. Two 16 GB budget/salvage cards splitting a model over PCIe and landing at **~17 t/s** is respectable — and it runs a model that fits on **neither** card alone. The heterogeneous PCIe-boundary handoff is the real cost (as expected); the `SOLVE_TRI→CPU` patch is **not** in the inference path, so it costs nothing at generation time (verified: 100% GPU at 8K ctx).

**Concurrency is where the second card earns its keep.** A layer split is serial for a *single* request — while card 0 runs the early layers, card 1 is idle (it's acting as a RAM coffer, not a co-processor). With **two requests in flight**, the pipeline overlaps: card 0 works request A's early layers while card 1 works request B's late layers. Measured: 2× throughput, near-perfectly. If your workload is batchable (corpus processing, multi-user), set `OLLAMA_NUM_PARALLEL=2` and you get both cards actually crunching.

---

## Hardware this was done on

| Card | Arch | VRAM | Role in split |
|------|------|------|---------------|
| Radeon Instinct **MI50** / Pro VII | gfx906 (Vega 20) | 16 GB **HBM2** (~1 TB/s) | the fast-memory half |
| Radeon **RX 9060 XT** (Gigabyte Gaming OC) | gfx1200 (Navi 44, RDNA4) | 16 GB GDDR6 | the modern half |

Host: Ryzen 9 7900X, Ubuntu 24.04.4, kernel 6.8, `amdgpu-dkms 6.12.12`.

> Note on the MI50 "32 GB" myth: a board reporting subsystem `1002:081e` can be a 32 GB-class PCB, but the VBIOS string is authoritative — ours reads `60CU/16GB 4HI` (4-high HBM stacks = 16 GB). 8-Hi = 32 GB. Don't assume 32 GB from the subsystem ID alone.

---

## Beyond MI50 + 9060 XT: this should work for *any* mix of AMD cards

> **Status: theory, not yet tested.** Only `gfx906 + gfx1200` has been run. Everything in this section follows from how the pieces work, not from a measurement. If you try another pair, open an issue with the result either way.

Nothing in the method is specific to Vega 20 or Navi 44. The two walls were (1) *the kernel didn't bind one card* and (2) *no single runtime carried both archs' kernels*. Both walls are generic, and so is the fix:

- **`-DAMDGPU_TARGETS` is a list.** HIP builds a fat binary with one code object per arch; at runtime each device loads the object matching its own ISA. `"gfx906;gfx1200"` is just the list we needed — `"gfx906;gfx1030;gfx1100;gfx1200"` is the same build with more entries and a bigger binary.
- **`HSA_OVERRIDE_GFX_VERSION` is process-wide.** That's exactly why the single-card fixes broke mixing: the override lies to *every* device about its ISA. Drop it and each card reports its real arch. (Only ever set it for a *lone* unsupported card.)
- **`OLLAMA_SCHED_SPREAD=1` doesn't care what the cards are.** It spreads layers across every ROCm device it enumerates, proportional to free VRAM. Three or four cards is the same flag.

So for a given card to join the pool it needs exactly three things:

| Requirement | How you know | If it's missing |
|---|---|---|
| **Kernel binds it** | `lspci -k` shows `amdgpu` on the card; it appears in `rocm-smi` | newer card → newer `amdgpu-dkms` (kernel module only, as above); older card → usually already bound |
| **rocBLAS/Tensile libs exist for its arch in the runtime** | `ls /opt/rocm/lib/rocblas/library/ \| grep gfxNNNN` inside the image | arch still supported → ROCm 7.1 already ships it; arch dropped → overlay the Tensile files from the **last ROCm that shipped it** (what we did for `gfx906` from 6.3) |
| **HIP compiler emits working code for it** | the model runs without segfault on that card | a compiler regression like `SOLVE_TRI` on gfx906 → find the op, force it to CPU fallback (same one-line `sed` pattern) |

Candidate pairings that *should* work by this logic (ordered by how confident we are — **none tested**):

| Pair | Combined VRAM | Why it should work | Watch out for |
|---|---|---|---|
| MI50 `gfx906` + RX 7900 XTX `gfx1100` | 16 + 24 = 40 GB | gfx1100 is a current RDNA3 target; same 906 overlay as here | nothing new — closest to the tested case |
| MI50 `gfx906` + RX 6800/6900 `gfx1030` | 16 + 16 = 32 GB | gfx1030 is RDNA2; check it's in 7.1's rocBLAS tree (`ls` above) — if not, overlay from the last ROCm that shipped it | same overlay pattern as 906 |
| 2× MI50 `gfx906` + RX 9060 XT `gfx1200` | 48 GB | same build, three devices, `SCHED_SPREAD` handles N | PCIe lanes / power; per-request speed bounded by the slowest hop |
| MI100 `gfx908` + anything modern | 32 + N GB | CDNA1; if 7.1's rocBLAS still ships gfx908 (`ls` above), **no overlay needed** — just add it to `AMDGPU_TARGETS` | we haven't checked 7.1's tree for gfx908 |
| MI210 `gfx90a` + RX 9070 `gfx1201` | 64 + 16 GB | CDNA2 is a current ROCm target; this is the "datacenter pull + gaming card" case | gfx1201 needs the newer `amdgpu-dkms` |
| Radeon VII `gfx906` + RX 9060 XT `gfx1200` | 16 + 16 GB | Radeon VII **is** gfx906 — this is the consumer version of the tested pair | should be identical |
| Vega 64 `gfx900` + modern | 8 + N GB | gfx900 was dropped **before** gfx906; Tensile overlay would come from ROCm ≤ 5.7 | older overlay on a 7.1 runtime = more ABI risk; the most speculative row |

Two honest limits that *don't* go away:

- **Per-request speed is gated by the slowest card and the PCIe hop**, not the sum. Mixing an MI50 with a 7900 XTX gets you the XTX's VRAM, not the XTX's speed, on the layers that land on the MI50. Concurrency (above) is how you recover aggregate throughput.
- **VRAM adds; bandwidth doesn't.** HBM2 + GDDR6 pool fine for *capacity*. Layers on the GDDR card run at GDDR speed.

The point of the method isn't a specific pair. It's that **AMD dropping an arch from ROCm no longer strands that card** — the Tensile overlay + multi-arch target keeps it computing next to whatever you buy next.

## Measurements still owed

Suggested by the ROCm AI Assistant on the AMD Discord (2026-08-20), and fair. Scripts are in `scripts/`; results go here as they land.

| Measurement | Why it matters | How | Result |
|---|---|---|---|
| **Peer-to-peer between the two archs** | If `hipDeviceCanAccessPeer` says no (common across GPU generations), every inter-card tensor copy stages through system RAM. That is the single-request speed ceiling, and a number beats a guess. | `scripts/p2p_check.sh` (builds `p2p_check.cpp` in the container; falls back to `rocm-bandwidth-test`) | *pending* |
| **Concurrency knee** | 2 in flight = 2.0x. Where does it stop? That N is the one to deploy at. | `scripts/concurrency_sweep.sh qwen2.5:32b 4` with `OLLAMA_NUM_PARALLEL=4` | *pending* (N=1: 16.6, N=2: 33.2 aggregate) |
| **`iommu=pt` on the test box** | Was it set, or did we get lucky? | `grep -o 'iommu=[a-z]*' /proc/cmdline` | *pending* |
| **Other card pairs** | Turns one proof of concept into a reference matrix. | Build with your archs in `AMDGPU_TARGETS`, run the model, report | see table below |

### Tested-pairs matrix

| Pair | Archs | Combined VRAM | 32B Q4 gen tok/s | Who | Notes |
|---|---|---|---|---|---|
| MI50 + RX 9060 XT | gfx906 + gfx1200 | 32 GB | 16.6 (33.2 @ 2 concurrent) | Elyan Labs | the original |
| *your pair here* | | | | | **open an issue** with `rocm-smi` output, the `AMDGPU_TARGETS` you built with, and `ollama ps` showing the split |

We do not own a gfx908 / gfx90a card, so the CDNA rows in the candidate table above stay theory until someone with an MI100 or MI210 runs it. If that is you, the whole build is `./build_dual_arch.sh` with your archs edited in.

## Status / verified

- ✅ Both cards kernel-bound after `amdgpu-dkms 6.4.2` + reboot (9060 went from "no driver" to enumerated).
- ✅ Each card computes on its *own* stock ollama (MI50 on 0.12.3, 9060 on 0.32.5).
- ✅ Reproduced the graft segfault on gfx906 under ROCm 7.2 (confirms the SOLVE_TRI source patch is required, not just Tensile files).
- ✅ **Dual-arch build runs, both cards in one runtime, model splits across them — no segfault.** One ollama process enumerates `compute=gfx906 (MI50, 16 GiB)` **and** `compute=gfx1200 (RX 9060 XT, 15.9 GiB)` under ROCm 7.1 (driver 70125).
- ✅ **32B Q4_K_M runs split across the pair at 16.6 tok/s, 100% GPU** (see [Benchmark](#benchmark)). A 14b split reports `offloaded 49/49 layers` across `ROCm0` + `ROCm1`.

## What people are saying

> "This method is truly groundbreaking. It removes the `HSA_OVERRIDE_GFX_VERSION=9.0.6` variable and compiles both architecture kernels (GFX906; GFX1200) together during compilation. Then, it uses `OLLAMA_SCHED_SPREAD=1` to distribute the model layers across both cards, each using its own actual model. With just two lines of modification, a defunct MI50 and a gaming card can achieve 32GB of storage, capable of running 32B (Q4, 65 layers all on GPU, 16.6 tok/s) that a single card can't handle. Impressive, very innovative, and provides a new solution for some users."
>
> — Guo Hongwei, AMD Developer Discord `#rocm`, 2026-08-20

## Credits
- [xxDoman/ollama_mi50](https://github.com/xxDoman/ollama_mi50) — ROCm 7.1 runtime + ROCm 6.3 Tensile method.
- [MTLoser/ollama-mi50-rocm71-build](https://github.com/MTLoser/ollama-mi50-rocm71-build) — `v0.18.2` source build + SOLVE_TRI patch.
- Multi-arch generalization + Navi 44 pairing: [Elyan Labs](https://rustchain.org).

## License
The multi-arch changes here are MIT. The base build (Tensile libs, SOLVE_TRI patch, Dockerfiles) belongs to its respective authors above — see their repos for terms.
