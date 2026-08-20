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
| **2 concurrent requests** | **23.7 tok/s aggregate** (1.5x; measured 2026-08-20 at 2K ctx, see below). An earlier "33.2 / 2.0x" figure summed per-request rates and was wrong. |

Context for the number: a single 24 GB card runs 32B Q4 at ~30–40 t/s; a V100-32 ~20–25. Two 16 GB budget/salvage cards splitting a model over PCIe and landing at **~17 t/s** is respectable — and it runs a model that fits on **neither** card alone. The heterogeneous PCIe-boundary handoff is the real cost (as expected); the `SOLVE_TRI→CPU` patch is **not** in the inference path, so it costs nothing at generation time (verified: 100% GPU at 8K ctx).

> **Update 2026-08-20: that 16.6 tok/s was measured with the MI50 on a PCIe Gen1 x4 link (0.8 GB/s).** We did not know that when we published. See [The P2P reality check](#the-p2p-reality-check-measured). The number is real; the ceiling it hit was a slot, not the architecture mix.

**Concurrency is where the second card earns some of its keep.** A layer split is serial for a *single* request — while card 0 runs the early layers, card 1 is idle (it's acting as a RAM coffer, not a co-processor). With **two requests in flight**, the pipeline overlaps: card 0 works request A's early layers while card 1 works request B's late layers. Measured cleanly (total tokens / wall-clock, idle GPUs, 2K ctx): **1.5x at N=2, 1.73x at N=4**, with the knee at N=2. We first published "2x" from summing per-request rates; that was the wrong arithmetic and is retracted. Note the inter-card hop on this box was PCIe Gen1 x4 (see [P2P](#the-p2p-reality-check-measured)), which also taxes the pipelined case; expect the curve to improve on a sane slot.

---

## Hardware this was done on

| Card | Arch | VRAM | Role in split |
|------|------|------|---------------|
| Radeon Instinct **MI50** / Pro VII | gfx906 (Vega 20) | 16 GB **HBM2** (~1 TB/s) | the fast-memory half |
| Radeon **RX 9060 XT** (Gigabyte Gaming OC) | gfx1200 (Navi 44, RDNA4) | 16 GB GDDR6 | the modern half |

Host: Ryzen 9 7900X, Ubuntu 24.04.4, kernel 6.8, `amdgpu-dkms 6.12.12`. Slots: 9060 XT on CPU lanes (`00:01.1`, x16 Gen5), MI50 on **chipset lanes** (`00:02.1` → 600-series → `05:00.0`, x4 physical, negotiated **Gen1**). The Raphael iGPU (gfx1036) is also visible to ROCm; ollama ignores it for the split.

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

For reference, the rocBLAS/Tensile tree inside our `ollama-dual` image (ROCm 7.1 runtime + the 906 overlay) contains exactly these archs:

```
gfx906 (overlaid)  gfx908  gfx90a  gfx942  gfx950
gfx1030  gfx1100  gfx1101  gfx1102  gfx1150  gfx1151  gfx1200  gfx1201
```

Anything on that list needs only an `AMDGPU_TARGETS` entry. Anything not on it (gfx900 Vega 10, gfx803 Polaris, gfx1010/1012 Navi 10) needs the overlay trick from whatever ROCm last shipped it.

Candidate pairings that *should* work by this logic (ordered by how confident we are — **none tested**):

| Pair | Combined VRAM | Why it should work | Watch out for |
|---|---|---|---|
| MI50 `gfx906` + RX 7900 XTX `gfx1100` | 16 + 24 = 40 GB | gfx1100 is a current RDNA3 target; same 906 overlay as here | nothing new — closest to the tested case |
| MI50 `gfx906` + RX 6800/6900 `gfx1030` | 16 + 16 = 32 GB | gfx1030 **is in the ROCm 7.1 rocBLAS tree** (verified in our image) | nothing new |
| 2× MI50 `gfx906` + RX 9060 XT `gfx1200` | 48 GB | same build, three devices, `SCHED_SPREAD` handles N | PCIe lanes / power; per-request speed bounded by the slowest hop |
| MI100 `gfx908` + anything modern | 32 + N GB | gfx908 **is in the ROCm 7.1 rocBLAS tree** (verified in our image), so **no overlay needed**: just add it to `AMDGPU_TARGETS` | — |
| MI210 `gfx90a` + RX 9070 `gfx1201` | 64 + 16 GB | gfx90a and gfx1201 are **both in the 7.1 rocBLAS tree**; this is the "datacenter pull + gaming card" case | gfx1201 needs the newer `amdgpu-dkms` |
| Radeon VII `gfx906` + RX 9060 XT `gfx1200` | 16 + 16 GB | Radeon VII **is** gfx906 — this is the consumer version of the tested pair | should be identical |
| Vega 64 `gfx900` + modern | 8 + N GB | gfx900 was dropped **before** gfx906; Tensile overlay would come from ROCm ≤ 5.7 | older overlay on a 7.1 runtime = more ABI risk; the most speculative row |

**Which slot matters more than which card.** On a consumer board the second x16-physical slot is usually x4 on chipset lanes, and (as we found the hard way) it can negotiate down to Gen1. Run `scripts/p2p_check.sh` before you benchmark anything; it prints the upstream port's real link speed. Put the card that holds the most layers on CPU lanes.

Two honest limits that *don't* go away:

- **Per-request speed is gated by the slowest card and the PCIe hop**, not the sum. Mixing an MI50 with a 7900 XTX gets you the XTX's VRAM, not the XTX's speed, on the layers that land on the MI50. Concurrency (above) is how you recover aggregate throughput.
- **VRAM adds; bandwidth doesn't.** HBM2 + GDDR6 pool fine for *capacity*. Layers on the GDDR card run at GDDR speed.

The point of the method isn't a specific pair. It's that **AMD dropping an arch from ROCm no longer strands that card** — the Tensile overlay + multi-arch target keeps it computing next to whatever you buy next.

## The P2P reality check (measured)

The ROCm AI Assistant on the AMD Discord asked for three things (2026-08-20): measure peer-to-peer between the archs, find the concurrency knee, and call out `iommu=pt`. Fair asks. Scripts are in `scripts/`, and here is what they found on the box above.

### Peer-to-peer: `scripts/p2p_check.sh`

`hipDeviceCanAccessPeer` says **yes** for every pair. Bandwidth says something else (256 MiB x 10, `hipMemcpyPeer` vs explicit device→pinned-host→device):

| src → dst | peer GB/s | via-host GB/s |
|---|---|---|
| RX 9060 XT (gfx1200) → MI50 (gfx906) | **0.79** | 0.81 |
| MI50 (gfx906) → RX 9060 XT (gfx1200) | **0.82** | 0.81 |
| RX 9060 XT → Raphael iGPU (gfx1036) | 27.01 | 11.32 |
| Raphael iGPU → RX 9060 XT | 14.59 | 11.62 |
| MI50 → iGPU / iGPU → MI50 | 0.82 / 0.83 | 0.80 / 0.81 |

Peer ≈ via-host on the 906↔1200 pair, so the peer path buys nothing there. But the real finding is the absolute number: **everything that touches the MI50 caps at 0.8 GB/s**, while the 9060 moves 27 GB/s to the iGPU under the same load. We chased it:

- Not the BAR: both cards have a 16 GB large BAR (`lspci -vv` Region 0).
- Not SDMA: `HSA_ENABLE_SDMA=0` gives the same 0.8 GB/s.
- Not the IOMMU: `iommu=pt` was **not** set on this box, and all three GPUs were already in `identity` IOMMU groups. Setting it would not have changed anything here.
- Not the card's link: `lspci -vv -s 08:00.0` and sysfs both say **x16 @ 16 GT/s**. That is the Vega 20 board's *internal* bridge→GPU link.
- **It is the slot.** `lspci -tv` puts the MI50 behind the 600-series chipset, and the chipset downstream port feeding it (`05:00.0`) is linked at **x4 @ 2.5 GT/s = PCIe Gen1 x4 ≈ 1 GB/s raw**. The kernel said so at boot and nobody read it:

```
pci 0000:08:00.0: 8.000 Gb/s available PCIe bandwidth, limited by 2.5 GT/s PCIe x4 link
    at 0000:05:00.0 (capable of 252.048 Gb/s with 16.0 GT/s PCIe x16 link)
```

The port is *capable* of x4 @ 16 GT/s (~7.9 GB/s) and never ramps: not under sustained copy load, not with `rocm-smi --setperflevel high`. So it is a BIOS/negotiation/seating problem on the chipset slot, not power management. Fix candidates, in order: set the chipset PCIe slot speed explicitly to Gen4 in BIOS instead of Auto; reseat the card; or, properly, put the dropped card on **CPU lanes** (swap slots, or bifurcate the CPU x16 to x8/x8 so both cards get direct lanes).

**The lesson, generalized:** `p2p_check.sh` now prints the link state of each GPU's **upstream port** and greps the kernel log for `limited by`. Check the slot, not the card. A salvage datacenter card in a consumer board's second x16-physical slot is very likely on chipset lanes, and that slot can negotiate down silently.

**What this means for the numbers:** the 16.6 tok/s single-request figure and the 0.8 GB/s handoff were measured on the same Gen1 x4 link. Every inter-card activation transfer paid ~10x more than it should have. We will re-measure after the slot is fixed; the expectation is a real single-request gain, since the layer-boundary handoff is the serial cost in a split.

### Concurrency knee: `scripts/concurrency_sweep.sh`

Under `OLLAMA_NUM_PARALLEL=2` at 8K context, qwen2.5:32b Q4_K_M occupies **31.1 GB of the 32 GB**. `OLLAMA_NUM_PARALLEL=4` at the default context does not load at all (KV cache for 4 slots does not fit). Results:

Measured 2026-08-20 with the corpus worker paused and the container freshly restarted (idle GPUs), `num_predict=128`, aggregate = total generated tokens / batch wall-clock:

| NUM_PARALLEL / ctx | N in flight | wall s | per-request tok/s | aggregate tok/s | speedup |
|---|---|---|---|---|---|
| 4 / 2048 | 1 | 8.2 | 16.5 | 15.7 | 1.00x |
| 4 / 2048 | 2 | 10.8 | 12.4 | 23.7 | **1.51x** |
| 4 / 2048 | 3 | 15.7 | 8.6 | 24.5 | 1.56x |
| 4 / 2048 | 4 | 18.9 | 7.1 | 27.1 | 1.73x |
| 2 / 32768 (model default) | 1 | 35.9 | **3.7** | 3.6 | — |
| 2 / 32768 (model default) | 2 | 49.9 | 3.1 | 5.1 | — |

Two lessons in that table:

- **The knee is N=2.** Going 2 → 4 buys another ~15% aggregate while per-request speed halves. For a batch workload run `OLLAMA_NUM_PARALLEL=2`; for interactive use run 1.
- **Set the context length explicitly.** Without `OLLAMA_CONTEXT_LENGTH` (or `num_ctx` on every request), ollama loads qwen2.5 at its **32K default**, multiplied by `NUM_PARALLEL`. At 2 slots that put 31.1 GB in VRAM (`size` > `size_vram`, i.e. partial offload) and single-request generation fell to **3.7 tok/s** with no warning. At 4 slots it refuses to load: `model requires more system memory (47.7 GiB) than is available`. The 16.6 tok/s headline number is at 8K; at 2K the same model takes 23 GB.

### `iommu=pt`

Documented in the setup section above. On this box it was **not set** and was **not the cause** of the slow link (IOMMU groups were already `identity`). We still recommend it for multi-GPU per AMD's docs; just do not expect it to fix a link-speed problem.

### Tested-pairs matrix

| Pair | Archs | Combined VRAM | 32B Q4 gen tok/s | Who | Notes |
|---|---|---|---|---|---|
| MI50 + RX 9060 XT | gfx906 + gfx1200 | 32 GB | 16.6 (23.7 aggregate @ 2 concurrent) | Elyan Labs | the original; MI50 was on a **Gen1 x4 chipset slot** at the time (see P2P section), re-measure pending |
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
