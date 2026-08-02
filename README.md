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

Context for the number: a single 24 GB card runs 32B Q4 at ~30–40 t/s; a V100-32 ~20–25. Two 16 GB budget/salvage cards splitting a model over PCIe and landing at **~17 t/s** is respectable — and it runs a model that fits on **neither** card alone. The heterogeneous PCIe-boundary handoff is the real cost (as expected); the `SOLVE_TRI→CPU` patch is **not** in the inference path, so it costs nothing at generation time (verified: 100% GPU at 8K ctx).

---

## Hardware this was done on

| Card | Arch | VRAM | Role in split |
|------|------|------|---------------|
| Radeon Instinct **MI50** / Pro VII | gfx906 (Vega 20) | 16 GB **HBM2** (~1 TB/s) | the fast-memory half |
| Radeon **RX 9060 XT** (Gigabyte Gaming OC) | gfx1200 (Navi 44, RDNA4) | 16 GB GDDR6 | the modern half |

Host: Ryzen 9 7900X, Ubuntu 24.04.4, kernel 6.8, `amdgpu-dkms 6.12.12`.

> Note on the MI50 "32 GB" myth: a board reporting subsystem `1002:081e` can be a 32 GB-class PCB, but the VBIOS string is authoritative — ours reads `60CU/16GB 4HI` (4-high HBM stacks = 16 GB). 8-Hi = 32 GB. Don't assume 32 GB from the subsystem ID alone.

---

## Status / verified

- ✅ Both cards kernel-bound after `amdgpu-dkms 6.4.2` + reboot (9060 went from "no driver" to enumerated).
- ✅ Each card computes on its *own* stock ollama (MI50 on 0.12.3, 9060 on 0.32.5).
- ✅ Reproduced the graft segfault on gfx906 under ROCm 7.2 (confirms the SOLVE_TRI source patch is required, not just Tensile files).
- ✅ **Dual-arch build runs, both cards in one runtime, model splits across them — no segfault.** One ollama process enumerates `compute=gfx906 (MI50, 16 GiB)` **and** `compute=gfx1200 (RX 9060 XT, 15.9 GiB)` under ROCm 7.1 (driver 70125).
- ✅ **32B Q4_K_M runs split across the pair at 16.6 tok/s, 100% GPU** (see [Benchmark](#benchmark)). A 14b split reports `offloaded 49/49 layers` across `ROCm0` + `ROCm1`.

## Credits
- [xxDoman/ollama_mi50](https://github.com/xxDoman/ollama_mi50) — ROCm 7.1 runtime + ROCm 6.3 Tensile method.
- [MTLoser/ollama-mi50-rocm71-build](https://github.com/MTLoser/ollama-mi50-rocm71-build) — `v0.18.2` source build + SOLVE_TRI patch.
- Multi-arch generalization + Navi 44 pairing: [Elyan Labs](https://rustchain.org).

## License
The multi-arch changes here are MIT. The base build (Tensile libs, SOLVE_TRI patch, Dockerfiles) belongs to its respective authors above — see their repos for terms.
