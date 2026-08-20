#!/bin/bash
# Build a multi-arch (default gfx906 + gfx1200) ollama from MTLoser's gfx906 base.
# Produces a docker image `ollama-dual` that runs an MI50 (Vega20) and a modern
# RDNA4 card (e.g. RX 9060 XT) in ONE runtime and splits a model across both.
# See README.md. Requires: docker, ~40GB disk, a decent internet link (ROCm 7.1
# is pulled twice). Prereq: both cards must already be kernel-bound -- see
# README "Wall 1" (amdgpu-dkms 6.4.2 for Navi 44).
set -euxo pipefail

# Which archs to compile for. Default is the tested pair. For another mix,
# e.g. MI50 + 7900 XTX:  ARCHS="gfx906;gfx1100" ./build_dual_arch.sh
# Any arch you add must have rocBLAS/Tensile libs in the runtime image
# (ROCm 7.1 ships current archs; dropped ones need an overlay like the 906 one
# MTLoser's base already carries). See README "Beyond MI50 + 9060 XT".
ARCHS="${ARCHS:-gfx906;gfx1200}"

# 1. base repo (MTLoser) -- the 906 Tensile libs, Dockerfiles, and SOLVE_TRI patch script
if [ ! -d ollama-mi50-rocm71-build ]; then
  git clone https://github.com/MTLoser/ollama-mi50-rocm71-build.git
fi
cd ollama-mi50-rocm71-build

# 2. ROCm 7.1 installer. Both Dockerfiles COPY a deb; the '6.3.70100' name is stale
#    upstream, so fetch the real 7.1 installer and alias it to the expected name.
if [ ! -f amdgpu-install_7.1.70100-1_all.deb ]; then
  wget -q https://repo.radeon.com/amdgpu-install/7.1/ubuntu/noble/amdgpu-install_7.1.70100-1_all.deb
fi
cp -f amdgpu-install_7.1.70100-1_all.deb amdgpu-install_6.3.70100-1_all.deb

# 3. THE multi-arch change: compile for BOTH archs (base is gfx906 only).
sed -i "s/-DAMDGPU_TARGETS=\"gfx906\"/-DAMDGPU_TARGETS=\"${ARCHS}\"/" build-ollama.sh
grep -q -- "-DAMDGPU_TARGETS=\"${ARCHS}\"" build-ollama.sh  # fail loudly if upstream changed the line
# make the compiled tgz land in the mounted /build/output so it survives `--rm`
sed -i 's#tar -czf /build/ollama-#tar -czf /build/output/ollama-#' build-ollama.sh
sed -i 's#ls -lh /build/ollama-#ls -lh /build/output/ollama-#' build-ollama.sh
# runtime: DROP the global gfx906 override (it forces a modern card down to 906);
# spread layers across both GPUs by default.
sed -i 's/HSA_OVERRIDE_GFX_VERSION=9.0.6/OLLAMA_SCHED_SPREAD=1/' Dockerfile

# 4. build: builder image (ROCm 7.1 dev) -> compile ollama -> runtime image
docker build -t ollama-builder -f Dockerfile.build .
mkdir -p output
docker run --rm \
  -v "$PWD/output:/build/output" \
  -v "$PWD/build-ollama.sh:/build/build-ollama.sh" \
  -e OLLAMA_VERSION=v0.18.2 \
  ollama-builder bash /build/build-ollama.sh
cp -f output/ollama-v0.18.2-rocm-gfx906.tgz ollama-v0.18.2-rocm-gfx906.tgz
docker build -t ollama-dual .

cat <<'EOF'

=== Built image: ollama-dual ===
Run it (SCHED_SPREAD fans layers across all visible GPUs):

  docker run -d --name ollama-dual \
    --device=/dev/kfd --device-cgroup-rule='c 226:* rmw' -v /dev/dri:/dev/dri \
    -v ollama_models:/models -e OLLAMA_MODELS=/models \
    -e OLLAMA_SCHED_SPREAD=1 -p 11434:11434 ollama-dual

Then: docker exec ollama-dual ollama run qwen2.5:32b "hello"
EOF
