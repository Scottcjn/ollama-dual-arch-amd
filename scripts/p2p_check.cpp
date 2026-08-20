// p2p_check: does HIP allow direct peer access between every GPU pair, and
// what bandwidth do you actually get, peer-direct vs staged through host RAM?
// Build: hipcc -O2 -o p2p_check p2p_check.cpp
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1);} } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  size_t mb = argc > 1 ? atoi(argv[1]) : 256;   // transfer size in MiB
  int iters = argc > 2 ? atoi(argv[2]) : 10;
  size_t bytes = mb << 20;
  int n = 0; CK(hipGetDeviceCount(&n));
  printf("devices: %d   transfer: %zu MiB x %d iters\n", n, mb, iters);
  std::vector<hipDeviceProp_t> props(n);
  for (int i = 0; i < n; i++) {
    CK(hipGetDeviceProperties(&props[i], i));
    printf("  [%d] %s  arch=%s  pci=%04x:%02x:%02x  %zu MiB\n", i, props[i].name,
           props[i].gcnArchName, props[i].pciDomainID, props[i].pciBusID, props[i].pciDeviceID,
           props[i].totalGlobalMem >> 20);
  }
  if (n < 2) { printf("need >=2 devices\n"); return 0; }

  std::vector<void*> buf(n);
  for (int i = 0; i < n; i++) { CK(hipSetDevice(i)); CK(hipMalloc(&buf[i], bytes)); CK(hipMemset(buf[i], i + 1, bytes)); }
  void* host = nullptr; CK(hipHostMalloc(&host, bytes));

  printf("\n%-6s %-6s %-10s %-14s %-14s\n", "src", "dst", "canAccess", "peer GB/s", "via-host GB/s");
  for (int s = 0; s < n; s++) for (int d = 0; d < n; d++) {
    if (s == d) continue;
    int can = 0; CK(hipDeviceCanAccessPeer(&can, s, d));
    double peer_gbs = -1, host_gbs = -1;

    // hipMemcpyPeer: if P2P is enabled the runtime goes direct; otherwise it
    // stages through system memory itself. Either way this is what a
    // framework's inter-GPU tensor copy pays.
    CK(hipSetDevice(s));
    if (can) { hipError_t e = hipDeviceEnablePeerAccess(d, 0); if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) printf("  enable %d->%d: %s\n", s, d, hipGetErrorString(e)); }
    CK(hipMemcpyPeer(buf[d], d, buf[s], s, bytes)); CK(hipDeviceSynchronize());
    double t0 = now();
    for (int i = 0; i < iters; i++) CK(hipMemcpyPeer(buf[d], d, buf[s], s, bytes));
    CK(hipDeviceSynchronize());
    peer_gbs = (double)bytes * iters / (now() - t0) / 1e9;

    // Explicit D2H then H2D through pinned host memory: the floor P2P falls back to.
    CK(hipSetDevice(s)); CK(hipMemcpy(host, buf[s], bytes, hipMemcpyDeviceToHost));
    CK(hipSetDevice(d)); CK(hipMemcpy(buf[d], host, bytes, hipMemcpyHostToDevice));
    t0 = now();
    for (int i = 0; i < iters; i++) {
      CK(hipSetDevice(s)); CK(hipMemcpy(host, buf[s], bytes, hipMemcpyDeviceToHost));
      CK(hipSetDevice(d)); CK(hipMemcpy(buf[d], host, bytes, hipMemcpyHostToDevice));
    }
    host_gbs = (double)bytes * iters / (now() - t0) / 1e9;

    printf("%-6d %-6d %-10s %-14.2f %-14.2f\n", s, d, can ? "yes" : "NO", peer_gbs, host_gbs);
  }
  printf("\nIf canAccess=NO, or peer GB/s ~= via-host GB/s, every inter-card tensor\n"
         "handoff is going through system RAM. That is the single-request ceiling.\n");
  return 0;
}
