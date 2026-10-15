// SPDX-License-Identifier: Apache-2.0
// RGP-R97 minimal repro: launch a small kernel in a loop for N seconds on one HIP device.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %d %s at %d\n", (int)e_, hipGetErrorString(e_), __LINE__); fflush(stdout); return 2; } } while (0)
__global__ void axpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
int main(int argc, char** argv) {
  int secs = argc > 1 ? atoi(argv[1]) : 60;
  size_t mb = argc > 2 ? (size_t)atoll(argv[2]) : 0;  // optional big allocation (MiB) to mimic a full VRAM pool
  int ndev = 0; CK(hipGetDeviceCount(&ndev));
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  printf("devices %d, dev0 %s %s\n", ndev, p.name, p.gcnArchName); fflush(stdout);
  const int n = 1 << 22;
  float *x, *y; CK(hipMalloc(&x, n * 4)); CK(hipMalloc(&y, n * 4));
  CK(hipMemset(x, 0, n * 4)); CK(hipMemset(y, 0, n * 4));
  void* big = nullptr; if (mb) { CK(hipMalloc(&big, mb << 20)); printf("big alloc %zu MiB ok\n", mb); }
  // idle a bit first (the capture tool arms after a delay; WHIRL is idle then too)
  auto t0 = std::chrono::steady_clock::now();
  long long launches = 0;
  int report = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
    for (int k = 0; k < 100; ++k) { axpy<<<n / 256, 256>>>(y, x, 1.0f, n); ++launches; }
    CK(hipDeviceSynchronize());
    int s = (int)std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (s >= report) { printf("t=%ds launches=%lld\n", s, launches); fflush(stdout); report = s + 5; }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  printf("done launches=%lld\n", launches); fflush(stdout);
  return 0;
}
