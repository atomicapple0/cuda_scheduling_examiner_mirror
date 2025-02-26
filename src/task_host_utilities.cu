// This file contains the implementation of the functions defined in
// task_host_utilities.h--used by task_host_utilities.c to work with the GPU.
#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "task_host_utilities.h"
#include "library_interface.h"

// The number of GPU nanoseconds to spin for GetGPUTimerScale. Increasing this
// will both increase the accuracy and the time the function takes to return.
#define TIMER_SPIN_DURATION (2ull * 1000 * 1000 * 1000)

// This macro takes a cudaError_t value. It prints an error message and returns
// 0 if the cudaError_t isn't cudaSuccess. Otherwise, it returns nonzero.
#define CheckCUDAError(val) (InternalCUDAErrorCheck((val), #val, __FILE__, __LINE__))

// Prints an error message and returns 0 if the given CUDA result is an error.
static int InternalCUDAErrorCheck(cudaError_t result, const char *fn,
    const char *file, int line) {
  if (result == cudaSuccess) return 1;
  printf("CUDA error %d: %s. In %s, line %d (%s)\n", (int) result,
    cudaGetErrorString(result), file, line, fn);
  return 0;
}

static double CurrentSeconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    printf("Error getting time.\n");
    exit(1);
  }
  return ((double) ts.tv_sec) + (((double) ts.tv_nsec) / 1e9);
}

#if __CUDA_ARCH__ >= 300
// Starting with sm_30, `globaltimer64` was added
// Compiles to 5 inline SASS instructions on sm_52
static __device__ inline uint64_t GlobalTimer64(void) {
  uint32_t lo_bits, hi_bits, hi_bits_2;
  uint64_t ret;
  // Upper bits may rollever between our 1st and 2nd read
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits));
  asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(lo_bits));
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(hi_bits_2));
  // If upper bits rolled over, lo_bits = 0
  lo_bits = (hi_bits != hi_bits_2) ? 0 : lo_bits;
  // As sm_52 SASS is naively 32-bit, the following ops get optimized out
  ret = hi_bits_2;
  ret <<= 32;
  ret |= lo_bits;
  return ret;
}
#elif __CUDA_ARCH__ < 200
static __device__ inline uint64_t GlobalTimer64(void) {
    uint32_t lo_bits;
    uint64_t ret;
    // On sm_13, we /only/ have `clock` (32 bits)
    asm volatile("mov.u32 %0, %%clock;" : "=r"(lo_bits));
    ret = 0;
    ret |= lo_bits;
    return ret;
}
#else
#error Unsupported architecture!
#endif


/*
// Returns the value of CUDA's global nanosecond timer.
static __device__ inline uint64_t GlobalTimer64(void) {
  // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
  // around after the upper bits have already been read. Work around this by
  // reading the high bits a second time. Use the second value to detect a
  // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
  // would be valid, it's passed over during the duration of the reading. If no
  // rollover occurred, just return the initial reading.
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t) second_reading) << 32;
}*/

// A simple kernel which writes the value of the globaltimer64 register to a
// location in device memory.
static __global__ void GetTime(uint64_t *time, volatile uint32_t *ready_barrier, volatile uint32_t *start_barrier, volatile uint32_t *end_barrier) {
  *ready_barrier = 1;
  while (!*start_barrier)
    continue;
  *time = GlobalTimer64();
  *end_barrier = 1;
}

// Allocates a private shared memory buffer containing the given number of
// bytes. Can be freed by using FreeSharedBuffer. Returns NULL on error.
// Initializes the buffer to contain 0.
static void* AllocateSharedBuffer(size_t size) {
  void *to_return = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS |
    MAP_SHARED, -1, 0);
  if (to_return == MAP_FAILED) return NULL;
  memset(to_return, 0, size);
  return to_return;
}

// Frees a shared buffer returned by AllocateSharedBuffer.
static void FreeSharedBuffer(void *buffer, size_t size) {
  munmap(buffer, size);
}

// This function should be run in a separate process in order to read the GPU's
// nanosecond counter. Returns 0 on error.
// TODO: Cleanup allocations if something goes bad
static void InternalReadGPUNanoseconds(int cuda_device, double *cpu_time,
    uint64_t *gpu_time) {
  uint64_t *device_time = NULL;
  volatile uint32_t *gpu_start_barrier, *start_barrier = NULL;
  volatile uint32_t *gpu_end_barrier, *end_barrier = NULL;
  volatile uint32_t *gpu_ready_barrier, *ready_barrier = NULL;
  volatile double cpu_start, cpu_end;
  if (!CheckCUDAError(cudaSetDevice(cuda_device))) return;
  if (!CheckCUDAError(cudaMalloc(&device_time, sizeof(*device_time)))) return;
  if (!CheckCUDAError(cudaHostAlloc(&start_barrier, sizeof(*start_barrier), cudaHostAllocMapped))) return;
  if (!CheckCUDAError(cudaHostAlloc(&end_barrier, sizeof(*end_barrier), cudaHostAllocMapped))) return;
  if (!CheckCUDAError(cudaHostAlloc(&ready_barrier, sizeof(*ready_barrier), cudaHostAllocMapped))) return;
  // Setup device pointers for all the barriers
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_start_barrier, (uint32_t*)start_barrier, 0))) return;
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_end_barrier, (uint32_t*)end_barrier, 0))) return;
  if (!CheckCUDAError(cudaHostGetDevicePointer((uint32_t**)&gpu_ready_barrier, (uint32_t*)ready_barrier, 0))) return;
  // Run the kernel a first time to warm up the GPU.
  GetTime<<<1, 1>>>(device_time, gpu_ready_barrier, gpu_start_barrier, gpu_end_barrier);
  *start_barrier = 1;
  if (!CheckCUDAError(cudaDeviceSynchronize())) return;
  // Now run the actual time-checking kernel.
  *start_barrier = 0;
  *end_barrier = 0;
  *ready_barrier = 0;
  GetTime<<<1, 1>>>(device_time, gpu_ready_barrier, gpu_start_barrier, gpu_end_barrier);
  while (!*ready_barrier)
    continue;
  *start_barrier = 1;
  cpu_start = CurrentSeconds();
  while (!*end_barrier)
    continue;
  cpu_end = CurrentSeconds();
  volatile double cpu_end2 = CurrentSeconds();
  if (!CheckCUDAError(cudaMemcpy(gpu_time, device_time, sizeof(device_time), cudaMemcpyDeviceToHost))) return;
  *cpu_time = (cpu_end - cpu_start) / 2.0 + cpu_start;
  double max_error = (cpu_end - cpu_start) / 2.0;
  fprintf(stderr, "max_error: %f us (error floor of %f us)\n", max_error * (1000.0 * 1000.0), (cpu_end2 - cpu_end) / 2.0 * (1000.0*1000.0));
  cudaFree((void*)device_time);
  cudaFree((void*)ready_barrier);
  cudaFree((void*)start_barrier);
  cudaFree((void*)end_barrier);
}

int GetHostDeviceTimeOffset(int cuda_device, double *host_seconds,
  uint64_t *gpu_nanoseconds) {
  uint64_t *shared_gpu_time = NULL;
  double *shared_cpu_time = NULL;
  int status;
  pid_t pid = -1;
  shared_gpu_time = (uint64_t *) AllocateSharedBuffer(
    sizeof(*shared_gpu_time));
  if (!shared_gpu_time) {
    printf("Failed allocating shared buffer for IPC.\n");
    return 0;
  }
  shared_cpu_time = (double *) AllocateSharedBuffer(sizeof(*shared_cpu_time));
  if (!shared_cpu_time) {
    printf("Failed allocating shared CPU time buffer for IPC.\n");
    FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
    return 0;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process to get GPU time: %s\n", strerror(
      errno));
    return 0;
  }
  if (pid == 0) {
    // The following CUDA code is run in the child process
    InternalReadGPUNanoseconds(cuda_device, shared_cpu_time, shared_gpu_time);
    exit(0);
  }
  // The parent will wait for the child to finish, then return the value
  // written to the shared buffer.
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(shared_cpu_time, sizeof(*shared_cpu_time));
    FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
    return 0;
  }
  *host_seconds = *shared_cpu_time;
  *gpu_nanoseconds = *shared_gpu_time;
  FreeSharedBuffer(shared_cpu_time, sizeof(*shared_cpu_time));
  FreeSharedBuffer(shared_gpu_time, sizeof(*shared_gpu_time));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  return 1;
}

// This function should always be run in a separate process.
static int InternalGetMaxResidentThreads(int cuda_device) {
  struct cudaDeviceProp properties;
  int warps_per_sm = 64;
  if (!CheckCUDAError(cudaGetDeviceProperties(&properties, cuda_device))) {
    return 0;
  }
  // Compute capability 2.0 devices have a 48 warps per SM.
  if (properties.major <= 2) warps_per_sm = 48;
  return warps_per_sm * properties.multiProcessorCount * properties.warpSize;
}

int GetMaxResidentThreads(int cuda_device) {
  int to_return, status;
  pid_t pid = -1;
  int *max_thread_count = NULL;
  max_thread_count = (int *) AllocateSharedBuffer(sizeof(*max_thread_count));
  if (!max_thread_count) {
    printf("Failed allocating shared buffer for IPC.\n");
    return 0;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process to get thread count: %s\n",
      strerror(errno));
    return 0;
  }
  if (pid == 0) {
    // The following CUDA code is run in the child process
    *max_thread_count = InternalGetMaxResidentThreads(cuda_device);
    exit(0);
  }
  // The parent will wait for the child to finish, then return the value
  // written to the shared buffer.
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(max_thread_count, sizeof(*max_thread_count));
    return 0;
  }
  to_return = *max_thread_count;
  FreeSharedBuffer(max_thread_count, sizeof(*max_thread_count));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return 0;
  }
  return to_return;
}

#if __CUDA_ARCH__ >= 300
// Starting with sm_30, `globaltimer64` was added
static __global__ void TimerSpin(uint64_t ns_to_spin) {
  uint64_t start_time = GlobalTimer64();
  while ((GlobalTimer64() - start_time) < ns_to_spin) {
    continue;
  }
}
#elif __CUDA_ARCH__ < 200
// On sm_13, we /only/ have `clock` (32 bits)
static __device__ inline uint32_t Clock32(void) {
    uint32_t lo_bits;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(lo_bits));
    return lo_bits;
}

static __global__ void TimerSpin(uint64_t ns_to_spin) {
  uint64_t total_time = 0;
  uint32_t last_time = Clock32();
  while (total_time < ns_to_spin) {
    uint32_t time = Clock32();
    if (time < last_time) {
      // Rollover. Compensate...
      total_time += time; // rollover to now
      total_time += UINT_MAX - last_time;// last to rollover
    } else {
      // Step counter
      total_time += time - last_time;
    }
    last_time = time;
  }
}
#else
#error Unsupported architecture!
#endif

// This function is intended to be run in a child process. Returns -1 on error.
double InternalGetGPUTimerScale(int cuda_device) {
  struct timespec start, end;
  uint64_t nanoseconds_elapsed;
  if (!CheckCUDAError(cudaSetDevice(cuda_device))) return -1;
  // Run the kernel once to warm up the GPU.
  TimerSpin<<<1, 1>>>(1000);
  if (!CheckCUDAError(cudaDeviceSynchronize())) return -1;
  // After warming up, do the actual timing.
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) != 0) {
    printf("Failed getting start time.\n");
    return -1;
  }
  TimerSpin<<<1, 1>>>(TIMER_SPIN_DURATION);
  if (!CheckCUDAError(cudaDeviceSynchronize())) return -1;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &end) != 0) {
    printf("Failed getting end time.\n");
    return -1;
  }
  nanoseconds_elapsed = end.tv_sec * 1e9 + end.tv_nsec;
  nanoseconds_elapsed -= start.tv_sec * 1e9 + start.tv_nsec;
  printf("%ldns elapsed\n", nanoseconds_elapsed);
  return ((double) nanoseconds_elapsed) / ((double) TIMER_SPIN_DURATION);
}

double GetGPUTimerScale(int cuda_device) {
  double to_return;
  double *scale = NULL;
  int status;
  pid_t pid;
  scale = (double *) AllocateSharedBuffer(sizeof(*scale));
  if (!scale) {
    printf("Failed allocating space to hold the GPU time scale.\n");
    return -1;
  }
  pid = fork();
  if (pid < 0) {
    printf("Failed creating a child process.\n");
    FreeSharedBuffer(scale, sizeof(*scale));
    return -1;
  }
  if (pid == 0) {
    // Access the GPU with the child process only.
    *scale = InternalGetGPUTimerScale(cuda_device);
    exit(0);
  }
  if (wait(&status) < 0) {
    printf("Failed waiting on the child process.\n");
    FreeSharedBuffer(scale, sizeof(*scale));
    return -1;
  }
  to_return = *scale;
  FreeSharedBuffer(scale, sizeof(*scale));
  if (!WIFEXITED(status)) {
    printf("The child process didn't exit normally.\n");
    return -1;
  }
  return to_return;
}
