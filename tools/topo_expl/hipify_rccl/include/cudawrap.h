/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "checks.h"

// Is cuMem API usage enabled
extern int ncclCuMemEnable();
extern int ncclCuMemHostEnable();

#if CUDART_VERSION >= 11030
#include <cudaTypedefs.h>

// Handle type used for hipMemCreate()
extern hipMemAllocationHandleType ncclCuMemHandleType;

#endif

#define CUPFN(symbol) pfn_##symbol

// Check CUDA PFN driver calls
#define CUCHECK(cmd) do {				      \
    hipError_t err = pfn_##cmd;				      \
    if( err != hipSuccess ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      return ncclUnhandledCudaError;			      \
    }							      \
} while(false)

#define CUCALL(cmd) do {				      \
    pfn_##cmd;				                \
} while(false)

#define CUCHECKGOTO(cmd, res, label) do {		      \
    hipError_t err = pfn_##cmd;				      \
    if( err != hipSuccess ) {				      \
      const char *errStr;				      \
      (void) pfn_cuGetErrorString(err, &errStr);	      \
      WARN("Cuda failure %d '%s'", err, errStr);	      \
      res = ncclUnhandledCudaError;			      \
      goto label;					      \
    }							      \
} while(false)

// Report failure but clear error and continue
#define CUCHECKIGNORE(cmd) do {						\
    hipError_t err = pfn_##cmd;						\
    if( err != hipSuccess ) {						\
      const char *errStr;						\
      (void) pfn_cuGetErrorString(err, &errStr);			\
      INFO(NCCL_ALL,"%s:%d Cuda failure %d '%s'", __FILE__, __LINE__, err, errStr); \
    }									\
} while(false)

#define CUCHECKTHREAD(cmd, args) do {					\
    hipError_t err = pfn_##cmd;						\
    if (err != hipSuccess) {						\
      INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, err); \
      args->ret = ncclUnhandledCudaError;				\
      return args;							\
    }									\
} while(0)

#define DECLARE_CUDA_PFN_EXTERN(symbol,version) extern PFN_##symbol##_v##version pfn_##symbol

#if CUDART_VERSION >= 11030
/* CUDA Driver functions loaded with hipGetProcAddress for versioning */
DECLARE_CUDA_PFN_EXTERN(hipDeviceGet, 2000);
DECLARE_CUDA_PFN_EXTERN(hipDeviceGetAttribute, 2000);
DECLARE_CUDA_PFN_EXTERN(hipDrvGetErrorString, 6000);
DECLARE_CUDA_PFN_EXTERN(hipDrvGetErrorName, 6000);
DECLARE_CUDA_PFN_EXTERN(hipMemGetAddressRange, 3020);
DECLARE_CUDA_PFN_EXTERN(hipCtxCreate, 11040);
DECLARE_CUDA_PFN_EXTERN(hipCtxDestroy, 4000);
DECLARE_CUDA_PFN_EXTERN(hipCtxGetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(hipCtxSetCurrent, 4000);
DECLARE_CUDA_PFN_EXTERN(hipCtxGetDevice, 2000);
DECLARE_CUDA_PFN_EXTERN(hipPointerGetAttribute, 4000);
DECLARE_CUDA_PFN_EXTERN(hipModuleLaunchKernel, 4000);
#if CUDART_VERSION >= 11080
DECLARE_CUDA_PFN_EXTERN(cuLaunchKernelEx, 11060);
#endif
// cuMem API support
DECLARE_CUDA_PFN_EXTERN(hipMemAddressReserve, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemAddressFree, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemCreate, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemMap, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemRelease, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN_EXTERN(hipMemSetAccess, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemUnmap, 10020);
DECLARE_CUDA_PFN_EXTERN(hipMemGetAllocationPropertiesFromHandle, 10020);
#if CUDA_VERSION >= 11070
DECLARE_CUDA_PFN_EXTERN(cuMemGetHandleForAddressRange, 11070); // DMA-BUF support
#endif
#if CUDA_VERSION >= 12010
/* NVSwitch Multicast support */
DECLARE_CUDA_PFN_EXTERN(cuMulticastAddDevice, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindMem, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastBindAddr, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastCreate, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastGetGranularity, 12010);
DECLARE_CUDA_PFN_EXTERN(cuMulticastUnbind, 12010);
#endif
#endif

ncclResult_t ncclCudaLibraryInit(void);

extern int ncclCudaDriverVersionCache;
extern bool ncclCudaLaunchBlocking; // initialized by ncclCudaLibraryInit()

inline ncclResult_t ncclCudaDriverVersion(int* driver) {
  int version = __atomic_load_n(&ncclCudaDriverVersionCache, __ATOMIC_RELAXED);
  if (version == -1) {
    CUDACHECK(hipDriverGetVersion(&version));
    __atomic_store_n(&ncclCudaDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}
#endif
