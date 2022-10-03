#include <stdio.h>
#include <cuda.h>

#define N 10000000

// __global__ void vector_add(float *out, float *a, float *b, int n) {
//     for(int i = 0; i < n; i++){
//         out[i] = a[i] + b[i];
//     }
// }

void printCharArray(char* charArray, int n){
    int i = 0;
    for (i = 0; i < n; i++)
        printf("%02hhX ", charArray[i]);
}

void printIntArray(int* arr, int n){
    int i = 0;
    printf("[");
    for (i = 0; i < n; i++){
        printf("%d", arr[i]);
        if (i != n -1)
            printf(", ");
    }
    printf("]");
}

int main(){
    int dev_count;
    cudaGetDeviceCount( &dev_count);

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties( &dev_prop, i);
        // decide if device has sufficient resources and capabilities
        printf("DEVICE NUMBER %d:\n", i + 1);
        printf("    accessPolicyMaxWindowSize : %d\n", dev_prop.accessPolicyMaxWindowSize);
        printf("    asyncEngineCount : %d\n", dev_prop.asyncEngineCount);
        printf("    canMapHostMemory : %d\n", dev_prop.canMapHostMemory);
        printf("    canUseHostPointerForRegisteredMem : %d\n", dev_prop.canUseHostPointerForRegisteredMem);
        printf("    clockRate : %d\n", dev_prop.clockRate);
        printf("    computeMode : %d\n", dev_prop.computeMode);
        printf("    computePreemptionSupported : %d\n", dev_prop.computePreemptionSupported);
        printf("    concurrentKernels : %d\n", dev_prop.concurrentKernels);
        printf("    concurrentManagedAccess : %d\n", dev_prop.concurrentManagedAccess);
        printf("    cooperativeLaunch : %d\n", dev_prop.cooperativeLaunch);
        printf("    cooperativeMultiDeviceLaunch : %d\n", dev_prop.cooperativeMultiDeviceLaunch);
        printf("    deviceOverlap : %d\n", dev_prop.deviceOverlap);
        printf("    directManagedMemAccessFromHost : %d\n", dev_prop.directManagedMemAccessFromHost);
        printf("    globalL1CacheSupported : %d\n", dev_prop.globalL1CacheSupported);
        printf("    hostNativeAtomicSupported : %d\n", dev_prop.hostNativeAtomicSupported);
        printf("    integrated : %d\n", dev_prop.integrated);
        printf("    isMultiGpuBoard : %d\n", dev_prop.isMultiGpuBoard);
        printf("    kernelExecTimeoutEnabled : %d\n", dev_prop.kernelExecTimeoutEnabled);
        printf("    l2CacheSize : %d\n", dev_prop.l2CacheSize);
        printf("    localL1CacheSupported : %d\n", dev_prop.localL1CacheSupported);

        printf("    luid : ");
        printCharArray(dev_prop.luid, 8);
        printf("\n");

        printf("    major : %d\n", dev_prop.major);
        printf("    managedMemory : %d\n", dev_prop.managedMemory);
        printf("    maxBlocksPerMultiProcessor : %d\n", dev_prop.maxBlocksPerMultiProcessor);

        printf("    maxGridSize : ");
        printIntArray(dev_prop.maxGridSize, 3);
        printf("\n");
        
        printf("    maxSurface1D : %d\n", dev_prop.maxSurface1D);

        printf("    maxSurface1DLayered : ");
        printIntArray(dev_prop.maxSurface1DLayered, 2);
        printf("\n");

        printf("    maxSurface2D : ");
        printIntArray(dev_prop.maxSurface2D, 2);
        printf("\n");

        printf("    maxSurface2DLayered : ");
        printIntArray(dev_prop.maxSurface2DLayered, 3);
        printf("\n");

        printf("    maxSurface3D : ");
        printIntArray(dev_prop.maxSurface3D, 3);
        printf("\n");

        printf("    maxSurfaceCubemap : %d", dev_prop.maxSurfaceCubemap);
        
        printf("    maxSurfaceCubemapLayered : ");
        printIntArray(dev_prop.maxSurfaceCubemapLayered, 2);
        printf("\n");

        printf("    maxTexture1D : %d", dev_prop.maxTexture1D);
        
        printf("    maxTexture1DLayered : ");
        printIntArray(dev_prop.maxTexture1DLayered, 2);
        printf("\n");

        printf("    maxTexture1DLinear : %d", dev_prop.maxTexture1DLinear);
        printf("    maxTexture1DMipmap : %d", dev_prop.maxTexture1DMipmap);
        
        printf("    maxTexture2D : ");
        printIntArray(dev_prop.maxTexture2D, 2);
        printf("\n");

        printf("    maxTexture2DGather : ");
        printIntArray(dev_prop.maxTexture2DGather, 2);
        printf("\n");

        printf("    maxTexture2DLayered : ");
        printIntArray(dev_prop.maxTexture2DLayered, 3);
        printf("\n");

        printf("    maxTexture2DLinear : ");
        printIntArray(dev_prop.maxTexture2DLinear, 3);
        printf("\n");

        printf("    maxTexture2DMipmap : ");
        printIntArray(dev_prop.maxTexture2DMipmap, 2);
        printf("\n");

        printf("    maxTexture3D : ");
        printIntArray(dev_prop.maxTexture3D, 3);
        printf("\n");

        printf("    maxTexture3DAlt : ");
        printIntArray(dev_prop.maxTexture3DAlt, 3);
        printf("\n");

        printf("    maxTextureCubemap : %d", dev_prop.maxTextureCubemap);
        
        printf("    maxTextureCubemapLayered : ");
        printIntArray(dev_prop.maxTextureCubemapLayered, 2);
        printf("\n");

        printf("    maxThreadsDim : ");
        printIntArray(dev_prop.maxThreadsDim, 3);
        printf("\n");

        printf("    maxThreadsPerBlock : %d\n", dev_prop.maxThreadsPerBlock);
        printf("    maxThreadsPerMultiProcessor : %d\n", dev_prop.maxThreadsPerMultiProcessor);
        printf("    memPitch : %zu\n", dev_prop.memPitch);
        printf("    memoryBusWidth : %d\n", dev_prop.memoryBusWidth);
        printf("    memoryClockRate : %d\n", dev_prop.memoryClockRate);
        printf("    minor : %d\n", dev_prop.minor);
        printf("    multiGpuBoardGroupID : %d\n", dev_prop.multiGpuBoardGroupID);
        printf("    multiProcessorCount : %d\n", dev_prop.multiProcessorCount);
        printf("    name : %s\n", dev_prop.name);
        printf("    pageableMemoryAccess : %d\n", dev_prop.pageableMemoryAccess);
        printf("    pageableMemoryAccessUsesHostPageTables : %d\n", dev_prop.pageableMemoryAccessUsesHostPageTables);
        printf("    pciBusID : %d\n", dev_prop.pciBusID);
        printf("    pciDeviceID : %d\n", dev_prop.pciDeviceID);
        printf("    pciDomainID : %d\n", dev_prop.pciDomainID);
        printf("    persistingL2CacheMaxSize : %d\n", dev_prop.persistingL2CacheMaxSize);
        printf("    regsPerBlock : %d\n", dev_prop.regsPerBlock);
        printf("    regsPerMultiprocessor : %d\n", dev_prop.regsPerMultiprocessor);
        printf("    reservedSharedMemPerBlock : %zu\n", dev_prop.reservedSharedMemPerBlock);
        printf("    sharedMemPerBlock : %zu\n", dev_prop.sharedMemPerBlock);
        printf("    sharedMemPerBlockOptin : %zu\n", dev_prop.sharedMemPerBlockOptin);
        printf("    sharedMemPerMultiprocessor : %zu\n", dev_prop.sharedMemPerMultiprocessor);
        printf("    singleToDoublePrecisionPerfRatio : %d\n", dev_prop.singleToDoublePrecisionPerfRatio);
        printf("    streamPrioritiesSupported : %d\n", dev_prop.streamPrioritiesSupported);
        printf("    surfaceAlignment : %zu\n", dev_prop.surfaceAlignment);
        printf("    tccDriver : %d\n", dev_prop.tccDriver);
        printf("    textureAlignment : %zu\n", dev_prop.textureAlignment);
        printf("    texturePitchAlignment : %zu\n", dev_prop.texturePitchAlignment);
        printf("    totalConstMem : %zu\n", dev_prop.totalConstMem);
        printf("    totalGlobalMem : %zu\n", dev_prop.totalGlobalMem);
        printf("    unifiedAddressing : %d\n", dev_prop.unifiedAddressing);

        printf("    uuid : ");
        printCharArray(dev_prop.uuid.bytes, 16);
        printf("\n");
        // "   uuid : ", uuid
        printf("    warpSize : %d", dev_prop.warpSize);
    }
}