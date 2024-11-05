#include <curand_kernel.h>

extern "C" __global__ void generateKeys(unsigned char* keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize CUDA random number generator with better entropy
    curandState state;
    curand_init(clock64() + idx, threadIdx.x, blockIdx.x, &state);
    
    // Generate 32 bytes for private key with improved entropy
    unsigned char key[32];
    for(int i = 0; i < 32; i++) {
        // Mix multiple sources of entropy
        unsigned int r1 = curand(&state);
        unsigned int r2 = curand(&state);
        unsigned int r3 = clock64() ^ r1;
        key[i] = (r1 ^ r2 ^ r3) % 256;
    }
    
    // Write to global memory
    for(int i = 0; i < 32; i++) {
        keys[idx * 32 + i] = key[i];
    }
    
    __syncthreads();
}