#ifndef THREAD_COUNT
    #define THREAD_COUNT 256 // multiple of 32 for hardware efficiency reasons
#endif

// This is being added to make sure we can take advantage of memory coelscing (close memory accesses)
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 32 // multiple of 32 for hardware efficiency reasons - CUDA hard limit per block is 1024 threads i think so 32 by 32 is max - TODO: check this otherwise make difference consts forr linear, 2D and 3D kernel blocks
#endif
