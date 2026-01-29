#ifndef THREAD_COUNT
    #define THREAD_COUNT 256 // multiple of 32 for hardware efficiency reasons
#endif

// This is being added to make sure we can take advantage of memory coelscing (close memory accesses)
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 1024 // multiple of 32 for hardware efficiency reasons
#endif
