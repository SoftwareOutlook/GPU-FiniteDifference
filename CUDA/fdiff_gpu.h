#include <cuda.h>                                                                                                                        
#include <cuda_runtime.h>                                                                                                                                                                                                                                                         
__global__ void initialiseArrays(float *X, float *Y, float *Z, float *inimage, int nx, int ny, int nz);

void printGrids(float*, float*, float*, int, int, int);

void cuMain(float *inimage, float *X, float *Y, float *Z, int sx, int sy, int sz, int bc, int dim, int T);
