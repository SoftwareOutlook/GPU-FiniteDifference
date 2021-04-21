#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "fdiff.h"
#include "fdiff_gpu.h"

#define RMAX 5
#define RMIN -5
#define PRINT_TIMINGS 1

int main(int argc, char** argv) {

  // Variable initialisation
  int sizeX = 64;     // Size of input image (X)
  int sizeY = 64;     // Size of input image (Y)
  int sizeZ = 64;     // Size of input image (Z)
  int dim   = 3;      // The number of dimensions of the input image
  int bc    = 0;      // Boundary condition selection {0 - neumann, 1 - periodic, 2 - ignore boundaries}
  int T     = 4;      // Number of OMP threads

  // The total volume of the input image
  int volume = sizeX * sizeY * sizeZ;

  // Array allocation for host image and X,Y,Z difference output
  float *inimage = (float *)malloc(volume * sizeof(float));
  float *outX = (float *)malloc(volume * sizeof(float));
  float *outY = (float *)malloc(volume * sizeof(float));
  float *outZ = (float *)malloc(volume * sizeof(float));

  // Allocate arrays that will be modified on the device
  float *x_d = (float *)malloc(volume * sizeof(float));
  float *y_d = (float *)malloc(volume * sizeof(float));
  float *z_d = (float *)malloc(volume * sizeof(float));

  // Random initialisation of the input array
  for(int i=0; i < volume; i++) {
    inimage[i] = rand() % (RMAX + 1 - RMIN) + RMIN;
  }

  double start = omp_get_wtime();

  // CPU FDM computation
  if(dim == 1) {
    fdiff_direct_1d(inimage, outX, sizeX, sizeY, sizeZ, bc, T);  // X derivates
  } else if(dim == 2) {
    fdiff_direct_2d(inimage, outX, outY, sizeX, sizeY, sizeZ, bc, T); // XY derivatives
  } else if(dim == 3) {
    fdiff_direct_3d(inimage, outX, outY, outZ, sizeX, sizeY, sizeZ, bc, T); // XYZ derivatives
  }

  double end = omp_get_wtime();

  printf("GPU:\n");

  omp_set_default_device(0);   // Set which GPU in your environment to offload execution onto

  /* OpenMP device data mapping */
  #pragma omp target enter data map(to:inimage[0:volume]) \
                                map(alloc:x_d[0:volume],  \
                                          y_d[0:volume],  \
                                          z_d[0:volume])
  start = omp_get_wtime();

  if(bc == 0) {
    if(dim == 1) {
      fdiff_gpu_1d_nm(inimage, x_d, sizeX, sizeY, sizeZ, bc, T);  // X derivatives
    } else if(dim == 2) {
      fdiff_gpu_2d_nm(inimage, x_d, y_d, sizeX, sizeY, sizeZ, bc, T); // XY derivatives
    } else if(dim == 3) {
      fdiff_gpu_3d_nm(inimage, x_d, y_d, z_d, sizeX, sizeY, sizeZ, bc, T); // XYZ derivatives
    }
  }

  else if(bc == 1) {
    if(dim == 1) {
      fdiff_gpu_1d_pd(inimage, x_d, sizeX, sizeY, sizeZ, bc, T);  // X derivatives
    } else if(dim == 2) {
      fdiff_gpu_2d_pd(inimage, x_d, y_d, sizeX, sizeY, sizeZ, bc, T); // XY derivatives
    }
    else if(dim == 3) {
      fdiff_gpu_3d_pd(inimage, x_d, y_d, z_d, sizeX, sizeY, sizeZ, bc, T); // XYZ derivatives
    }
  }

  else {
    if(dim == 1) {
      fdiff_gpu_1d_nb(inimage, x_d, sizeX, sizeY, sizeZ, bc, T);  // X derivatives
    } else if(dim == 2) {
      fdiff_gpu_2d_nb(inimage, x_d, y_d, sizeX, sizeY, sizeZ, bc, T); // XY derivatives
    }
    else if(dim == 3) {
      fdiff_gpu_3d_nb(inimage, x_d, y_d, z_d, sizeX, sizeY, sizeZ, bc, T); // XYZ derivatives
    }
  }

  end = omp_get_wtime();

  /* OpenMP exit data mapping - map output arrays back to host for verification */
  #pragma omp target exit data map(from:x_d[0:volume], \
				        y_d[0:volume], \
                                        z_d[0:volume])

  if(PRINT_TIMINGS == 1) printf("GPU Run:\t%lf\n", end - start);

  verifyResults(outX, outY, outZ, x_d, y_d, z_d, sizeX, sizeY, sizeZ);

  return 0;

}
