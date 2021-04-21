#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "fdiff.h"

#define RMAX 5
#define RMIN -5
#define PRINT_TIMINGS 1

int main(int argc, char** argv) {

  // Variable initialisation
  int sizeX = 8;    // Size of input image (X)
  int sizeY = 8;    // Size of input image (Y)
  int sizeZ = 1;    // Size of input image (Z)
  int dim   = 2;    // The number of dimensions of the input image
  int bc    = 1;    // Boundary condition selection {0, 1, 2}
  int T     = 4;    // Number of OMP threads

  // The total volume of the input image
  int volume = sizeX * sizeY * sizeZ;

  // Array allocation for image and X,Y,Z difference output
  float *inimage = (float *)malloc(volume * sizeof(float));
  float *outX = (float *)malloc(volume * sizeof(float));
  float *outY = (float *)malloc(volume * sizeof(float));
  float *outZ = (float *)malloc(volume * sizeof(float));

  // Random initialisation of the input array
  for(int i=0; i < volume; i++) {
    inimage[i] = rand() % (RMAX + 1 - RMIN) + RMIN;
  }

  double start = omp_get_wtime();

  if(dim == 1) {
    fdiff_direct_1d(inimage, outX, sizeX, sizeY, sizeZ, bc, T);  // X derivates
  } else if(dim == 2) {
    fdiff_direct_2d(inimage, outX, outY, sizeX, sizeY, sizeZ, bc, T); // XY derivatives
  } else if(dim == 3) {
    fdiff_direct_3d(inimage, outX, outY, outZ, sizeX, sizeY, sizeZ, bc, T); // XYZ derivatives
  }

  double end = omp_get_wtime();

  if(PRINT_TIMINGS == 1) printf("Run:\t%lf\n", end - start);

  return 0;

}
