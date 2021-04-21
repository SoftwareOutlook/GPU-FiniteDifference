#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

void fdiff_direct_1d(const float *inimagefull, float *outimageXfull,
                     int nx, int ny, int nz, int bc, int T)
{

  long c, ind, k, j, i;
  float pix0;
  size_t volume = nx;
  const float *inimage = inimagefull;
  float *outimageX = outimageXfull;

  #pragma omp parallel num_threads(T)
  {

    #pragma omp for
    for(ind = 0; ind < nx - 1; ind++) {
      pix0 = -inimage[ind];
      outimageX[ind] = pix0 + inimage[ind + 1];
    }

    // Neumann boundaries
    if(bc == 0) {
      #pragma omp for
      for(k = 0; k < nz; k++) {
        for(j = 0; j < ny; j++) {
          outimageX[k * ny * nx + j * nx + nx - 1] = 0;
        }
      }
    }

    // Periodic boundaries
    else if(bc == 1) {
      #pragma omp for
      for(k = 0; k < nz; k++) {
        for(j = 0; j < ny; j++) {
          int ind1 = k * ny * nx + j * nx;
	  int ind2 = ind1 + nx - 1;
	  outimageX[ind2] = -inimage[ind2] + inimage[ind1];
        }
      }
    }

    // Anything besides 0, 1 bc will ignore the boundary conditions

  }
}

void fdiff_direct_2d(const float *inimagefull, float *outimageXfull,
                     float *outimageYfull, int nx, int ny, int nz,
		     int bc, int T)
{

  long c, ind, k, j, i;
  float pix0;
  size_t volume = nx * ny * nz;
  const float *inimage = inimagefull;
  float *outimageX = outimageXfull;
  float *outimageY = outimageYfull;

  int offset1 = (nz - 1) * nx * ny;       // index of start of last slice
  int offset2 = offset1 + (ny - 1) * nx;  // index of start of last row

  #pragma omp parallel num_threads(T)
  {

    #pragma omp for
    for(ind = 0; ind < nx * ny * (nz - 1); ind++) {
      pix0 = -inimage[ind];
      outimageX[ind] = pix0 + inimage[ind + 1];
      outimageY[ind] = pix0 + inimage[ind + nx];
    }

    // Compute last row computation
    #pragma omp for
    for(ind = 0; ind < nx * (ny - 1); ind++) {
      pix0 = -inimage[ind + offset1];
      outimageX[ind + offset1] = pix0 + inimage[ind + offset1 + 1];
      outimageY[ind + offset1] = pix0 + inimage[ind + offset1 + nx];
    }

    // Compute final slice
    #pragma omp for
    for(ind = 0; ind < nx - 1; ind++) {
      pix0 = -inimage[ind + offset2];
      outimageX[ind + offset2] = pix0 + inimage[ind + offset2 + 1];
    }

    // Compute the boundary conditions
    // Neumann
    if(bc == 0) {
      #pragma omp for
      for(k = 0; k < nz; k++){
        for(i = 0; i < nx; i++) {
	  outimageY[(k * ny * nx) + (ny - 1) * nx + 1] = 0;
	}
      }

      #pragma omp for
      for(k = 0; k < nz; k++){
        for(j = 0; j < ny; j++) {
	  outimageX[k * ny * nx + j * nx + nx - 1] = 0;
	}
      }

    // Periodic boundaries
    } else if (bc == 1) {
      #pragma omp for
      for (k = 0; k < nz; k++)
      {
        for (i = 0; i < nx; i++)
	{
	  int ind1 = (k * ny * nx);
	  int ind2 = ind1 + (ny - 1) * nx;
	  outimageY[ind2 + i] = -inimage[ind2 + i] + inimage[ind1 + i];
	}
      }

      // process X boundaries by setting to 0
      #pragma omp for
      for (k = 0; k < nz; k++)
      {
	for (j = 0; j < ny; j++)
	{
	  int ind1 = k * ny * nx + j * nx;
	  int ind2 = ind1 + nx - 1;
	  outimageX[ind2] = -inimage[ind2] + inimage[ind1];
	}
      }
    }

    // Anything besides 0, 1 bc will ignore the boundary conditions

  }
}

void fdiff_direct_3d(const float *inimagefull, float *outimageXfull,
                     float *outimageYfull, float *outimageZfull,
		     int nx, int ny, int nz, int bc, int T)
{

  long c, ind, k, j, i;
  float pix0;
  size_t volume = nx * ny * nz;
  const float *inimage = inimagefull;
  float *outimageX = outimageXfull;
  float *outimageY = outimageYfull;
  float *outimageZ = outimageZfull;

  int offset1 = (nz - 1) * nx * ny;       // index of start of last slice
  int offset2 = offset1 + (ny - 1) * nx;  // index of start of last row

  #pragma omp parallel num_threads(T)
  {

    #pragma omp for
    for(ind = 0; ind < nx * ny * (nz - 1); ind++) {
      pix0 = -inimage[ind];
      outimageX[ind] = pix0 + inimage[ind + 1];
      outimageY[ind] = pix0 + inimage[ind + nx];
      outimageZ[ind] = pix0 + inimage[ind + nx * ny];
    }

    // Compute last row computation
    #pragma omp for
    for(ind = 0; ind < nx * (ny - 1); ind++) {
      pix0 = -inimage[ind + offset1];
      outimageX[ind + offset1] = pix0 + inimage[ind + offset1 + 1];
      outimageY[ind + offset1] = pix0 + inimage[ind + offset1 + nx];
    }

    // Compute final slice
    #pragma omp for
    for(ind = 0; ind < nx - 1; ind++) {
      pix0 = -inimage[ind + offset2];
      outimageX[ind + offset2] = pix0 + inimage[ind + offset2 + 1];
    }

    // Compute the boundary conditions
    // Neumann
    if(bc == 0) {
      #pragma omp for
      for(k = 0; k < nz; k++){
        for(i = 0; i < nx; i++) {
	  outimageY[(k * ny * nx) + (ny - 1) * nx + 1] = 0;
	}
      }

      #pragma omp for
      for(k = 0; k < nz; k++){
        for(j = 0; j < ny; j++) {
	  outimageX[k * ny * nx + j * nx + nx - 1] = 0;
	}
      }

      #pragma omp for
      for(ind = 0; ind < ny * nx; ind++) {
	outimageZ[nx * ny * (nz - 1) + ind] = 0;
      }

    // Periodic boundaries
    } else if (bc == 1) {
      #pragma omp for
      for (k = 0; k < nz; k++)
      {
        for (i = 0; i < nx; i++)
	{
	  int ind1 = (k * ny * nx);
	  int ind2 = ind1 + (ny - 1) * nx;
	  outimageY[ind2 + i] = -inimage[ind2 + i] + inimage[ind1 + i];
	}
      }

      // process X boundaries by setting to 0
      #pragma omp for
      for (k = 0; k < nz; k++)
      {
	for (j = 0; j < ny; j++)
	{
	  int ind1 = k * ny * nx + j * nx;
	  int ind2 = ind1 + nx - 1;
	  outimageX[ind2] = -inimage[ind2] + inimage[ind1];
	}
      }

      #pragma omp for
      for(ind = 0; ind < ny * nx; ind++) {
	outimageZ[nx * ny * (nz - 1) + ind] = -inimage[nx * ny * (nz - 1) + ind] + inimage[ind];
      }
    }

    // Anything besides 0, 1 bc will ignore the boundary conditions
  }
}
