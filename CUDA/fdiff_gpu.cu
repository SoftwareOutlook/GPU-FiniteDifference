#define SIZEX 32 // SHARED MEMORY TILE X size
#define SIZEY 32 // SHARED MEMORY TILE Y size (must be set to 1 for 1dimensional case)
#define SIZEZ 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <omp.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "fdiff_gpu.h"

__global__ void initialiseArrays(float *X, float *Y, float *Z, float *inimage,
				 int nx, int ny, int nz) {

  for(int j=0; j<ny; j++) {
    for(int i=0; i<nx; i++) {
      X[j*ny+i] = 0;
      Y[j*ny+i] = 0;
      Z[j*ny+i] = 0;
    }
  }
}

__global__ void fd_x_nm( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX+1][SIZEY];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  for(int i=0; i<nz; i++) {

    s_i[tx][ty] =  inimage[idx];

    if(tx == SIZEX-1) s_i[tx+1][ty] = inimage[idx+1];

    __syncthreads();

    X[idx] = s_i[tx+1][ty] - s_i[tx][ty];

    if(ix == nx-1) X[idx] = 0;

    __syncthreads();

    idx += stride;

  }

    return;
}


__global__ void fd_x_pd( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX+1][SIZEY];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  for(int i=0; i<nz; i++) {

    s_i[tx][ty] =  inimage[idx];

    if(tx == SIZEX-1) s_i[tx+1][ty] = inimage[idx+1];

    if(ix == nx-1) s_i[tx+1][ty] = inimage[idx-(nx-1)];

    __syncthreads();

    X[idx] = s_i[tx+1][ty] - s_i[tx][ty];

    __syncthreads();

    idx += stride;

  }

    return;
}

__global__ void fd_x_nb( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX+1][SIZEY];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  s_i[tx][ty] =  inimage[idx];

  if(tx == SIZEX-1) s_i[tx+1][ty] = inimage[idx+1];

  __syncthreads();

  X[idx] = s_i[tx+1][ty] - s_i[tx][ty];

  return;
}

__global__ void fd_y_nm( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  for(int i=0; i<nz; i++) {

    s_i[tx][ty] =  inimage[idx];

    if(ty == SIZEY-1) s_i[tx][ty+1] = inimage[idx+nx];

    __syncthreads();

    Y[idx] = s_i[tx][ty+1] - s_i[tx][ty];

    __syncthreads();

    if(iy == ny-1) Y[idx] = 0;

    idx += stride;

  }

    return;
}

__global__ void fd_y_pd( float *X,
                        float *Y,
                        float *Z,
                        float *inimage,
                        int nx,
                        int ny,
                        int nz,
                        int bc,
                        int vol) {

  __shared__ float s_i[SIZEX][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  for(int i=0; i<nz; i++) {

    s_i[tx][ty] =  inimage[idx];

    if(ty == SIZEY-1) s_i[tx][ty+1] = inimage[idx+nx];

    if(iy == ny-1) s_i[tx][ty+1] = inimage[i*(nx*ny) + ix];

    __syncthreads();

    Y[idx] = s_i[tx][ty+1] - s_i[tx][ty];

    __syncthreads();

    idx += stride;

  }

    return;

}

__global__ void fd_y_nb( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;

  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  for(int i=0; i<nz; i++) {

    s_i[tx][ty] =  inimage[idx];

    if(ty == SIZEY-1) s_i[tx][ty+1] = inimage[idx+nx];

    __syncthreads();

    Y[idx] = s_i[tx][ty+1] - s_i[tx][ty];

    __syncthreads();

    idx += stride;

  }

  return;

}

__global__ void fd_z_nm( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {


  __shared__ float s_i[SIZEX][SIZEY];
  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt;

  cur = inimage[idx];
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[tx][ty] = cur;

    __syncthreads();

    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;
    odx += stride;
    cur = inimage[odx];
    nxt = inimage[idx];

  }

  Z[odx] = 0;

  return;
}

__global__ void fd_z_pd( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX][SIZEY];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt, og;

  og  = inimage[idx];
  cur = og;
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;

    odx += stride;

    cur = inimage[odx];

    nxt = inimage[idx];
  }

  Z[odx] = og - cur;

  return;
}

__global__ void fd_z_nb( float *X,
                         float *Y,
                         float *Z,
                         float *inimage,
                         int nx,
                         int ny,
                         int nz,
                         int bc,
                         int vol) {

  __shared__ float s_i[SIZEX][SIZEY];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt, og;

  og  = inimage[idx];
  cur = og;
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;

    odx += stride;

    cur = inimage[odx];

    nxt = inimage[idx];
  }

  return;
}


__global__ void fd_xy_nm( float *X,
                           float *Y,
                           float *Z,
                           float *inimage,
                           int nx,
                           int ny,
                           int nz,
                           int bc,
                           int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  float cur;
  cur = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[idx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[idx+nx];

    if(ix == nx-1) s_i[ty][tx+1] = cur;
    if(iy == nx-1) s_i[ty+1][tx] = cur;

    __syncthreads();

    X[idx] = s_i[ty][tx+1] - cur;
    Y[idx] = s_i[ty+1][tx] - cur;

    __syncthreads();
  }

  s_i[ty][tx] = cur;

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[idx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[idx+nx];

  if(ix == nx - 1) s_i[ty][tx + 1] = cur;
  if(iy == ny - 1) s_i[ty + 1][tx] = cur;

  __syncthreads();

  X[idx] = s_i[ty][tx + 1] - cur;
  Y[idx] = s_i[ty + 1][tx] - cur;

  return;
}

__global__ void fd_xy_pd( float *X,
			  float *Y,
			  float *Z,
			  float *inimage,
			  int nx,
			  int ny,
			  int nz,
			  int bc,
			  int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  float cur;
  cur = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[idx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[idx+nx];

    if(ix == nx-1) s_i[ty][tx+1] = inimage[idx - (nx - 1)];
    if(iy == nx-1) s_i[ty+1][tx] = inimage[i * (nx * ny) + ix];

    __syncthreads();

    X[idx] = s_i[ty][tx+1] - cur;
    Y[idx] = s_i[ty+1][tx] - cur;

    __syncthreads();
  }

  s_i[ty][tx] = cur;

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[idx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[idx+nx];

  if(ix == nx-1) s_i[ty][tx+1] = inimage[idx - (nx - 1)];
  if(iy == nx-1) s_i[ty+1][tx] = inimage[(nz - 1) * (nx * ny) + ix];

  __syncthreads();

  X[idx] = s_i[ty][tx + 1] - cur;
  Y[idx] = s_i[ty + 1][tx] - cur;

  return;
}


__global__ void fd_xy_nb( float *X,
			  float *Y,
			  float *Z,
			  float *inimage,
			  int nx,
			  int ny,
			  int nz,
			  int bc,
			  int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;

  int stride = nx*ny;

  float cur;
  cur = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[idx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[idx+nx];

    __syncthreads();

    X[idx] = s_i[ty][tx+1] - cur;
    Y[idx] = s_i[ty+1][tx] - cur;

    __syncthreads();
  }

  s_i[ty][tx] = cur;

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[idx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[idx+nx];

  __syncthreads();

  X[idx] = s_i[ty][tx + 1] - cur;
  Y[idx] = s_i[ty + 1][tx] - cur;

  return;
}



__global__ void fd_xyz_nm( float *X,
                           float *Y,
                           float *Z,
                           float *inimage,
                           int nx,
                           int ny,
                           int nz,
                           int bc,
                           int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt, og;

  og  = inimage[idx];
  cur = og;
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    __syncthreads();

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[odx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[odx+nx];

    if(ix == nx-1) s_i[ty][tx+1] = cur;

    __syncthreads();

    X[odx] = s_i[ty][tx+1] - cur;
    Y[odx] = s_i[ty+1][tx] - cur;
    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;
    odx += stride;
    cur = inimage[odx];
    nxt = inimage[idx];

  }

  s_i[ty][tx] = cur;

  __syncthreads();

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[odx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[odx+nx];

  if(ix == nx - 1) s_i[ty][tx + 1] = cur;
  if(iy == ny - 1) s_i[ty+1][tx] = cur;

  __syncthreads();

  X[odx] = s_i[ty][tx + 1] - cur;
  Y[odx] = s_i[ty + 1][tx] - cur;
  Z[odx] = 0;

  return;
}

__global__ void fd_xyz_pd( float *X,
                           float *Y,
                           float *Z,
                           float *inimage,
                           int nx,
                           int ny,
                           int nz,
                           int bc,
                           int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt, og;

  og  = inimage[idx];
  cur = og;
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[odx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[odx+nx];

    if(ix == nx-1) s_i[ty][tx+1] = inimage[odx - (nx - 1)];

    __syncthreads();

    X[odx] = s_i[ty][tx+1] - cur;
    Y[odx] = s_i[ty+1][tx] - cur;
    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;
    odx += stride;
    cur = inimage[odx];
    nxt = inimage[idx];

  }

  s_i[ty][tx] = cur;

  __syncthreads();

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[odx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[odx+nx];

  if(ix == nx-1) s_i[ty][tx+1] = inimage[odx - (nx - 1)];
  if(iy == nx-1) s_i[ty+1][tx] = inimage[(nz - 1) * (nx * ny) + ix];

  __syncthreads();

  X[odx] = s_i[ty][tx + 1] - cur;
  Y[odx] = s_i[ty + 1][tx] - cur;
  Z[odx] = og - cur;

  return;
}

__global__ void fd_xyz_nb( float *X,
                           float *Y,
                           float *Z,
                           float *inimage,
                           int nx,
                           int ny,
                           int nz,
                           int bc,
                           int volume ) {

  __shared__ float s_i[SIZEX+1][SIZEY+1];

  int ix  = blockIdx.x * blockDim.x + threadIdx.x;
  int iy  = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int idx = (iy * nx) + ix;
  int odx;

  int stride = nx*ny;

  float cur, nxt, og;

  og  = inimage[idx];
  cur = og;
  odx = idx;
  idx += stride;
  nxt = inimage[idx];

  for(int i=0; i<nz-1; i++) {

    s_i[ty][tx] = cur;

    __syncthreads();

    if(tx == SIZEX-1) s_i[ty][tx+1] = inimage[odx+1];
    if(ty == SIZEY-1) s_i[ty+1][tx] = inimage[odx+nx];

    __syncthreads();

    X[odx] = s_i[ty][tx+1] - cur;
    Y[odx] = s_i[ty+1][tx] - cur;
    Z[odx] = nxt - cur;

    __syncthreads();

    idx += stride;
    odx += stride;
    cur = inimage[odx];
    nxt = inimage[idx];

  }

  s_i[ty][tx] = cur;

  if(tx == SIZEX - 1) s_i[ty][tx + 1] = inimage[odx+1];
  if(ty == SIZEY - 1) s_i[ty + 1][tx] = inimage[odx+nx];

  __syncthreads();

  X[odx] = s_i[ty][tx + 1] - cur;
  Y[odx] = s_i[ty + 1][tx] - cur;
  Z[odx] = nxt-cur;

  return;
}


void cuMain(float *inimage, float *X, float *Y, float *Z,
	    int sx, int sy, int sz, int bc, int dim, int T) {

  cudaSetDevice(0);

  float time;
  cudaEvent_t start, stop;
  float *d_in, *d_x, *d_y, *d_z;
  float *x_arr, *y_arr, *z_arr;

  int volume = sx*sy*sz;

  dim3 dimBlock(SIZEX, SIZEY, 1);
  dim3 dimGrid(sx/SIZEX, sy/SIZEY, 1);

  x_arr = (float *)malloc(volume * sizeof(float));
  y_arr = (float *)malloc(volume * sizeof(float));
  z_arr = (float *)malloc(volume * sizeof(float));

  cudaMalloc((void**)&d_in, volume * sizeof(float));
  cudaMalloc((void**)&d_x, volume * sizeof(float));
  cudaMalloc((void**)&d_y, volume * sizeof(float));
  cudaMalloc((void**)&d_z, volume * sizeof(float));

  cudaMemcpy(d_in, inimage, volume * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int i=0; i<1; i++) {

    cudaEventRecord(start, 0);

    if(bc == 0) {
      if(dim == 1) {
	fd_x_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 2) {
	//fd_x_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	//fd_y_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	fd_xy_nm<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 3) {
	// fd_x_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_y_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_z_nm<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);

	fd_xyz_nm<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }
    }

    else if(bc == 1) {
      if(dim == 1) {
	fd_x_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 2) {
	//fd_x_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	//fd_y_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	fd_xy_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 3) {
	// fd_x_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_y_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_z_pd<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	fd_xyz_pd<<<dimGrid, dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }
    }

    else if(bc == 2) {
      if(dim == 1) {
	fd_x_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 2) {
	// fd_x_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_y_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	fd_xy_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }

      if(dim == 2) {
	// fd_x_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_y_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	// fd_z_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
	fd_xyz_nb<<<dimGrid,dimBlock>>>(d_x, d_y, d_z, d_in, sx, sy, sz, bc, volume);
      }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time taken: %lf\n", time/1000);

  }

  cudaMemcpy(x_arr,   d_x,  volume*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y_arr,   d_y,  volume*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(z_arr,   d_z,  volume*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(inimage, d_in, volume*sizeof(float), cudaMemcpyDeviceToHost);

  free(X);
  free(Y);
  free(Z);

  free(x_arr);
  free(y_arr);
  free(z_arr);

  cudaFree(d_in);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return;

}
