#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

void fdiff_gpu_1d_nm(float *inimage, float *outX_d,
                     int nx, int ny, int nz, int bc, int T)
{

  #pragma omp target teams
  {
    #pragma omp distribute parallel for simd
    for(int ind = 0; ind < nx - 1; ind++) {
      outX_d[ind] = -inimage[ind] + inimage[ind + 1];
    }

    #pragma omp distribute parallel for simd
    for(int j = 0; j < ny; j++) {
	outX_d[ny * nx + j * nx + nx - 1] = 0;
    }
  }

}

void fdiff_gpu_1d_pd(float *inimage, float *outX_d,
                     int nx, int ny, int nz, int bc, int T)
{

  #pragma omp target teams
  {
    #pragma omp distribute parallel for simd
    for(int ind = 0; ind < nx - 1; ind++) {
      outX_d[ind] = -inimage[ind] + inimage[ind + 1];
    }

    #pragma omp distribute parallel for simd
    for(int j = 0; j < ny; j++) {
	int ind1 = ny * nx + j * nx;
	int ind2 = ind1 + nx - 1;
	outX_d[ind2] = -inimage[ind2] + inimage[ind1];
    }
  }

}

void fdiff_gpu_1d_nb(float *inimage, float *outX_d,
                     int nx, int ny, int nz, int bc, int T)
{

  #pragma omp target teams
  {
    #pragma omp distribute parallel for simd
    for(int ind = 0; ind < nx - 1; ind++) {
      outX_d[ind] = -inimage[ind] + inimage[ind + 1];
    }
  }

}

void fdiff_gpu_2d_nm(float *inimage, float *outX_d, float *outY_d,
                     int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }

      for(int k = 0; k < nz; k++) {
        #pragma omp distribute parallel for simd
	for(int j = 0; j < ny; j++) {
	  outX_d[k * ny * nx + j * nx + nx - 1] = 0;
	}

	#pragma omp distribute parallel for simd
	for(int i = 0; i < nx; i++) {
	  outY_d[(k * ny * nx) + (ny - 1) * nx + 1] = 0;
	}
      }
    }
  }
}

void fdiff_gpu_2d_pd(float *inimage, float *outX_d, float *outY_d,
                     int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }

      for(int k = 0; k < nz; k++) {
        #pragma omp distribute parallel for simd
	for(int j = 0; j < ny; j++) {
	  int ind1 = k * ny * nx + j * nx;
	  int ind2 = ind1 + nx - 1;
	  outX_d[ind2] = -inimage[ind2] + inimage[ind1];
	}

        #pragma omp distribute parallel for simd
	for(int i = 0; i < nx; i++) {
	  int ind1 = (k * ny * nx);
	  int ind2 = ind1 + (ny - 1) * nx;
	  outY_d[ind2 + i] = -inimage[ind2 + i] + inimage[ind1 + i];
	}
      }
    }
  }
}

void fdiff_gpu_2d_nb(float *inimage, float *outX_d, float *outY_d,
                     int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }
    }
  }
}

void fdiff_gpu_3d_nm(float *inimage, float *outX_d, float *outY_d,
                     float *outZ_d, int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
	outZ_d[ind] = -inimage[ind] + inimage[ind + nx * ny];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }

      for(int k = 0; k < nz; k++){
        #pragma omp distribute parallel for simd
	for(int j = 0; j < ny; j++) {
	  outX_d[k * ny * nx + j * nx + nx - 1] = 0;
	}
      }

      for(int k = 0; k < nz; k++){
        #pragma omp distribute parallel for simd
	for(int i = 0; i < nx; i++) {
	  outY_d[(k * ny * nx) + (ny - 1) * nx + 1] = 0;
	}
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < ny * nx; ind++) {
	outZ_d[nx * ny * (nz - 1) + ind] = 0;
      }
    }
  }
}

void fdiff_gpu_3d_pd(float *inimage, float *outX_d, float *outY_d,
                     float *outZ_d, int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
	outZ_d[ind] = -inimage[ind] + inimage[ind + nx * ny];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }


      for(int k = 0; k < nz; k++){
        #pragma omp distribute parallel for simd
	for(int j = 0; j < ny; j++) {
	  int ind1 = k * ny * nx + j * nx;
	  int ind2 = ind1 + nx - 1;
	  outX_d[ind2] = -inimage[ind2] + inimage[ind1];
	}
      }

      for(int k = 0; k < nz; k++){
        #pragma omp distribute parallel for simd
	for(int i = 0; i < nx; i++) {
	  int ind1 = (k * ny * nx);
	  int ind2 = ind1 + (ny - 1) * nx;
	  outY_d[ind2 + i] = -inimage[ind2 + i] + inimage[ind1 + i];
	}
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < ny * nx; ind++) {
	outZ_d[nx * ny * (nz - 1) + ind] = -inimage[nx * ny * (nz - 1) + ind] + inimage[ind];
      }
    }
  }
}

void fdiff_gpu_3d_nb(float *inimage, float *outX_d, float *outY_d,
                     float *outZ_d, int nx, int ny, int nz, int bc, int T)
{

  int offset1 = (nz - 1) * nx * nz;
  int offset2 = offset1 + (ny - 1) * nx;

  #pragma omp target
  {
    #pragma omp teams
    {
      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * ny * (nz - 1); ind++) {
	outX_d[ind] = -inimage[ind] + inimage[ind + 1];
	outY_d[ind] = -inimage[ind] + inimage[ind + nx];
	outZ_d[ind] = -inimage[ind] + inimage[ind + nx * ny];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx * (ny - 1); ind++) {
	outX_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + 1];
	outY_d[ind + offset1] = -inimage[ind + offset1] + inimage[ind + offset1 + nx];
      }

      #pragma omp distribute parallel for simd
      for(int ind = 0; ind < nx - 1; ind++) {
	outX_d[ind + offset2] = -inimage[ind + offset2] + inimage[ind + offset2 + 1];
      }
    }
  }
}


void cdiff_k_1d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K)
{

  #pragma omp target teams
  {
    #pragma omp distribute parallel for
    for(int ind = K; ind < nx - K; ind++) {

      float sumx = 0;

      #pragma omp simd reduction(+:sumx)
      for(int k=1; k <= K; k++) {
        sumx += inimage[ind + k] - inimage[ind - k];
      }

      outX_d[ind] = sumx;
    }
  }
}


void cdiff_k_2d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K)
{

  #pragma omp target teams
  {
    #pragma omp distribute parallel for
    for(int ind = K * nx ; ind < nx * ny - (K * nx); ind++) {

      float sumx = 0, sumy = 0;

      #pragma omp simd reduction(+:sumx, sumy)
      for(int k=1; k <= K; k++) {
	sumx += inimage[ind + k] - inimage[ind - k];
	sumy += inimage[ind + (k * nx)] -
	        inimage[ind - (k * nx)];
      }

      outX_d[ind] = sumx;
      outY_d[ind] = sumy;

    }
  }
}


void cdiff_k_3d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K)
{

  int s = 0;

  #pragma omp target teams firstprivate(K, nx, ny, nz, s)
  {
    for(s = K * nx * ny; s < nx * ny * (nz - K); s+=nx*ny) {
      #pragma omp distribute parallel for
      for(int ind = K * nx ; ind < nx * ny - (K * nx); ind++) {
	float sumx = 0, sumy = 0, sumz = 0;

        #pragma omp simd reduction(+:sumx, sumy, sumz)
	for(int k=1; k <= K; k++) {
	  sumx += inimage[s + ind + k] - inimage[s + ind - k];
	  sumy += inimage[s + ind + (k * nx)] -
	          inimage[s + ind - (k * nx)];
	  sumz += inimage[s + ind + (k * nx * ny)] -
	          inimage[s + ind - (k * nx * ny)];
	}

	outX_d[s + ind] = sumx;
	outY_d[s + ind] = sumy;
	outZ_d[s + ind] = sumz;

      }
    }
  }
}
