void fdiff_gpu_1d_nm(float *inimage, float *outX_d,
                     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_1d_pd(float *inimage, float *outX_d,
		     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_1d_nb(float *inimage, float *outX_d,
		     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_2d_nm(float *inimage, float *outX_d, float *outY_d,
                     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_2d_pd(float *inimage, float *outX_d, float *outY_d,
		     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_2d_nb(float *inimage, float *outX_d, float *outY_d,
		     int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_3d_nm(float *inimage, float *outX_d, float *outY_d,
                     float *outZ_d, int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_3d_pd(float *inimage, float *outX_d, float *outY_d,
		     float *outZ_d, int nx, int ny, int nz, int bc, int T, int volume);

void fdiff_gpu_3d_nb(float *inimage, float *outX_d, float *outY_d,
		     float *outZ_d, int nx, int ny, int nz, int bc, int T, int volume);

void cdiff_k_1d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K, int volume);

void cdiff_k_2d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K, int volume);

void cdiff_k_3d(float *inimage, float *outX_d, float *outY_d,
                float *outZ_d, int nx, int ny, int nz, int bc, int T, int K, int volume);
