void fdiff_direct_1d(const float *inimagefull, float* outimageXfull,
		     int nx, int ny, int nz, int bc, int T);

void fdiff_direct_2d(const float *inimagefull, float* outimageXfull, float *outimageYfull,
		     int nx, int ny, int nz, int bc, int T);

void fdiff_direct_3d(const float *inimagefull, float* outimageXfull, float *outimageYfull,
		     float *outimageZfull, int nx, int ny, int nz, int bc, int T);
