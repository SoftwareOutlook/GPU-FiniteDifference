#define SIZEX 16
#define SIZEY 16

kernel void fd_x_nm(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ, const unsigned int n)
{

  int ix = get_global_id(0);
  int iy = get_global_id(1);
  int id = (iy * sizeX) + ix;

  int tx = get_local_id(0);
  int ty = get_local_id(1);

  __local float s_i[SIZEX+1];

  int stride = sizeX * sizeY;

  for(int i=0; i<sizeZ; i++) {

    s_i[tx] = inimage[id];

    if(tx == SIZEX-1 && id < n - 1) s_i[tx+1] = inimage[id + 1];

    X[id] = s_i[tx + 1] - s_i[tx];

    if(id == sizeX - 1) X[id] = 0;

    barrier(CLK_GLOBAL_MEM_FENCE);

    id += stride;

  }
}


kernel void fd_xy_nm(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;
    int od;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX + 1][SIZEY + 1];

    int stride = sizeX * sizeY;

    float cur = inimage[id];

    s_i[tx][ty] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tx == SIZEX - 1 && id < n - 1) s_i[tx+1][ty] = inimage[id + 1];
    if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty+1] = inimage[id + sizeX];

    if(ix == sizeX - 1) s_i[tx+1][ty] = cur;
    if(iy == sizeY - 1) s_i[tx][ty+1] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    X[id] = s_i[tx+1][ty] - cur;
    Y[id] = s_i[tx][ty+1] - cur;

}




kernel void fd_xyz_nm(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

  int ix = get_global_id(0);
  int iy = get_global_id(1);
  int id = (iy * sizeX) + ix;
  int od;

  int tx = get_local_id(0);
  int ty = get_local_id(1);

  __local float s_i[SIZEX + 1][SIZEY + 1];

  int stride = sizeX * sizeY;

  float cur, nxt, og;

  og = inimage[id];
  cur = og;
  od = id;
  id += stride;
  nxt = inimage[id];

  for(int i = 0; i < sizeZ - 1; i++) {

    s_i[tx][ty] = cur;

    if(tx == SIZEX - 1 && id < n - 1) s_i[tx+1][ty] = inimage[od + 1];
    if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty+1] = inimage[od + sizeX];

    if(ix == sizeX - 1) s_i[tx+1][ty] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    X[od] = s_i[tx + 1][ty] - cur;
    Y[od] = s_i[tx][ty + 1] - cur;
    Z[od] = nxt - cur;

    barrier(CLK_GLOBAL_MEM_FENCE);

    id += stride;
    od += stride;
    cur = inimage[od];
    nxt = inimage[id];

    barrier(CLK_GLOBAL_MEM_FENCE);

  }

  s_i[tx][ty] = cur;

  barrier(CLK_LOCAL_MEM_FENCE);

  if(tx == SIZEX - 1) s_i[tx + 1][ty] = inimage[od + 1];
  if(ty == SIZEY - 1) s_i[tx][ty + 1] = inimage[od + sizeX];

  if(ix == sizeX - 1) s_i[tx + 1][ty] = cur;
  if(iy == sizeY - 1) s_i[tx][ty + 1] = cur;

  barrier(CLK_LOCAL_MEM_FENCE);

  X[od] = s_i[tx + 1][ty] - cur;
  Y[od] = s_i[tx][ty + 1] - cur;
  Z[od] = 0;

}

kernel void fd_x_pd(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ, const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX+1];

    int stride = sizeX * sizeY;

    for(int i=0; i<sizeZ; i++) {

        s_i[tx] = inimage[id];

	if(tx == SIZEX-1 && id < n - 1) s_i[tx+1] = inimage[id + 1];

	if(id == sizeX - 1) s_i[tx + 1] = inimage[id - (sizeX - 1)];

	barrier(CLK_LOCAL_MEM_FENCE);

	X[id] = s_i[tx + 1] - s_i[tx];

	barrier(CLK_GLOBAL_MEM_FENCE);

	id += stride;

    }
}

kernel void fd_xy_pd(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;
    int od;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX + 1][SIZEY + 1];

    int stride = sizeX * sizeY;

    float cur = inimage[id];

    s_i[tx][ty] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tx == SIZEX - 1 && id < n - 1) s_i[tx+1][ty] = inimage[id + 1];
    if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty+1] = inimage[id + sizeX];

    if(ix == sizeX - 1) s_i[tx+1][ty] = inimage[id - (sizeX - 1)];
    if(iy == sizeY - 1) s_i[tx][ty+1] = inimage[ix];

    barrier(CLK_LOCAL_MEM_FENCE);

    X[id] = s_i[tx+1][ty] - cur;
    Y[id] = s_i[tx][ty+1] - cur;

}

kernel void fd_xyz_pd(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;
    int od;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX + 1][SIZEY + 1];

    int stride = sizeX * sizeY;

    float cur, nxt, og;

    og = inimage[id];
    cur = og;
    od = id;
    id += stride;
    nxt = inimage[id];

    for(int i = 0; i < sizeZ - 1; i++) {

        s_i[tx][ty] = cur;

	if(tx == SIZEX - 1 && id < n - 1) s_i[tx + 1][ty] = inimage[od + 1];
        if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty + 1] = inimage[od + sizeX];

        if(ix == sizeX - 1) s_i[tx + 1][ty] = inimage[od - (sizeX - 1)];
        if(iy == sizeY - 1) s_i[tx][ty + 1] = inimage[i * (sizeX * sizeY) + ix];

    	barrier(CLK_LOCAL_MEM_FENCE);

    	X[od] = s_i[tx + 1][ty] - cur;
    	Y[od] = s_i[tx][ty + 1] - cur;
    	Z[od] = nxt - cur;

    	barrier(CLK_GLOBAL_MEM_FENCE);

    	id += stride;
    	od += stride;
    	cur = inimage[od];
    	nxt = inimage[id];

    	barrier(CLK_GLOBAL_MEM_FENCE);

    }


    s_i[tx][ty] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tx == SIZEX - 1) s_i[tx + 1][ty] = inimage[od + 1];
    if(ty == SIZEY - 1) s_i[tx][ty + 1] = inimage[od + sizeX];

    if(ix == sizeX - 1) s_i[tx + 1][ty] = inimage[od - (sizeX - 1)];;
    if(iy == sizeY - 1) s_i[tx][ty + 1] = inimage[(sizeZ - 1) * (sizeX * sizeY) + ix];;

    barrier(CLK_LOCAL_MEM_FENCE);

    X[od] = s_i[tx + 1][ty] - cur;
    Y[od] = s_i[tx][ty + 1] - cur;
    Z[od] = og - cur;

}

kernel void fd_x_nb(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ, const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int idx = (iy * sizeX) + ix;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX+1];

    int stride = sizeX * sizeY;

    for(int i=0; i<sizeZ; i++) {

        s_i[tx] = inimage[idx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if(tx == SIZEX - 1 && idx < n - 1) s_i[tx+1] = inimage[idx + 1];

	X[idx] = s_i[tx + 1] - s_i[tx];

	barrier(CLK_GLOBAL_MEM_FENCE);

	idx += stride;

    }
}

kernel void fd_xy_nb(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;
    int od;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX + 1][SIZEY + 1];

    int stride = sizeX * sizeY;

    float cur = inimage[id];

    s_i[tx][ty] = cur;

    barrier(CLK_LOCAL_MEM_FENCE);

    if(tx == SIZEX - 1 && id < n - 1) s_i[tx + 1][ty] = inimage[id + 1];
    if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty + 1] = inimage[id + sizeX];

    barrier(CLK_LOCAL_MEM_FENCE);

    X[id] = s_i[tx + 1][ty] - cur;
    Y[id] = s_i[tx][ty + 1] - cur;

}

kernel void fd_xyz_nb(__global float *X, __global float *Y,
		    __global float *Z, __global float *inimage,
		    int sizeX, int sizeY, int sizeZ,
		    const unsigned int n)
{

    int ix = get_global_id(0);
    int iy = get_global_id(1);
    int id = (iy * sizeX) + ix;
    int od;

    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float s_i[SIZEX + 1][SIZEY + 1];

    int stride = sizeX * sizeY;

    float cur, nxt, og;

    og = inimage[id];
    cur = og;
    od = id;
    id += stride;
    nxt = inimage[id];

    for(int i = 0; i < sizeZ - 1; i++) {

        s_i[tx][ty] = cur;

	if(tx == SIZEX - 1 && id < n - 1) s_i[tx + 1][ty] = inimage[od + 1];
        if(ty == SIZEY - 1 && id < n - sizeX) s_i[tx][ty + 1] = inimage[od + sizeX];

    	barrier(CLK_LOCAL_MEM_FENCE);

    	X[od] = s_i[tx + 1][ty] - cur;
    	Y[od] = s_i[tx][ty + 1] - cur;
    	Z[od] = nxt - cur;

    	barrier(CLK_GLOBAL_MEM_FENCE);

    	id += stride;
    	od += stride;
    	cur = inimage[od];
    	nxt = inimage[id];

    	barrier(CLK_GLOBAL_MEM_FENCE);

    }
}
