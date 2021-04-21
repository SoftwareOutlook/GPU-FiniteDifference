#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <omp.h>
#include <fstream>

#include "fdiff.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define RAND_MAX 5
#define RAND_MIN -5
#define PRINT_TIMINGS 1
#define ORDER 1
#define SIZEX 16
#define SIZEY 16

int main(int argc, char** argv) {

  // Variable initialisation
  int sizeX = 64;             // Size of input image (X)
  int sizeY = 64;                 // Size of input image (Y)
  int sizeZ = 64;                // Size of input image (Z)
  int dim   = 3;                // The number of dimensions of the input image
  int bc    = 0;                // Boundary condition selection {0, 1, 2}
  int T     = 4;                // Number of OMP threads

  // The total volume of the input image
  int volume = sizeX * sizeY * sizeZ;
  int bytes = volume * sizeof(float);

  // Array allocation for host image and X,Y,Z difference output
  float *inimage = (float *)malloc(volume * sizeof(float));
  float *outX = (float *)malloc(volume * sizeof(float));
  float *outY = (float *)malloc(volume * sizeof(float));
  float *outZ = (float *)malloc(volume * sizeof(float));

  // Device buffers
  cl::Buffer d_x, d_y, d_z, d_in;

  // Allocate arrays that will copied back to the device
  float *h_x = (float *)malloc(volume * sizeof(float));
  float *h_y = (float *)malloc(volume * sizeof(float));
  float *h_z = (float *)malloc(volume * sizeof(float));

  // Random initialisation of the input array
  for(int i=0; i < volume; i++) {
    inimage[i] = rand() % (RAND_MAX + 1 - RAND_MIN) + RAND_MIN;
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

  printf("CPU:\n", end - start);

  cl_int err = CL_SUCCESS;

  // OpenCL initialisation
  try {

    // Query platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.size() == 0) {
      std::cout << "Platform size 0\n";
      return -1;
    }

    // Get first available GPU device which supports double precision.
    cl::Context context;
    std::vector<cl::Device> devices;
    for (auto p = platforms.begin(); devices.empty() && p != platforms.end(); p++)
    {
      std::vector<cl::Device> pldev;

      try
      {
	p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

	for (auto d = pldev.begin(); devices.empty() && d != pldev.end(); d++)
	{
	  if (!d->getInfo<CL_DEVICE_AVAILABLE>())
	    continue;

	  std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

	  if (ext.find("cl_khr_fp64") == std::string::npos &&
	      ext.find("cl_amd_fp64") == std::string::npos)
	    continue;

	  devices.push_back(*d);
	  context = cl::Context(devices);
	}
      }

      catch (...)
      {
	devices.clear();
      }
    }

    if (devices.empty())
    {
      std::cerr << "No GPU with double precision not found." << std::endl;
      return 1;
    }

    std::cout << "Using device " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;


    // Quet GPUs max work item
    auto dimensions = devices[0].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    std::cout << "Max dimensions: " << dimensions[0] << "x" << dimensions[1]
	      << "x" << dimensions[2] <<  std::endl;

    size_t global_dim = dimensions[0] * dimensions[1] * dimensions[2];

    // Create command queue
    cl::CommandQueue queue(context, devices[0], 0, &err);

    // Create device memory buffers
    d_in = cl::Buffer(context, CL_MEM_WRITE_ONLY , bytes);
    d_x  = cl::Buffer(context, CL_MEM_READ_WRITE , bytes);
    d_y  = cl::Buffer(context, CL_MEM_READ_WRITE, bytes);
    d_z  = cl::Buffer(context, CL_MEM_READ_WRITE, bytes);

    // Bind memory buffers - map data from host to device
    queue.enqueueWriteBuffer(d_in, CL_TRUE, 0, bytes, inimage);

    //Build kernel from source string
    std::ifstream source_file("fdiff.cl");
    std::string source_code(std::istreambuf_iterator<char>(source_file),
			   (std::istreambuf_iterator<char>()));

    cl::Program program_(context, cl::Program::Sources(1,
			 std::make_pair(source_code.c_str(),
			 source_code.length())));

    program_.build(devices);

    // Create kernel object
    cl::Kernel fd_x, fd_xy, fd_xyz;

    if(bc == 0) {
      fd_x   = cl::Kernel(program_, "fd_x_nm", &err);
      fd_xy  = cl::Kernel(program_, "fd_xy_nm", &err);
      fd_xyz = cl::Kernel(program_, "fd_xyz_nm", &err);
    }

    if(bc == 1) {
      fd_x   = cl::Kernel(program_, "fd_x_pd", &err);
      fd_xy  = cl::Kernel(program_, "fd_xy_pd", &err);
      fd_xyz = cl::Kernel(program_, "fd_xyz_pd", &err);
    }

    if(bc == 2) {
      fd_x   = cl::Kernel(program_, "fd_x_pd", &err);
      fd_xy  = cl::Kernel(program_, "fd_xy_pd", &err);
      fd_xyz = cl::Kernel(program_, "fd_xyz_pd", &err);
    }


    // Bind kernel arguements to kernel
    fd_x.setArg(0, d_x);
    fd_x.setArg(1, d_y);
    fd_x.setArg(2, d_z);
    fd_x.setArg(3, d_in);
    fd_x.setArg(4, sizeX);
    fd_x.setArg(5, sizeY);
    fd_x.setArg(6, sizeZ);
    fd_x.setArg(7, volume);

    fd_xy.setArg(0, d_x);
    fd_xy.setArg(1, d_y);
    fd_xy.setArg(2, d_z);
    fd_xy.setArg(3, d_in);
    fd_xy.setArg(4, sizeX);
    fd_xy.setArg(5, sizeY);
    fd_xy.setArg(6, sizeZ);
    fd_xy.setArg(7, volume);

    fd_xyz.setArg(0, d_x);
    fd_xyz.setArg(1, d_y);
    fd_xyz.setArg(2, d_z);
    fd_xyz.setArg(3, d_in);
    fd_xyz.setArg(4, sizeX);
    fd_xyz.setArg(5, sizeY);
    fd_xyz.setArg(6, sizeZ);
    fd_xyz.setArg(7, volume);

    size_t localSize1d[]  = {SIZEX};
    size_t globalSize1d[] = {sizeX};

    size_t localSize2d[]  = {SIZEX, SIZEY};
    size_t globalSize2d[] = {sizeX, sizeY};

    // // Enqueue kernel
    cl_event event;

    double st = omp_get_wtime();

    if(dim == 1) {

      clEnqueueNDRangeKernel(queue(),
      			     fd_x(),
      			     1,
      			     0,
      			     globalSize1d,
      			     localSize1d,
      			     0, 0, &event);

      clWaitForEvents(1, &event);
    }


    if(dim == 2) {

      clEnqueueNDRangeKernel(queue(),
    			     fd_xy(),
    			     2,
    			     0,
    			     globalSize2d,
    			     localSize2d,
    			     0, 0, &event);

      clWaitForEvents(1, &event);
    }

    if(dim == 3) {

      clEnqueueNDRangeKernel(queue(),
    			     fd_xyz(),
    			     2,
    			     0,
    			     globalSize2d,
    			     localSize2d,
    			     0, 0, &event);

      clWaitForEvents(1, &event);

    }

    clWaitForEvents(0, &event);

    double ed = omp_get_wtime();

    if(PRINT_TIMINGS == 1) printf("GPU Time taken: %lf\n", ed-st);

    queue.enqueueReadBuffer(d_x, CL_TRUE, 0, bytes, h_x);
    queue.enqueueReadBuffer(d_y, CL_TRUE, 0, bytes, h_y);
    queue.enqueueReadBuffer(d_z, CL_TRUE, 0, bytes, h_z);

  } catch(cl::Error err) {
    std::cerr << "ERROR : " << err.what() << "(" << err.err() << ")" << std::endl;
  }

  if(PRINT_TIMINGS == 1) printf("CPU Run:\t%lf\n", end - start);

  return 0;

}
