#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>


#define BLOCKSIZE 16

__device__ float globalArray[256];

//sequential addressing shared memory
__global__ void find_maximum_kernel_seq(float* array, float* max, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while (index + offset < n) {
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	//sequential addressing by reverse loop
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + s]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		*max = fmaxf(*max, cache[0]);
	}
}

//interleaving addressing shared memory
__global__ void find_maximum_kernel_interleaving(float* array, float* max, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	__shared__ float cache[256];


	float temp = -1.0;
	while (index + offset < n) {
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	//interleaving addressing
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + s]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		*max = fmaxf(*max, cache[0]);
	}
}

// Used global memory
__global__ void find_maximum_kernel_global(float* array, float* max, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = gridDim.x * blockDim.x;
	unsigned int offset = 0;

	//__shared__ float cache[256];


	float temp = -1.0;
	while (index + offset < n) {
		temp = fmaxf(temp, array[index + offset]);

		offset += stride;
	}

	globalArray[threadIdx.x] = temp;

	__syncthreads();


	// reduction
	//use Global memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			globalArray[threadIdx.x] = fmaxf(globalArray[threadIdx.x], globalArray[threadIdx.x + s]);
		}

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		*max = fmaxf(*max, globalArray[0]);
	}
}

int main()
{
	unsigned int N = 10000;
	float* h_array;
	float* d_array;
	float* h_max;
	float* h_max_seq;
	float* d_max;
	float* d_max_seq;
	float* d_max_g;
	float* h_max_g;


	// allocate memory
	h_array = (float*)malloc(N * sizeof(float));
	h_max = (float*)malloc(sizeof(float));
	h_max_seq = (float*)malloc(sizeof(float));
	h_max_g = (float*)malloc(sizeof(float));
	cudaMalloc((void**)&d_array, N * sizeof(float));
	cudaMalloc((void**)&d_max, sizeof(float));
	cudaMalloc((void**)&d_max_seq, sizeof(float));
	cudaMalloc((void**)&d_max_g, sizeof(float));
	cudaMemset(d_max, 0, sizeof(float));
	cudaMemset(d_max_seq, 0, sizeof(float));
	cudaMemset(d_max_g, 0, sizeof(float));


	// fill host array with data
	for (unsigned int i = 0; i < N; i++) {
		h_array[i] = float(rand() % 1000);
		//printf("%f \n", h_array[i]);
	}

	//========================= Using interleaving Addressing Shared Memory ==================================//

	// set up timing variables
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


	// copy from host to device
	cudaEventRecord(gpu_start, 0);
	cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);


	// call kernel
	for (unsigned int j = 0; j < N; j++) {
		dim3 gridSize = 256;
		dim3 blockSize = 256;
		find_maximum_kernel_interleaving<< < gridSize, blockSize >> > (d_array, d_max, N);
		cudaThreadSynchronize();
	}


	// copy from device to host
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	//report results
	std::cout << "Maximum number found on gpu (interleaving addressing) is: " << *h_max << std::endl;
	std::cout << "The gpu took: " << gpu_elapsed_time << " milli-seconds" << std::endl;

	//========================= Using Sequential Addressing Shared Memory ==================================//

	//cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);
	// call kernel
	for (unsigned int j = 0; j < N; j++) {
		dim3 gridSize = 256;
		dim3 blockSize = 256;
		find_maximum_kernel_seq << < gridSize, blockSize >> > (d_array, d_max_seq, N);
		cudaThreadSynchronize();
	}

	cudaMemcpy(h_max_seq, d_max_seq, sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	//report results
	std::cout << "Maximum number found on gpu (sequential addressing) was: " << *h_max_seq << std::endl;
	std::cout << "The gpu took: " << gpu_elapsed_time << " milli-seconds" << std::endl;

	//========================= Using Global Memory ==================================//

	//cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);
	// call kernel
	for (unsigned int j = 0; j < N; j++) {
		dim3 gridSize = 256;
		dim3 blockSize = 256;
		find_maximum_kernel_global << < gridSize, blockSize >> > (d_array, d_max_g, N);
		cudaThreadSynchronize();
	}

	cudaMemcpy(h_max_g, d_max_g, sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	//report results
	std::cout << "Maximum number found on gpu (global memory) was: " << *h_max_g << std::endl;
	std::cout << "The gpu took: " << gpu_elapsed_time << " milli-seconds" << std::endl;



	// run cpu version
	clock_t cpu_start = clock();
	for (unsigned int j = 0; j < 1000; j++) {
		*h_max = -1.0;
		for (unsigned int i = 0; i < N; i++) {
			if (h_array[i] > * h_max) {
				*h_max = h_array[i];
			}
		}
	}
	clock_t cpu_stop = clock();
	clock_t cpu_elapsed_time = 1000 * (cpu_stop - cpu_start) / CLOCKS_PER_SEC;

	std::cout << "Maximum number found on cpu was: " << *h_max << std::endl;
	std::cout << "The cpu took: " << cpu_elapsed_time << " milli-seconds" << std::endl;



	// free memory
	free(h_array);
	free(h_max);
	free(h_max_seq);
	free(h_max_g);
	cudaFree(d_array);
	cudaFree(d_max);
	cudaFree(d_max_seq);
	cudaFree(d_max_g);
	cudaFree(globalArray);
}


