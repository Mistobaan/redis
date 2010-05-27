#include <cudpp.h>
#include <stdio.h>
#include "gpusort.h"

#define check_cuda_error() {\
	if (cudaError_t e = cudaGetLastError()) { \
		fprintf(stderr, "%s:%i: %s\n", __FILE__, __LINE__, \
				cudaGetErrorString(e)); \
		exit(-1); \
	} }

void runSortingKernel(float *keys, int *values, int n) {
	float *d_keys = 0;
	int *d_values = 0;
	CUDPPConfiguration config;
	size_t keys_bytes = n * sizeof(*d_keys),
		   values_bytes = n * sizeof(*d_values);
	config.algorithm = CUDPP_SORT_RADIX;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	config.datatype = CUDPP_FLOAT;
	CUDPPHandle planhandle = 0;
	CUDPPResult result = cudppPlan(&planhandle, config, n, 1, 0);
	if (CUDPP_SUCCESS != result) {
		fprintf(stderr, "Error creating CUDPPPlan\n");
		exit(-1);
	}
	cudaMalloc((void**) &d_keys, keys_bytes);
	check_cuda_error();
	cudaMalloc((void**) &d_values, values_bytes);
	check_cuda_error();
	cudaMemcpy(d_keys, keys, keys_bytes, cudaMemcpyHostToDevice);
	check_cuda_error();
	cudaMemcpy(d_values, values, values_bytes, cudaMemcpyHostToDevice);
	check_cuda_error();
	cudppSort(planhandle, d_keys, d_values, sizeof(*d_keys) * 8, n);
	check_cuda_error();
	cudppDestroyPlan(planhandle);
	check_cuda_error();
	cudaMemcpy(values, d_values, values_bytes, cudaMemcpyDeviceToHost);
	check_cuda_error();
	cudaThreadSynchronize();
	check_cuda_error();
	cudaFree(d_keys);
	cudaFree(d_values);
}
