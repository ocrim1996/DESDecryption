#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <numeric>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

// From https://stackoverflow.com/a/47661335
#define IS_BIG_ENDIAN() (*(uint16_t *)"\0\xff" < 0x100)

// Adapted from https://stackoverflow.com/a/21632224
__device__ __host__ uint64_t str2uint64(const char *input) {
	uint64_t result = 0;
	if (IS_BIG_ENDIAN()) {
		for(int i=0; i<8; i++) {
			result |= (uint64_t)input[i];
			if (i < 7) { result <<= 8; }
		}
	} else {
		for(int i=7; i>=0; --i) {
			result <<= 8;
			result |= (uint64_t)input[i];
		}
	}
	return result;
}
