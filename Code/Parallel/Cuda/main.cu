#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <inttypes.h>
#include <cmath>
#include "des/des.h"
#include "utils.cuh"

#define WORD_SIZE 8 // word size
#define DICTIONARY "dict_1500000.txt"
#define DICTIONARY_SIZE 1500000

// Load full dictionary into memory
uint64_t *populateDictionary() {
	uint64_t *data = (uint64_t*)malloc(sizeof(uint64_t) * DICTIONARY_SIZE);
	FILE* dictionary;
	if((dictionary = fopen(DICTIONARY, "r")) == NULL) {
		fprintf(stderr, "[?] error: dictionary not found\n");
		exit (EXIT_FAILURE);
	}
	size_t len = 0;
	char* current_word = (char*)malloc(WORD_SIZE * sizeof(char));
	int i = 0;
	while ((getline(&current_word, &len, dictionary)) != -1) {
		//printf("[*] current word: %s", current_word);
		//printf("[*] current word converted to uint64_t: %" PRIu64 "\n", str2uint64(current_word));
		data[i] = str2uint64(current_word);
		i++;
	}

	fclose(dictionary);
	free(current_word);
	return data;
}

__constant__ uint64_t deviceConstantSalt;
__constant__ uint64_t deviceConstantEncryptedPassword;

__global__ void kernel(uint64_t *fullDictionary, uint64_t *found) {
	int wordIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (wordIndex < DICTIONARY_SIZE) {
		uint64_t current_password = fullDictionary[wordIndex];
		uint64_t encrypted = full_des_encode_block(current_password, deviceConstantSalt);
		//printf("[cuda] index: %d, current_password: %" PRIu64 ", salt: %" PRIu64 ", deviceConstantEncryptedPassword: %" PRIu64 "\n", wordIndex, current_password, deviceConstantSalt, deviceConstantEncryptedPassword);
		if (encrypted == deviceConstantEncryptedPassword) {
			*found = current_password;
			return;
		}
	}
}

float execute(int blockSize, const char *password, int nAverage) {

	float gpuTime;
	const char *salt = "PC";

	printf("[*] Target password: %s\n", password);
	printf("[*] Salt: %s\n", salt);

	uint64_t uint_salt = str2uint64(salt);
	uint64_t uint_password = str2uint64(password);

	printf("[*] Target password as uint64_t: ");
	bits_print_grouped(uint_password, 8, 64);

	// encrypt password with DES
	printf("[*] Encrypting password...\n");
	uint64_t encrypted_password = full_des_encode_block(uint_password, uint_salt);

	uint64_t *host_dictionary;
	uint64_t *device_dictionary;

	uint64_t *password_found;
	uint64_t *device_password_found;

	// Allocate host memory
	printf("[*] Allocating host memory...\n");
	host_dictionary = populateDictionary();
	password_found = (uint64_t *)malloc(sizeof(uint64_t));

	// Allocate and set device memory
	printf("[*] Allocating device memory...\n");
	CUDA_CHECK_RETURN(cudaMalloc((void **)&device_dictionary, sizeof(uint64_t)*DICTIONARY_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&device_password_found, sizeof(uint64_t)));
	CUDA_CHECK_RETURN(cudaMemset(device_password_found, 0, sizeof(uint64_t)));

	// Copy from host to device memory
	printf("[*] Copying memory host->device...\n");
	CUDA_CHECK_RETURN(cudaMemcpy(device_dictionary, host_dictionary, sizeof(uint64_t)*DICTIONARY_SIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceConstantSalt, &uint_salt, sizeof(uint64_t)));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(deviceConstantEncryptedPassword, &encrypted_password, sizeof(uint64_t)));

	// Organize grid
	dim3 blockDim(blockSize);
	dim3 gridDim(DICTIONARY_SIZE/blockDim.x + 1);

	printf("[*] Launching kernel(s) and measuring averaged time...\n");

	float gpuTimes[nAverage];
	for (int i=0; i<nAverage; i++) {

		// Start clock
		clock_t start = clock();

		// Execute kernel
		kernel<<<gridDim, blockDim>>>(device_dictionary, device_password_found);

		// Copy results from device memory to host
		CUDA_CHECK_RETURN(cudaMemcpy(password_found, device_password_found, sizeof(uint64_t), cudaMemcpyDeviceToHost));

		// Check if password was decrypted
		if (*password_found) {

			// End clock
			clock_t end = clock();

			//printf("[?] Successfully decrypted password: ");
			//bits_print_grouped(*password_found, 8, 64);

			gpuTime = (float)(end - start)/CLOCKS_PER_SEC;
			//printf("[i] GPU time: %.5f seconds\n", gpuTime);

			gpuTimes[i] = gpuTime;
		} else {
			printf("[*] Password not found in dictionary!\n");
			gpuTimes[i] = 0;
		}
	}

	printf("[*] Cleaning up...\n");

	// Free GPU memory
	CUDA_CHECK_RETURN(cudaFree(device_dictionary));
	CUDA_CHECK_RETURN(cudaFree(device_password_found));

	// Free host memory
	free(host_dictionary);
	free(password_found);

	// Get average
	float avg = 0;
	float sum = 0;
	for (int i=0; i<nAverage; i++) {
		sum += gpuTimes[i];
	}
	avg = (float)sum/nAverage;

	printf("[*] Done! Average GPU time: %.5f", avg);

	return avg;
}

int main(void) {

	// FIRST TEST (test all block sizes on three different passwords)

	int nAverage = 5; // number of runs for each password
	int size = 7;
	float executionTimes[size];
	
	// Test with 8, 16, 32, 64, 128, 256, 512
	int blocks[size];
	for (int i=3, j=0; i < 10; i++, j++) {
		blocks[j] = pow(2,i);
	}

	// Passwords to test
	const char *passwords[3] = {"3kingdom", "giacomix", "6Melissa"};

	// Execute tests for each password
	for (int i=0; i<3; i++) {

		// Iterate over each block size
		for (int j=0; j<size; j++) {
			printf("\n\n===> Executing test for password '%s' with %d blocks (and threads per block) averaged %d times\n\n", passwords[i], blocks[j], nAverage);
			executionTimes[j] = execute(blocks[j], passwords[i], nAverage);
		}

		// Print execution times
		printf("\n\n[*] Times for password '%s': ", passwords[i]);
		for (int z=0; z<size; z++) {
			printf(z==size-1 ? "%f" : "%f, ", executionTimes[z]);
		}
	}

	// SECOND TEST (10 different passwords using 128 blocks)

	/*float secondExecutionTimes[10];
	const char *passwords[10] = {"sirpizza", "3kingdom", "tyleisha", "marumari", "giacomix", "dbcookie2", "Yessssss", "Mypaypa12", "6Melissa", "1Mazzola"};
	for (int i=0; i<10; i++) {
		secondExecutionTimes[i] = execute(128, passwords[i], 5);
		printf("\n\n");
	}
	// Print execution times
	printf("\n\n[*] Times: ");
	for (int z=0; z<10; z++) {
		printf(z==9 ? "%f" : "%f, ", secondExecutionTimes[z]);
	}*/
}
