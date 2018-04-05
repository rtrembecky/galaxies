#define GRIDX 16
#define BLOCK 256

//#include <cuda_profiler_api.h>

// full tiles
__global__ void tiles(int n, float* tmp, sGalaxy A, sGalaxy B, int tbound) {
	const int x = threadIdx.x;
	int i = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float smem[6 * BLOCK];
	float val = 0.0f;
	if (i >= n - tbound) return;

	float ab[6];
	ab[0] = A.x[i];
	ab[1] = A.y[i];
	ab[2] = A.z[i];
	ab[3] = B.x[i];
	ab[4] = B.y[i];
	ab[5] = B.z[i];

	int cx;
	for (cx = i - x + BLOCK; cx <= n - BLOCK; cx += BLOCK) {
		smem[x] = A.x[cx + x]; smem[1 * BLOCK + x] = A.y[cx + x]; smem[2 * BLOCK + x] = A.z[cx + x];
		smem[3 * BLOCK + x] = B.x[cx + x]; smem[4 * BLOCK + x] = B.y[cx + x]; smem[5 * BLOCK + x] = B.z[cx + x];
		__syncthreads();
#pragma nounroll
		for (int j = 0; j < BLOCK; j++) {
			float a[6];
			for (int k = 0; k < 6; k++)
				a[k] = smem[k * BLOCK + j];
			float da = sqrt((ab[0] - a[0])*(ab[0] - a[0])
				+ (ab[1] - a[1])*(ab[1] - a[1])
				+ (ab[2] - a[2])*(ab[2] - a[2]));
			float db = sqrt((ab[3] - a[3])*(ab[3] - a[3])
				+ (ab[4] - a[4])*(ab[4] - a[4])
				+ (ab[5] - a[5])*(ab[5] - a[5]));
			val += (da - db) * (da - db);
		}
		__syncthreads();
	}
	if (cx + x < n) {    // posledny block - len niektore vlakna ulozia smem
		smem[x] = A.x[cx + x]; smem[1 * BLOCK + x] = A.y[cx + x]; smem[2 * BLOCK + x] = A.z[cx + x];
		smem[3 * BLOCK + x] = B.x[cx + x]; smem[4 * BLOCK + x] = B.y[cx + x]; smem[5 * BLOCK + x] = B.z[cx + x];
	}
	__syncthreads();
	for (int j = 0; j < tbound; j++) {
		float a[6];
		for (int k = 0; k < 6; k++)
			a[k] = smem[k * BLOCK + j];
		float da = sqrt((ab[0] - a[0])*(ab[0] - a[0])
			+ (ab[1] - a[1])*(ab[1] - a[1])
			+ (ab[2] - a[2])*(ab[2] - a[2]));
		float db = sqrt((ab[3] - a[3])*(ab[3] - a[3])
			+ (ab[4] - a[4])*(ab[4] - a[4])
			+ (ab[5] - a[5])*(ab[5] - a[5]));
		val += (da - db) * (da - db);
	}
	__syncthreads();
	smem[x] = val;
	__syncthreads();
	// do reduction2 in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (x < s && i + s < n - 1)
		{
			smem[x] += smem[x + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (x == 0) tmp[blockIdx.y*gridDim.x + blockIdx.x] = smem[0];
}
// diagonal tiles
__global__ void diagonal(int n, float* tmp, sGalaxy A, sGalaxy B) {
	const int x = threadIdx.x;
	int i = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float smem[6 * BLOCK];
	float val = 0.0f;
	if (i >= n) return;

	float ab[6];
	ab[0] = A.x[i];
	ab[1] = A.y[i];
	ab[2] = A.z[i];
	ab[3] = B.x[i];
	ab[4] = B.y[i];
	ab[5] = B.z[i];

	for (int k = 0; k < 6; k++)
		smem[k * BLOCK + x] = ab[k];
	if (i == n - 1) return;
	__syncthreads();

	//for (int j = x + 1; j < BLOCK; j++) {
	for (int j = BLOCK - 1; j >= x + 1; j--) {
		if (i - x + j <= n - 1) {
			float a[6];
			for (int k = 0; k < 6; k++)
				a[k] = smem[k * BLOCK + j];
			float da = sqrt((ab[0] - a[0])*(ab[0] - a[0])
				+ (ab[1] - a[1])*(ab[1] - a[1])
				+ (ab[2] - a[2])*(ab[2] - a[2]));
			float db = sqrt((ab[3] - a[3])*(ab[3] - a[3])
				+ (ab[4] - a[4])*(ab[4] - a[4])
				+ (ab[5] - a[5])*(ab[5] - a[5]));
			val += (da - db) * (da - db);
		}
	}

	__syncthreads();
	smem[x] = val;
	__syncthreads();
	// do reduction2 in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (x < s && i + s < n - 1)
		{
			smem[x] += smem[x + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (x == 0) tmp[blockIdx.y*gridDim.x + blockIdx.x] = smem[0];
}

float solveGPU(sGalaxy A, sGalaxy B, int n) {
	//cudaProfilerStart();
	const int gridy = (n + BLOCK*GRIDX - 2) / (BLOCK*GRIDX);
	
	float res = 0.0f;
	const int tbound = n % BLOCK;

	//kernel call and data manipulation
	dim3 grid(GRIDX, gridy);
	dim3 block(BLOCK, 1);

	float* tmp = NULL;
	cudaError_t cudaStatus = cudaMalloc((void**)&tmp, GRIDX * gridy * sizeof(tmp[0]));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! attempted to allocate %zd bytes.\n", GRIDX * gridy * sizeof(tmp[0]));
	}
	float* h_tmp = (float*)malloc(GRIDX * gridy * sizeof(h_tmp[0]));

	float* tmp_diag = NULL;
	cudaStatus = cudaMalloc((void**)&tmp_diag, GRIDX * gridy * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! attempted to allocate %zd bytes.\n", GRIDX * gridy * sizeof(float));
	}
	float* h_tmp_diag = (float*)malloc(GRIDX * gridy * sizeof(float));

	cudaStream_t streams[2];
	for (int i = 0; i < 2; i++)
		cudaStreamCreate(&streams[i]);

	tiles<<<grid, block, 0, streams[0]>>>(n, tmp, A, B, tbound);	
	diagonal<<<grid, block, 0, streams[1]>>>(n, tmp_diag, A, B);
	cudaMemcpy(h_tmp, tmp, GRIDX * gridy * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_tmp_diag, tmp_diag, GRIDX * gridy * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < gridy; i++)
		for (int j = 0; j < GRIDX; j++) {
			if ((i*GRIDX + j) * BLOCK < n) {
				if ((i*GRIDX + j) * BLOCK < n - BLOCK) {
					res += h_tmp[i*GRIDX + j];
				}
				res += h_tmp_diag[i*GRIDX + j];
			}
		}
	for (int i = 0; i < 2; i++)
		cudaStreamDestroy(streams[i]);
	free(h_tmp);
	cudaFree(tmp);
	free(h_tmp_diag);
	cudaFree(tmp_diag);
	
	res = sqrt(1 / ((float)n*((float)n - 1)) * res);
	//cudaProfilerStop();
    return res;
}
