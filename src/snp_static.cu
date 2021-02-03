#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#define assert
//#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define IDXCB(i,j,ld) (((j)*(ld))+(i))   // indexing for CUBLAS

#include <snp_static.hpp>

using namespace std;

/** Allocation */
SNP_static_cublas::SNP_static_cublas(uint n, uint m) : SNP_model(n,m)
{
    // n is num of rows, m is num of colums. 
    cudaError_t cudaStat;
    cublasStatus_t stat;
    // done by subclasses
    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    memset(this->trans_matrix,0,sizeof(short)*n*m);
    /*for (int i = 0; i < m; i++) // for each row = rule
        for (int j = 0; j<n; j++) // for each column = neuron
            this->trans_matrix[i*n+j] = 0;*/

    cudaStat  = cudaMalloc(&this->d_trans_matrix,  sizeof(short)*n*m);
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        exit(1);
        // return EXIT_FAILURE;
    }

    this->cublas_handle = (cublasHandle_t *) malloc(sizeof(cublasHandle_t));
    stat = cublasCreate((cublasHandle_t *)cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        //return EXIT_FAILURE;
        exit(1);
    }
}

/** Free mem */
SNP_static_cublas::~SNP_static_cublas()
{
    free(this->trans_matrix);
    free(this->cublas_handle);
    cudaFree(this->d_trans_matrix);
}

void SNP_static_cublas::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        //trans_matrix[r*n+i] = rules.c[r];
        //trans_matrix[r*n+j] = rules.p[r];
        trans_matrix[IDXCB(r,i,n)] = rules.c[r];
        trans_matrix[IDXCB(r,i,n)] = rules.p[r];
    }
}


void SNP_static_cublas::load_transition_matrix () 
{
    //handled by sublcasses
    //cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(short)*n*m,  cudaMemcpyHostToDevice); // now handled by CuBLAS
    cublasStatus_t stat;
    stat = cublasSetMatrix (n, m, sizeof(*this->trans_matrix), this->trans_matrix, n, this->d_trans_matrix, n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("loading transition matrix to GPU failed");
        exit(1);
        // return EXIT_FAILURE;
    }

    // TODO The following shoud be done in another function, but for simplicity I put it here
    // TODO check if we need to set matrices for spiking and configuration vectors
}


/*__global__ void ksmvv (short* a, short* v, short* w, uint m) i
{
    uint n = blockIdx.x;
    uint acum = =0;
    for (uint i=tid; i<m; i+=blockDim.x) {
        acum+=a[i]*v[i];
    }
    __syncthreads();

    // reduce

    if (threadIdx.x==0)
        w[n] = acum;
}*/

void SNP_static_cublas::calc_transition()
{
    cublasStatus_t stat;
    float alpha =1.0f;
    float beta =0.0f;
    stat = cublasSgemv((cublasHandle_t)*this->cublas_handle,CUBLAS_OP_N,n,m,&alpha,this->d_trans_matrix,n,this->d_spiking_vector,1,&beta,this->d_conf_vector,1);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("matrix-vector multiplication on GPU failed");
        return EXIT_FAILURE;
    }
}

