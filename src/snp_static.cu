#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <snp_static.hpp> 

using namespace std;

/** Allocation */
SNP_static::SNP_static(uint n, uint m) : SNP_model(n,m)
{
    // n is num of rows, m is num of colums. 
    cudaError_t cudaStat;
    // done by subclasses
    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    memset(this->trans_matrix,0,sizeof(short)*n*m);
    /*for (int i = 0; i < m; i++) // for each row = rule
        for (int j = 0; j<n; j++) // for each column = neuron
            this->trans_matrix[i*n+j] = 0;*/

    cudaStat  = cudaMalloc((&this->d_trans_matrix),  sizeof(short)*n*m);
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        exit(1);
        // return EXIT_FAILURE;
    }


}

/** Free mem */
SNP_static::~SNP_static()
{
    free(this->trans_matrix);
    cudaFree(this->d_trans_matrix);
}

void SNP_static::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = -rules.c[r];
        if (j<n) trans_matrix[r*n+j] = rules.p[r];
    }
}


void SNP_static::load_transition_matrix () 
{
    //handled by sublcasses
    cudaError_t error;
    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(short)*n*m,  cudaMemcpyHostToDevice); 

    // TODO The following should be done in another function, but for simplicity I put it here
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


__global__ void kalc_transition(ushort* spiking_vector, uchar* trans_matrix, ushort* conf_vector, int n, int m){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid==0){
        for (int r=0; r<m; r++){
            conf_vector[nid] += spiking_vector[r] * trans_matrix[r*n+nid]; 
            //printf("%d ",spiking_vector[r]);
            printf("%d ",trans_matrix[r*n+nid]);
        }
        printf("%d ",conf_vector[nid]);

    }

}

void SNP_static::calc_transition()
{
    kalc_transition<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector,n,m);
    cudaDeviceSynchronize();
    
    

}

