#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#define assert
#include <cuda.h>

#include <snp_static.hpp>

using namespace std;

/** Allocation */
SNP_static_cublas::SNP_static_cublas(uint n, uint m) : SNP_model(n,m)
{
    // done by subclasses
    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    cudaMalloc(&this->d_trans_matrix,  sizeof(short)*n*m);
    for (int i = 0; i < m; i++) // for each row = rule
        for (int j = 0; j<n; j++) // for each column = neuron
            this->trans_matrix[i*n+j] = 0;
}

/** Free mem */
SNP_static_cublas::~SNP_static_cublas()
{
    
}

void SNP_static_cublas::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = rules.c[r];
        trans_matrix[r*n+j] = rules.p[r];
    }
}


void SNP_static_cublas::load_transition_matrix () 
{
    //handled by sublcasses
    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(short)*n*m,  cudaMemcpyHostToDevice);

}

void SNP_static_cublas::calc_transition()
{

}

