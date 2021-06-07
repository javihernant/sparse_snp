#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <snp_static.hpp> 

using namespace std;


/** Allocation */
SNP_static_optimized::SNP_static_optimized(uint n, uint m) : SNP_model(n,m)
{
    //Allocate cpu variables
    this -> spiking_vector = (ushort*) malloc(sizeof(ushort)*n);
    memset(this->spiking_vector,0,  sizeof(ushort)*n);
    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    memset(this->trans_matrix,0,sizeof(short)*n*m);

    //Allocate device variables
    cudaMalloc((&this->d_spiking_vector),  sizeof(ushort)*m);
    cudaMalloc((&this->d_trans_matrix),  sizeof(short)*n*m);

}

/** Free mem */
SNP_static_optimized::~SNP_static_optimized()
{
    free(this->trans_matrix);
    // cudaFree(this->d_trans_matrix);
    free(this->comp_trans_matrix);
    cudaFree(this->d_comp_trans_matrix);
}

void SNP_static_optimized::include_synapse(uint i, uint j)
{
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[r*n+i] = -rules.c[r];
        if (j<n) trans_matrix[r*n+j] = rules.p[r];
    }
}

void SNP_static_optimized::init_compressed_matrix(){
    //find z
    calc_z();

    this->comp_trans_matrix    = (short*)  malloc(sizeof(short)*n*z);
    memset(this->comp_trans_matrix,-1,sizeof(short)*n*z);

    //fill ell matrix
    for (int nidx=0; nidx<n; nidx++){
        //take one of the neuron's rule idx and use it to access trans_matrix to view the synapses.
        int i = rule_index[nidx]; 
        int i_aux=0;
        for(int j=0; j<n; j++){
            if(trans_matrix[i*n+j]>0){
                comp_trans_matrix[i_aux*n + nidx] = j;
                i_aux++;
            }
        }

    }

    for (int i=0; i<z; i++){
		
		for (int j=0; j<n; j++){
            int idx = (i*n + j);
			std::cout << "(" << comp_trans_matrix[idx] << ") ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";

}

void SNP_static_optimized::load_transition_matrix () 
{
    //handled by sublcasses
    init_compressed_matrix();

    cudaMalloc((&this->d_comp_trans_matrix),  sizeof(short)*n*z);
    cudaMemcpy(d_comp_trans_matrix,  comp_trans_matrix,   sizeof(short)*n*z,  cudaMemcpyHostToDevice); 

    // TODO The following should be done in another function, but for simplicity I put it here
    // TODO check if we need to set matrices for spiking and configuration vectors
}

__global__ void kalc_spiking_vector_optimized(ushort* spiking_vector, int* conf_vector, uint* rule_index, uint* rnid, short* rei, short* ren, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n) {
        //vector<int> active_rule_idxs_ni;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar i = rei[r];
            uchar n = ren[r];
            int x = conf_vector[rnid[r]];
            if ((int) (i&(x==n)) || ((1-i)&(x>=n))){
                //active_ridx.push_back(r);
                spiking_vector[nid] = r;
                break;
            }
        }
        //get_random(active_rule_idxs_ni);
    }
}

void SNP_static_optimized::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (n+255)/256;
    
    kalc_spiking_vector_optimized<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_rule_index, d_rules.nid, d_rules.Ei, d_rules.En, n);
    cudaDeviceSynchronize();


}


__global__ void kalc_transition_optimized(ushort* spiking_vector, short* trans_matrix, int* conf_vector, short* rc, short* rp, int z, int n){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;

    if(nid<n){
        int rid = spiking_vector[nid];
        int c = rc[rid];
        int p = rp[rid];

        printf("nid:%d, rid:%d, c:%d, p:%d\n", nid, rid, c, p);

        atomicSub((int *) &conf_vector[nid], c);

        for(int j=0; j<z; j++){
            
            int n_j = trans_matrix[j*n+nid];
            if(n_j >= 0){
                atomicAdd((int *) &conf_vector[n_j], p);


            }else{
                //if padded value (-1)
                break;
            }

        }
    }
    

}

void SNP_static_optimized::calc_transition()
{
    kalc_transition_optimized<<<n+255,256>>>(d_spiking_vector,d_comp_trans_matrix, d_conf_vector, d_rules.c, d_rules.p, z,n);
    cudaDeviceSynchronize();

    
    

}

