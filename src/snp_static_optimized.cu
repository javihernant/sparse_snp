#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <snp_static.hpp> 

using namespace std;


/** Allocation */
SNP_static_optimized::SNP_static_optimized(uint n, uint m, int mode) : SNP_model(n,m, mode)
{
    //Allocate cpu variables
    this -> spiking_vector = (int*) malloc(sizeof(int)*n);
    //if no rule selected for the current computation, spiking_vector[i]=-1 for neuron i
    memset(this->spiking_vector,-1,  sizeof(int)*n); 
    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*n);
    memset(this->trans_matrix,-1,sizeof(int)*n*n);
    this->z_vector    = (int*) malloc(sizeof(int)*n);
    memset(this->z_vector,0,sizeof(int)*n);

    //Allocate device variables
    cudaMalloc((&this->d_spiking_vector),  sizeof(int)*m);
    cudaMemcpy(d_spiking_vector,spiking_vector, sizeof(int)*n, cudaMemcpyHostToDevice);
    //d_trans_matrix allocated when z is known

}

/** Free mem */
SNP_static_optimized::~SNP_static_optimized()
{
    free(this->trans_matrix);
    // cudaFree(this->d_trans_matrix);
    cudaFree(this->d_trans_matrix);
}

void SNP_static_optimized::printTransMX(){
    for (int i=0; i<z; i++){
		
		for (int j=0; j<n; j++){
            int idx = (i*n + j);
			std::cout << trans_matrix[idx] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void SNP_static_optimized::include_synapse(uint i, uint j)
{
    trans_matrix[z_vector[i]*n+i] = j;
    z_vector[i]++;
    
}


void SNP_static_optimized::load_transition_matrix () 
{
    int z_aux;
    for(int i=0; i<n; i++){
        z_aux = z_vector[i];
        if(z_aux>z){
            z = z_aux;    
        }
    }

    this-> trans_matrix = (int *) realloc(this->trans_matrix,sizeof(int)*n*z);
    cudaMalloc((&this->d_trans_matrix),  sizeof(int)*n*z);
    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(int)*n*z,  cudaMemcpyHostToDevice);
     

    // TODO check if we need to set matrices for spiking and configuration vectors
}

__global__ void kalc_spiking_vector_optimized(int* spiking_vector, int* conf_vector, int* delays_vector, int* rule_index,int* rc,uint* rd,uint* rnid, int* rei, int* ren, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n && delays_vector[nid]==0) {
        //vector<int> active_rule_idxs_ni;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar i = rei[r];
            uchar n = ren[r];
            int x = conf_vector[nid];
            if (((int) (i&(x==n)) || ((1-i)&(x>=n)))){
                //active_ridx.push_back(r);
                delays_vector[nid] = rd[r];
                conf_vector[nid]-=rc[r];
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
    
    kalc_spiking_vector_optimized<<<gs,bs>>>(d_spiking_vector, d_conf_vector, d_delays_vector, d_rule_index, d_rules.c, d_rules.d, d_rules.nid, d_rules.Ei, d_rules.En, n);
    cudaDeviceSynchronize();
    cudaMemcpy(spiking_vector, d_spiking_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(delays_vector, d_delays_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);


}


__global__ void kalc_transition_optimized(int* spiking_vector, int* trans_matrix, int* conf_vector, int* delays_vector, int* rc, int* rp, int z, int n){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;

    if(nid<n && delays_vector[nid]==0){
        int rid = spiking_vector[nid];
        spiking_vector[nid]= -1;
        int c = rc[rid];
        int p = rp[rid];

        // printf("nid:%d, rid:%d, c:%d, p:%d\n", nid, rid, c, p);

        for(int j=0; j<z; j++){
            
            int n_j = trans_matrix[j*n+nid];

            if(delays_vector[n_j]>0) break;

            if(n_j >= 0){
                atomicAdd((int *) &conf_vector[n_j], p);


            }else{
                //if padded value (-1)
                break;
            }

        }
    }

    if(nid<n && delays_vector[nid]>0){
        delays_vector[nid]--;
    }
    

}

void SNP_static_optimized::calc_transition()
{
    kalc_transition_optimized<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.c, d_rules.p, z,n);
    cudaDeviceSynchronize();


    
    

}

