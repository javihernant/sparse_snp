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
    //Allocate cpu variables
    this -> spiking_vector = (ushort*) malloc(sizeof(ushort)*m);
    memset(this->spiking_vector,0,  sizeof(ushort)*m);

    this->trans_matrix    = (short*)  malloc(sizeof(short)*n*m);
    memset(this->trans_matrix,0,sizeof(short)*n*m);

    //Allocate device variables
    cudaMalloc((&this->d_spiking_vector),  sizeof(ushort)*m);
    cudaMalloc((&this->d_trans_matrix),  sizeof(short)*n*m);
    
}

/** Free mem */
SNP_static::~SNP_static()
{
    free(this->spiking_vector);
    cudaFree(this->d_spiking_vector);

    free(this->trans_matrix);
    cudaFree(this->d_trans_matrix);
}

void SNP_static::printTransMX()
{
    for (int i=0; i<m; i++){
		
		for (int j=0; j<n; j++){
			std::cout << trans_matrix[i*n + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void SNP_static::include_synapse(uint i, uint j)
{
    //store by columns for better VxM performance
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        trans_matrix[i*m+r] = 0;  
        trans_matrix[j*m+r] = rules.p[r];
    }
}


void SNP_static::load_transition_matrix () 
{

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
__global__ void kalc_spiking_vector(ushort* spiking_vector, int* delays_vector, ushort* rd, int* conf_vector, int* rule_index,short* rc, short* rei, short* ren, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;

    if (nid<n && delays_vector[nid]==0) {

        
        for (uint r=rule_index[nid]; r<rule_index[nid+1]; r++){

            uchar e_i = rei[r];
            uchar e_n = ren[r];
            int x = conf_vector[nid];

            if ((int) (e_i&(x==e_n)) || ((1-e_i)&(x>=e_n))) {
                
                spiking_vector[r] = 1;
                conf_vector[nid]-=rc[r];
                delays_vector[nid] = rd[r];

                break;
            }

            

        }

        
    }
}

void SNP_static::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (n+255)/256;
    kalc_spiking_vector<<<gs,bs>>>(d_spiking_vector, d_delays_vector, d_rules.d, d_conf_vector, d_rule_index,d_rules.c, d_rules.Ei, d_rules.En, n);
    cudaDeviceSynchronize();

    //send spiking_vector and delays_vector to host in order to decide if stop criterion has been reached
    cudaMemcpy(spiking_vector, d_spiking_vector,  sizeof(ushort)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(delays_vector, d_delays_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);


}

__global__ void kalc_transition(ushort* spiking_vector, short* trans_matrix, int* conf_vector,int * delays_vector, uint * rnid , int n, int m){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    //nid<n
    if (nid<n && delays_vector[nid]==0){
        for (int r=0; r<m; r++){
            //only sum spikes from neurons that are open, even though spiking_vector[r]=1. TODO: In cublas make trans_matrix_copy and make 0 every row of every rule corresponding to a closed neuron.
            if(delays_vector[rnid[r]] == 0){
                conf_vector[nid] += spiking_vector[r] * trans_matrix[nid*m+r]; 
                spiking_vector[r] = 0; //disable rule that has been used
            }
            
        }



        // printf("%d ",conf_vector[nid]);
    }

    if(nid<n && delays_vector[nid]>0){
        delays_vector[nid]--;
    }

}

void SNP_static::calc_transition()
{
    kalc_transition<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.nid,n,m);
    cudaDeviceSynchronize();

}

