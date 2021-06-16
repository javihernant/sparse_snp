#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <snp_static.hpp> 

using namespace std;


/** Allocation */
SNP_static_ell::SNP_static_ell(uint n, uint m, int mode, bool debug) : SNP_model(n,m, mode, debug)
{
    //Allocate cpu variables
    this -> spiking_vector = (int*) malloc(sizeof(int)*m);
    memset(this->spiking_vector,0,  sizeof(int)*m);

    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*m*2);
    memset(this->trans_matrix,-1,sizeof(int)*n*m*2);

    this->z_vector    = (int*) malloc(sizeof(int)*m);
    memset(this->z_vector,0,sizeof(int)*m);

    //Allocate device variables
    cudaMalloc((&this->d_spiking_vector),  sizeof(int)*m);
    cudaMemset(&this->d_spiking_vector, 0, sizeof(int)*m);
    //trans_matrix allocated when z is known

}

/** Free mem */
SNP_static_ell::~SNP_static_ell()
{
    free(this->spiking_vector);
    cudaFree(this->d_spiking_vector);

    free(this->trans_matrix);
    cudaFree(this->d_trans_matrix);

}

void SNP_static_ell::printTransMX(){
    for (int i=0; i<z; i++){
		
		for (int j=0; j<m; j++){
            int idx = (i*m*2 + j*2);
			std::cout << "(" << trans_matrix[idx] << "," << trans_matrix[idx+1] << ") ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

void SNP_static_ell::include_synapse(uint i, uint j)
{
    // for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
    //     trans_matrix[r*n+i] = -rules.c[r];
    //     if (j<n) trans_matrix[r*n+j] = rules.p[r];
    // }

    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        //forgeting rules are not stored in trans_mx. 
        if(rules.p[r]>0){
            trans_matrix[z_vector[r]*m*2+r*2] = j;
            trans_matrix[(z_vector[r]*m*2+r*2)+1] = rules.p[r];
            z_vector[r]++;
        }
        
    }
}

void SNP_static_ell::init_compressed_matrix(){
    
    //find z (max output degree)
    // calc_z();
    // z++;

    // this->comp_trans_matrix    = (short*)  malloc(sizeof(short)*z*m*2);
    // memset(this->comp_trans_matrix,0,sizeof(short)*z*m*2);

    // //fill ell matrix
    // for(int i=0; i<m;i++){
    //     int aux_row=1; //start filling ell from position [1][rule]. row 0 reserved for (negative) c values.
    //     for(int j=0; j<n;j++){
    //         short num = trans_matrix[i*n+j]; //[i][j]
    //         if(num!=0){
    //             if(num>0){
    //                 int idx = (m*aux_row+i)*2;
    //                 comp_trans_matrix[idx] = j;
    //                 comp_trans_matrix[idx+1] = num;
                    
                    
    //                 aux_row++;

    //             }else{
    //                 //first row
    //                 comp_trans_matrix[i*2] = j;
    //                 comp_trans_matrix[i*2+1] = num;
    //                 // std::cout << num << " " << "i:" << i << " j:" <<j << " m:" <<m <<" aux_row:" <<aux_row <<"\n";
    //             }

    //         }
    //     }
        

    // }
    
    //get z (num of rows of the mx) max(z_vector)
    for(int r=0; r<m; r++){
        int aux_z=z_vector[r];
        if(aux_z>z){
            z=aux_z;
        }
    }


    // this -> trans_matrix = (short *) realloc(trans_matrix, z*m*2);

}

void SNP_static_ell::load_transition_matrix () 
{
    //handled by sublcasses
    init_compressed_matrix();

    cudaMalloc((&this->d_trans_matrix),  sizeof(int)*z*m*2);
    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(int)*z*m*2,  cudaMemcpyHostToDevice); 

    // TODO The following should be done in another function, but for simplicity I put it here
    // TODO check if we need to set matrices for spiking and configuration vectors
}

__global__ void kalc_spiking_vector_ell(int* spiking_vector, int* delays_vector, int* conf_vector, int* rule_index, uint* rnid, int* rc, int* rei, int* ren, uint* rd, uint n)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    if (nid<n && delays_vector[nid]==0) {
        //vector<int> active_rule_idxs_ni;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar i = rei[r];
            uchar n = ren[r];
            int x = conf_vector[nid];
            if ((int) (i&(x==n)) || ((1-i)&(x>=n))){
                conf_vector[nid]-= rc[r];
                spiking_vector[r] = 1;
                delays_vector[nid] = rd[r]; 
                break;
            }
        }
        
    }
}

void SNP_static_ell::calc_spiking_vector() 
{
    uint bs = 256;
    uint gs = (n+255)/256;
    
    kalc_spiking_vector_ell<<<gs,bs>>>(d_spiking_vector, d_delays_vector, d_conf_vector, d_rule_index, d_rules.nid, d_rules.c, d_rules.Ei, d_rules.En, d_rules.d, n);
    cudaDeviceSynchronize();

    cudaMemcpy(spiking_vector, d_spiking_vector,  sizeof(int)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(delays_vector, d_delays_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);

}


__global__ void kalc_transition_ell(int* spiking_vector, int* trans_matrix, int* conf_vector, int * delays_vector, uint* rnid, int z, int m){
    int rid = threadIdx.x+blockIdx.x*blockDim.x;
    
    //nid<n
    
    if (rid<m && spiking_vector[rid]>0 && delays_vector[rnid[rid]]==0){
        spiking_vector[rid] = 0;
        for(int i=0; i<z; i++){
            int neuron = trans_matrix[m*2*i+rid*2];
            int value = trans_matrix[m*2*i+rid*2+1];
            if(neuron==-1 && value==-1){
                break;
            }
            if(delays_vector[neuron]==0){
                atomicAdd((int *)&conf_vector[neuron], (int)value);
            }
            
        }
        
        // printf("%d ",conf_vector[nid]);
    }
    



}

__global__ void update_delays_vector(int * delays_vector, int n){
    int nid=threadIdx.x+blockIdx.x*blockDim.x;
    if(nid<n && delays_vector[nid]>0){
        delays_vector[nid]--;
    }
}

void SNP_static_ell::calc_transition()
{
    kalc_transition_ell<<<n+255,256>>>(d_spiking_vector,d_trans_matrix, d_conf_vector, d_delays_vector, d_rules.nid,z,m);
    cudaDeviceSynchronize();
    update_delays_vector<<<n+255,256>>>(d_delays_vector, n);
    cudaDeviceSynchronize();

}

