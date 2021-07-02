#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>


#include <snp_static.hpp> 

using namespace std;

void checkErr2(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s", cudaGetErrorString(err));
    }
}

/** Allocation */
SNP_static_cublas::SNP_static_cublas(uint n, uint m, int mode, bool debug) : SNP_model(n,m, mode, debug)
{
    
    cublasCreate(&(this->handle));
    //Allocate cpu variables
    this -> cublas_spiking_vector = (float*) malloc(sizeof(float)*m);
    memset(this->cublas_spiking_vector,0,  sizeof(float)*m);


    this->cublas_trans_matrix    = (float*)  malloc(sizeof(float)*n*m);
    memset(this->cublas_trans_matrix,0,sizeof(float)*n*m);

    //Allocate device variables
    cudaMalloc((&this->d_cublas_spiking_vector),  sizeof(float)*m);
    // checkErr2(cudaMemset((void *) &this->d_cublas_spiking_vector, 0, sizeof(float)*m));
    thrust::device_ptr<float> dev_ptr(this->d_cublas_spiking_vector);
    thrust::fill(dev_ptr, dev_ptr + n, 0.0f);
    cudaMalloc((&this->d_cublas_spiking_vector_aux),  sizeof(float)*m);
    
    cudaMalloc((&this->d_cublas_trans_matrix),  sizeof(float)*n*m);
    
}

/** Free mem */
SNP_static_cublas::~SNP_static_cublas()
{
    free(this->cublas_spiking_vector);
    cudaFree(this->d_cublas_spiking_vector);
    cudaFree(this->d_cublas_spiking_vector_aux);

    free(this->cublas_trans_matrix);
    cudaFree(this->d_cublas_trans_matrix);

    cublasDestroy(handle);
}

__global__ void printMX(float * cublas_trans_matrix, int m, int n){

    for (int j=0; j<m; j++){
		
		for (int i=0; i<n; i++){
            printf("%.1f ",cublas_trans_matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");

}

void SNP_static_cublas::printTransMX()
{
    printMX<<<1,1>>>(d_cublas_trans_matrix,m,n);    
    cudaDeviceSynchronize();
    
}

void SNP_static_cublas::include_synapse(uint i, uint j)
{
    //store by columns for better VxM performance
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        cublas_trans_matrix[i*m+r] = 0;  
        cublas_trans_matrix[j*m+r] = rules.p[r];
    }
}


void SNP_static_cublas::load_transition_matrix () 
{

    cudaMemcpy(d_cublas_trans_matrix,  cublas_trans_matrix,   sizeof(float)*n*m,  cudaMemcpyHostToDevice); 

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
__global__ void cublas_kalc_spiking_vector(float* spiking_vector, float* spiking_vector_aux, float* trans_matrix, int* delays_vector, uint* rd, float* conf_vector, int* rule_index,int* rc, int* rei, int* ren, uint n, uint m)
{
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    
    if (nid<n) {
        if(nid==0){
            for(int i=0; i<5;i++){
                printf("{%.1f}\n",spiking_vector_aux[i]);
            }
        }
        
        bool rule_set = false;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar e_i = rei[r];
            uchar e_n = ren[r];
            int x = conf_vector[nid];

            
            if (!rule_set && ((int) (e_i&(x==e_n)) || ((1-e_i)&(x>=e_n))) && delays_vector[nid]==0) {
                
                spiking_vector[r] = 1;
                conf_vector[nid]-=rc[r];
                printf("%d spikes are retracted\n", rc[r]);
                for(int i=0; i<3; i++){
                    printf("conf_vector[%d]=%.1f\n", i, conf_vector[i]);
                }
                
                delays_vector[nid] = rd[r];

                rule_set=true;

            }

            
            if(delays_vector[nid]>0){
                //remember trans_matrix is being stored by columns
                
                for(int i=0; i<n; i++){
                    trans_matrix[i*m+r] = 0;
                }
            }
            
            

            if(delays_vector[nid]==0 && spiking_vector[r]==1){
                spiking_vector_aux[r] = 1;
                
            }

            
        }

        if(delays_vector[nid]>0){
            for(int j=0; j<m; j++){
                trans_matrix[nid*m+j] = 0;
            }
            
        }

            


           
    }
    
}

void SNP_static_cublas::calc_spiking_vector() 
{
    cudaError_t error;
    uint bs = 256;
    uint gs = (n+255)/256;
    // checkErr2(cudaMemset((void *) &this->d_cublas_spiking_vector_aux, 0, sizeof(float)*m));
    thrust::device_ptr<float> dev_ptr(this->d_cublas_spiking_vector_aux);
    thrust::fill(dev_ptr, dev_ptr + m, 0.0f);
    cublas_kalc_spiking_vector<<<gs,bs>>>(d_cublas_spiking_vector, d_cublas_spiking_vector_aux, d_cublas_trans_matrix, d_delays_vector, d_rules.d, d_cublas_conf_vector, d_rule_index,d_rules.c, d_rules.Ei, d_rules.En, n, m);
    cudaDeviceSynchronize();

    //send spiking_vector and delays_vector to host in order to decide if stop criterion has been reached
    // checkErr2(cudaMemcpy(cublas_spiking_vector, d_cublas_spiking_vector,  sizeof(float)*m, cudaMemcpyDeviceToHost));
    cudaMemcpy(cublas_spiking_vector, d_cublas_spiking_vector,  sizeof(float)*m, cudaMemcpyDeviceToHost);
    cudaMemcpy(delays_vector, d_delays_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);
    

}

__global__ void update_spiking_and_delays(float* spiking_vector, float* spiking_vector_aux, int * delays_vector, int * rule_index, int n, int m){
    int nid = threadIdx.x+blockIdx.x*blockDim.x;
    //nid<n
    if (nid<n){
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            if(spiking_vector_aux[r]==1){
                spiking_vector[r]=0;
            } 
        }

        if(delays_vector[nid]>0){
            delays_vector[nid]--;
        }



        // printf("%d ",conf_vector[nid]);
    }


}



void SNP_static_cublas::calc_transition()
{
    float al =1.0f;
    float bet =1.0f;
    cublasSgemv(handle,CUBLAS_OP_T,m,n,&al,d_cublas_trans_matrix,m,d_cublas_spiking_vector_aux,1,&bet,d_cublas_conf_vector,1);
    
    update_spiking_and_delays<<<n+255,256>>>(d_cublas_spiking_vector, d_cublas_spiking_vector_aux, d_delays_vector, d_rule_index, n,m);
    cudaDeviceSynchronize();



}

