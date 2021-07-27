#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusparse.h> 
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>


#include <snp_static.hpp> 

using namespace std;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
                                                          \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
                                                        \
    }                                                                          \
}

__global__ void cusparse_gen_CSR_vectors(int * trans_matrix, int nrows, int ncols, int * csrOffsets, int * csrColumns, float* csrValues){
    //remember at this point trans_matrix is transposed. (neurons x rules)
    //only one thread working. Modify if more are used.
    int i_nz = 0; 
    csrOffsets[0]=0;
    for(int i=0; i<nrows; i++){
        for(int j=0; j<ncols; j++){
            if(trans_matrix[i*ncols + j]!=0){
                csrColumns[i_nz] = j;
                // printf("csrColumn[%d]=%d\n",i_nz,j);
                csrValues[i_nz] = trans_matrix[i*ncols + j];
                // printf("csrValues[%d]=%d\n",i_nz,trans_matrix[i*ncols + j]);
                i_nz++;
            }
        }
        csrOffsets[i+1]=i_nz;
        // printf("csrOffset[%d]=%d\n",i+1,i_nz);
    }

    // printf("csrValues:");
    // for(int i=0; i<i_nz; i++){
    //     printf("%.1f ",csrValues[i]);

    // }
    // printf("\n");

    // printf("csrColumns:");
    // for(int i=0; i<i_nz; i++){
    //     printf("%d ",csrColumns[i]);

    // }
    // printf("\n");
    // printf("csrRows:");
    // for(int i=0; i<nrows+1; i++){
    //     printf("%d ",csrOffsets[i]);

    // }
    // printf("\n");
}

/** Allocation */
SNP_static_cusparse::SNP_static_cusparse(uint n, uint m, int mode, int verbosity) : SNP_model(n,m, mode, verbosity){

    CHECK_CUSPARSE(cusparseCreate(&(this->cusparse_handle)));

    CHECK_CUSPARSE(cusparseSetStream(this->cusparse_handle, this->stream2));


    //Allocate cpu variables
    this -> cublas_spiking_vector = (float*) malloc(sizeof(float)*m);
    memset(this->cublas_spiking_vector,0,  sizeof(float)*m);

    this-> spiking_vector_aux = (float *) malloc(sizeof(float)*m);

    this->trans_matrix    = (int*)  malloc(sizeof(int)*n*m);
    memset(this->trans_matrix,0,sizeof(float)*n*m);

    

    //Allocate device variables
    cudaMalloc((&this->d_cublas_spiking_vector),  sizeof(float)*m);
    thrust::device_ptr<float> dev_ptr(this->d_cublas_spiking_vector);
    thrust::fill(dev_ptr, dev_ptr + n, 0.0f);
    
    CHECK_CUDA(cudaMalloc((&this->d_cublas_spiking_vector_aux),  sizeof(float)*m));
    
    cudaMalloc((&this->d_trans_matrix),  sizeof(int)*n*m);

    cudaMallocHost(&this->nnz,sizeof(int));
    memset(this->nnz,0,  sizeof(int));

    cudaMalloc((&this->d_nnz),  sizeof(int));
    cudaMemset((void * ) this->d_nnz, 0, sizeof(int));  

}

/** Free mem */
SNP_static_cusparse::~SNP_static_cusparse()
{
    free(this->cublas_spiking_vector);
    cudaFree(this->d_cublas_spiking_vector);
    cudaFree(this->d_cublas_spiking_vector_aux);

    free(this->trans_matrix);
    cudaFree(this->d_trans_matrix);

    free(this->nnz);
    cudaFree(this->d_nnz);

    cudaFree(this->d_csrColumns);
    cudaFree(this->d_csrValues);
    cudaFree(this->d_csrOffsets);
    
    CHECK_CUSPARSE( cusparseDestroyDnVec(cusparse_confv) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(cusparse_spkv) );
    if(!this->delays_active){
        CHECK_CUSPARSE( cusparseDestroySpMat(cusparse_trans_mx) );
    }
    CHECK_CUSPARSE( cusparseDestroy(this -> cusparse_handle) );
}

__global__ void printMX(int * trans_matrix, int m, int n){

    for (int j=0; j<m; j++){
		
		for (int i=0; i<n; i++){
            printf("%d ",trans_matrix[i*m + j]);
		}
		printf("\n");
	}
	printf("\n");

}

void SNP_static_cusparse::printTransMX()
{
    printMX<<<1,1>>>(d_trans_matrix,m,n);    
    cudaDeviceSynchronize();
    
}

void SNP_static_cusparse::include_synapse(uint i, uint j)
{
    //store by columns for better VxM performance
    for (int r = rule_index[i]; r < rule_index[i+1]; r++) {
        if(!delays_active && rules.d[r]>0){
            delays_active = true;
        }
        
        trans_matrix[i*m+r] = 0;  
        trans_matrix[j*m+r] = rules.p[r];
        
    }
}


void SNP_static_cusparse::load_transition_matrix () 
{   

    cudaMemcpy(d_trans_matrix,  trans_matrix,   sizeof(int)*n*m,  cudaMemcpyHostToDevice); 

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
__global__ void cusparse_kalc_spiking_vector(float* spiking_vector, float* spiking_vector_aux, int* trans_matrix, int* delays_vector, uint* rd, float* conf_vector, int* rule_index,int* rc, int* rei, int* ren, uint n, uint m)
{
    //TODO: Comprobar q funciona para cusparse
    uint nid = threadIdx.x+blockIdx.x*blockDim.x;
    
    if (nid<n) {
        // if(nid==0){
        //     for(int i=0; i<5;i++){
        //         printf("{%.1f}\n",spiking_vector_aux[i]);
        //     }
        // }
        
        bool rule_set = false;
        for (int r=rule_index[nid]; r<rule_index[nid+1]; r++){
            uchar e_i = rei[r];
            uchar e_n = ren[r];
            int x = conf_vector[nid];

            
            if (!rule_set && ((int) (e_i&(x==e_n)) || ((1-e_i)&(x>=e_n))) && delays_vector[nid]==0) {
                
                spiking_vector[r] = 1;
                conf_vector[nid]-=rc[r];
                // printf("%d spikes are retracted\n", rc[r]);
                // for(int i=0; i<3; i++){
                //     printf("conf_vector[%d]=%d\n", i, conf_vector[i]);
                // }
                
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

void SNP_static_cusparse::calc_spiking_vector() 
{
    
    uint bs = 256;
    uint gs = (n+255)/256;
    // CHECK_CUDA(cudaMemset((void *) &this->d_cublas_spiking_vector_aux, 0, sizeof(int)*m));
    thrust::device_ptr<float> dev_ptr(this->d_cublas_spiking_vector_aux);
    thrust::fill(dev_ptr, dev_ptr + m, 0.0f);
    cusparse_kalc_spiking_vector<<<gs,bs>>>(d_cublas_spiking_vector, d_cublas_spiking_vector_aux, d_trans_matrix, d_delays_vector, d_rules.d, d_cublas_conf_vector, d_rule_index,d_rules.c, d_rules.Ei, d_rules.En, n, m);
    cudaDeviceSynchronize();

    //send spiking_vector and delays_vector to host in order to decide if stop criterion has been reached
    // checkErr2(cudaMemcpy(cublas_spiking_vector, d_cublas_spiking_vector,  sizeof(float)*m, cudaMemcpyDeviceToHost));
    
    
    // cudaMemcpy(cublas_spiking_vector, d_cublas_spiking_vector,  sizeof(float)*m, cudaMemcpyDeviceToHost);
    // cudaMemcpy(spiking_vector_aux, d_cublas_spiking_vector_aux,  sizeof(float)*m, cudaMemcpyDeviceToHost);
    // cudaMemcpy(delays_vector, d_delays_vector,  sizeof(int)*n, cudaMemcpyDeviceToHost);
    

}

__global__ void cusparse_update_spiking_and_delays(float* spiking_vector, float* spiking_vector_aux, int * delays_vector, int * rule_index, int n, int m){
    //d_cublas_spiking_vector, d_cublas_spiking_vector_aux, d_delays_vector, d_rule_index, n, m)
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

__global__ void count_nnz(int * trans_matrix, int nrows, int ncols, int * nnz){
    //counts number of non-zero elements.
    int row = threadIdx.x+blockIdx.x*blockDim.x;
    if(row<nrows){
        int counter = 0;
        for(int j=0; j<ncols; j++){
            if(trans_matrix[row*ncols+j]!=0){
                counter++;
                
            }
        }
        atomicAdd((int *)&nnz[0], counter);

    }
    
}

__global__ void cpy_conf_vector_cusparse(float * conf_v, int *conf_v_cpy, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<n){
        printf("%.1f",conf_v_cpy[idx]);
        conf_v_cpy[idx]= (int) conf_v[idx];
    }
}


void SNP_static_cusparse::calc_transition()
{

    float alpha           = 1.0f;
    float beta            = 1.0f;



    //trans_mx changes only when delays are active. If inactive, no need to build new compressed mx.
    if(this->delays_active || this->step == 0){

        cudaMemset(this->d_nnz,0,sizeof(int)); //use stream2
        count_nnz<<<n+255,256>>>(d_trans_matrix, n, m, d_nnz);
        cudaMemcpy(nnz, d_nnz,  sizeof(int), cudaMemcpyDeviceToHost); 
        
        // printf("nnz addr:%p\n", nnz);
        if(this->step == 0){
            this->nnz0=nnz[0];
            printf("non-zero values: %d\n",nnz0);
            CHECK_CUDA(cudaMalloc(&(this->d_csrColumns),  sizeof(int)*nnz[0]));
            cudaMalloc(&(this->d_csrValues),  sizeof(float)*nnz[0]); 
            cudaMalloc((&this->d_csrOffsets),  sizeof(int)*n+1); 

        }
        
        
        cusparse_gen_CSR_vectors<<<1,1,0,this->stream2>>>(d_trans_matrix, n, m, d_csrOffsets,d_csrColumns, d_csrValues); //TODO: maybe create each of the three vectors in a different thread (much better performace?)

        // Create sparse matrix A in CSR format
        CHECK_CUSPARSE( cusparseCreateCsr(&(this->cusparse_trans_mx), n, m, nnz[0],
            d_csrOffsets, d_csrColumns, d_csrValues,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    }

    if(this->step == 0){
        CHECK_CUSPARSE( cusparseCreateDnVec(&(this->cusparse_confv), n, this->d_cublas_conf_vector, CUDA_R_32F) );
        CHECK_CUSPARSE( cusparseCreateDnVec(&(this->cusparse_spkv), m, this->d_cublas_spiking_vector_aux, CUDA_R_32F) );

        CHECK_CUSPARSE( cusparseSpMV_bufferSize(
            this->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, this->cusparse_trans_mx, this->cusparse_spkv, &beta, this->cusparse_confv, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, &this->bufferSize) );
        CHECK_CUDA( cudaMalloc(&(this->d_buffer),   this->bufferSize) ); 
        this->buffer_created = true;

    }
    
    

    //multiplication
    CHECK_CUSPARSE( cusparseSpMV(this->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, cusparse_trans_mx, cusparse_spkv, &beta, cusparse_confv, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, d_buffer));

    
    
    if(this->delays_active){
        CHECK_CUSPARSE( cusparseDestroySpMat(cusparse_trans_mx) );
    }
    // CHECK_CUSPARSE( cusparseDestroyDnVec(cusparse_spkv) );
    
    
    CHECK_CUDA(cudaGetLastError())
    //updating spikes and delays
    
    cusparse_update_spiking_and_delays<<<n+255,256,0,this->stream2>>>(d_cublas_spiking_vector, d_cublas_spiking_vector_aux, d_delays_vector, d_rule_index, n, m);
    cudaMemset(d_csrValues,0,nnz0); 
    cudaMemset(d_csrColumns,0,nnz0);
    cudaStreamSynchronize(this->stream1);
    cpy_conf_vector_cusparse<<<n+255,256,0,this->stream2>>>(d_cublas_conf_vector, d_conf_vector_cpy, n);
    cudaDeviceSynchronize();
    

}

