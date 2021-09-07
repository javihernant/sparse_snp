#include <stdio.h>
#include <stdlib.h>
#include <assert.h> //#define assert
#include <cuda.h>


#include <snp_model.hpp>  // "../include/snp_model.hpp" //
// Algorithms
#define CPU      		0
#define GPU_SPARSE		1
#define GPU_ELL 		2
#define GPU_OPTIMIZED	3
#define GPU_CUBLAS 		4
#define GPU_CUSPARSE 	5

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

/** Allocation */

SNP_model::SNP_model(uint n, uint m, int mode, int verbosity, bool write2csv, int repetition)
{
    cudaProfilerStart();
    
    CHECK_CUDA(cudaEventCreate(&this->start));
    CHECK_CUDA(cudaEventCreate(&this->stop));
    CHECK_CUDA(cudaEventRecord(this->start, 0));   
    
    // allocation in CPU
    this->m = m;  // number of rules
    this->n = n;  // number of neurons
    this->ex_mode = mode;
    this->verbosity = verbosity;
    this->write_CSV = write2csv;
    this->step = 0;
    
    if(write_CSV){
        this->csv.open("csv_solutions/conf_vs"+to_string(repetition)+".csv");
    }
    cudaStreamCreate(&this->stream1);
    cudaStreamCreate(&this->stream2);
    //CHECK_CUDA(cudaStreamCreateWithFlags(&this->stream2, cudaStreamNonBlocking)); TODO: usar para cublas y cusparse
    
    cudaMallocHost(&this->conf_vector, sizeof(int)*n);
    memset(this->conf_vector,   0,  sizeof(int)*n);

    if (mode==GPU_CUBLAS || mode==GPU_CUSPARSE){
        cudaMalloc(&this->d_cublas_conf_vector,   sizeof(float)*n);
        this->cublas_conf_vector = (float *) malloc(sizeof(float)*n);
        
        
        // checkErr(cudaMemset((void *) &this->d_cublas_conf_vector,   0,  sizeof(float)*n));
    }else{  
        cudaMalloc(&this->d_conf_vector,   sizeof(int)*n);
        CHECK_CUDA(cudaMemset((void*) this->d_conf_vector,   0,  sizeof(int)*n));
    }
    cudaMalloc(&this->d_conf_vector_cpy, sizeof(int)*n);
    

    this->spiking_vector  = NULL; // spiking vector
    this->delays_vector = (int*) malloc(sizeof(int)*(n));
    this->rule_index      = (int*)   malloc(sizeof(int)*(n+1)); // indices of rules inside neuron (start index per neuron)
    this->rules.Ei        = (int*)  malloc(sizeof(int)*m); // Regular expression Ei of a rule
    this->rules.En        = (int*)  malloc(sizeof(int)*m); // Regular expression En of a rule
    this->rules.c         = (int*)  malloc(sizeof(int)*m); // LHS of rule
    this->rules.p         = (int*)  malloc(sizeof(int)*m); // RHS of rule
    this->rules.d = (uint*) malloc(sizeof(uint)*(m));
    this->rules.nid       = (uint*)   malloc(sizeof(uint)*(m)); // Index of the neuron where the rule is
    if(this->verbosity>=2){
        cudaMallocHost(&this->calc_next_trans, sizeof(bool));
    }else{
        this->calc_next_trans = (bool *) malloc(sizeof(bool));
    }
    

    // initialization (only in CPU, having updated version)
    
    // memset(this->spiking_vector,0,  sizeof(ushort)*m);
    memset(this->delays_vector,0,  sizeof(int)*n);
    memset(this->rule_index,    -1,  sizeof(int)*(n+1));
    rule_index[0]=0;
    memset(this->rules.Ei,      0,  sizeof(int)*m);
    memset(this->rules.En,      0,  sizeof(int)*m);
    memset(this->rules.c,       0,  sizeof(int)*m);
    memset(this->rules.p,       0,  sizeof(int)*m);
    memset(this->rules.d,     0,  sizeof(uint)*(m));
    memset(this->rules.nid,     0,  sizeof(uint)*(m));

    this->d_trans_matrix=NULL;
    this->trans_matrix=NULL;

    // allocation in GPU
    
    // cudaMalloc(&this->d_spiking_vector,sizeof(ushort)*m);
    cudaMalloc(&this->d_delays_vector,sizeof(int)*n);
    cudaMalloc(&this->d_rule_index,    sizeof(int)*(n+1));
    cudaMalloc(&this->d_rules.Ei,      sizeof(int)*m);
    cudaMalloc(&this->d_rules.En,      sizeof(int)*m);
    cudaMalloc(&this->d_rules.c,       sizeof(int)*m);
    cudaMalloc(&this->d_rules.p,       sizeof(int)*m);
    cudaMalloc(&this->d_rules.d,       sizeof(uint)*m);
    cudaMalloc(&this->d_rules.nid,     sizeof(uint)*m);
    cudaMalloc(&this->d_calc_next_trans,     sizeof(bool));

    
   
    // memory consistency, who has the updated copy?
    gpu_updated = false; cpu_updated = true;
    done_rules = false;
}

/** Free mem */
SNP_model::~SNP_model()
{
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize (stop) );

    CHECK_CUDA(cudaEventElapsedTime(&this->elapsed, start, stop) );

    CHECK_CUDA(cudaEventDestroy(this->start));
    CHECK_CUDA(cudaEventDestroy(this->stop));

    printf("The elapsed time in gpu was %.2f ms\n", elapsed);
    
    //MEMORY USAGE
    size_t free_byte ;

    size_t total_byte ;
    cudaError_t cuda_status;

    cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != cuda_status ){

        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );

        exit(1);

    }

    double free_db = (double)free_byte ;

    double total_db = (double)total_byte ;

    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",

        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    ////////////////////////////////////



    cudaFree(this->d_conf_vector_cpy);
    if(ex_mode ==GPU_CUBLAS || ex_mode ==GPU_CUSPARSE){
        free(this->cublas_conf_vector);
        cudaFree(this->d_cublas_conf_vector);
    }else{
        
        cudaFree(this->d_conf_vector);
    }
    cudaFreeHost(this->conf_vector);
    
    // free(this->spiking_vector);
    // if (this->trans_matrix) free(this->trans_matrix);
    free(this->delays_vector);
    free(this->rule_index);
    free(this->rules.Ei);
    free(this->rules.En);
    free(this->rules.c);
    free(this->rules.p);
    free(this->rules.d);
    free(this->rules.nid);
    if(this->verbosity>=2){
        cudaFreeHost(this->calc_next_trans);
    }else{
        cudaFree(this->calc_next_trans);
    }
    
    

    
    // cudaFree(this->d_spiking_vector);
    // if (this->d_trans_matrix) cudaFree(this->d_trans_matrix);
    cudaFree(this->d_delays_vector);
    cudaFree(this->d_rule_index);
    cudaFree(this->d_rules.Ei);
    cudaFree(this->d_rules.En);
    cudaFree(this->d_rules.c);
    cudaFree(this->d_rules.p);
    cudaFree(this->d_rules.d);
    cudaFree(this->d_rules.nid);
    cudaFree(this->d_calc_next_trans);
    
    if(this->stream1){
        cudaStreamDestroy(this->stream1);
    }

    if(this->stream2){
        cudaStreamDestroy(this->stream2);
    }

    if(this->csv){
        this->csv.close();
    }
    

    cudaProfilerStop();
}

void SNP_model::set_spikes (uint nid, uint s)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) {
        load_to_cpu(NULL);
        cpu_updated=true;
    }
    gpu_updated = false;
    //////////////////////////////////////////////////////

    if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
        cublas_conf_vector[nid] = s; 
    }else{
        conf_vector[nid] = s; 
    }
       
}

uint SNP_model::get_spikes (uint nid)
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (gpu_updated && !cpu_updated) {
        load_to_cpu(NULL);
        cpu_updated=true;
    }
    //////////////////////////////////////////////////////
    if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
        return cublas_conf_vector[nid]; 
    }else{
        return conf_vector[nid];  
    }
    
}

/** Add a rule to neuron nid, regular expression defined by e_n and e_i, and a^c -> a^p.
    Must be called sorted by neuron */
void SNP_model::add_rule (uint nid, int e_n, int e_i, int c, int p, uint d) 
{
    //////////////////////////////////////////////////////
    assert(nid < n);
    assert(!done_rules);
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    if(rule_index[nid+1]==-1){
        rule_index[nid+1] = rule_index[nid] + 1;
    }else{
        rule_index[nid+1] = rule_index[nid+1] + 1;
    }
    
    

 
    uint rid = rule_index[nid+1]-1;
    // printf("rule_index[%d]=%d, rule_index[%d]=%d",nid,rule_index[nid], nid+1,rule_index[nid+1]);

    rules.Ei[rid] = e_i;
    rules.En[rid] = e_n;
    rules.c[rid]  = c;
    rules.p[rid]  = p;
    rules.d[rid] = d;
    // if(rid==4){
    //     printf("rules.d[4]=%d",rules.d[4]);
    // }
    rules.nid[rid]= nid;

}

/** Add synapse from neuron i to j. 
    Must be called after adding all rules */
void SNP_model::add_synapse (uint i, uint j) 
{
    //////////////////////////////////////////////////////
    // ensure parameters within limits
    assert(i < n && j < n);
    // ensure all rules have been introduced already
    
    // assert(rule_index[n]==m); //TODO: What if neuron n does not contain rules. Sometimes rule_index ends before index n.
    // SNP does not allow self-synapses
    assert(i!=j);
    done_rules = true; // from now on, no more rules can be added
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    gpu_updated = false; cpu_updated = true;
    //////////////////////////////////////////////////////

    include_synapse(i,j);
}

void SNP_model::calc_z(){
    assert(trans_matrix != NULL);
    z=0;
    for(int i=0; i<m;i++){
        int z_aux =0;
        for(int j=0; j<n;j++){
            if(trans_matrix[i*n+j]>0){
                z_aux++;
            }
        }
        if(z_aux>z){
            z=z_aux;
        }

    }
}




void SNP_model::printDelaysV(){
    printf("delays_vector= "); 
    for(int i=0; i<n; i++){
        printf("{%d}",delays_vector[i]);
    }
    printf("\n");
}

void SNP_model::printConfV(){
    if(this->csv){
        this->csv << "Step " + to_string(this->step) + "\n";
    }
    
    printf("conf_vector= ");
    if(this->step==0 && (ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE)){
        for(int i=0; i< n; i++){
            printf("{%.0f}",cublas_conf_vector[i]);
            if(this->csv){
                csv<< std::to_string(cublas_conf_vector[i]) + ",";
            }
            
        }
        if(this->csv){
            csv<<"\n";
        }
    }else{
        for(int i=0; i< n; i++){
            printf("{%d}",conf_vector[i]);

            if(this->csv){
                csv<< std::to_string(conf_vector[i]) + ",";
            }
            
        }
        if(this->csv){
            csv<<"\n";
        }   

    }
    
    printf("\n");
}


__global__ void does_it_calc_nxt(bool * calc_nxt, int* spkv, int spkv_size, int * delays, int neurons, int ex_mode, int verbosity){
    calc_nxt[0] = false;
    
    for(int i=0; i<spkv_size; i++){
        if((ex_mode != GPU_OPTIMIZED && spkv[i] !=0) || ex_mode == GPU_OPTIMIZED && spkv[i] !=-1){
            calc_nxt[0] = true;
            break;
            
        }
    }
    
    
    if(!calc_nxt[0] || verbosity>=3){
        
        for(int i=0; i<neurons; i++){
            if(delays[i] >0){
                calc_nxt[0] = true;
                break;
            }
    
        }
    }

}

__global__ void printVectorsK(float* spkv, float* spkv_aux, int spkv_size, int * delays, int neurons, int verbosity){

    
        printf("Spiking_vector:");
        for(int i=0; i<spkv_size; i++){
            printf("%.1f ",spkv[i]);
            
        }
        printf("\n");
    
        printf("Spiking_vector_aux= ");
        for(int i=0; i<spkv_size; i++){
            printf("%.1f ",spkv_aux[i]);
            
        }
        printf("\n");
        printf("Delays_v:");
        for(int i=0; i<neurons; i++){
            printf("%d ",delays[i]);
        }
        printf("\n");
        

    

}

__global__ void does_it_calc_nxt_cu(bool * calc_nxt, float* spkv, float* spkv_aux, int spkv_size, int * delays, int neurons, int verbosity){
    calc_nxt[0] = false;
    
    for(int i=0; i<spkv_size; i++){
        if(spkv[i] !=0){
            calc_nxt[0] = true;
            break;
       
        }
    }

    if(!calc_nxt[0] || verbosity>=3){
        
        for(int i=0; i<neurons; i++){
            
            if(delays[i] >0){
                calc_nxt[0] = true;
                break;
                
                
            }
    
        }
           
    }

    
    
}

//TODO: las dos funciones siguientes son iguales, solo cambia un argumento float por int. Unificar???

__global__ void cpy_conf_vector(float * conf_v, int *conf_v_cpy, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<n){
        conf_v_cpy[idx]= (int) conf_v[idx];
        // printf("%d",conf_v_cpy[idx]);
    }
}

__global__ void cpy_conf_vector(int * conf_v, int *conf_v_cpy, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx<n){
        conf_v_cpy[idx]= (int) conf_v[idx];
        // printf("%d",conf_v_cpy[idx]);
    }
}

bool SNP_model::transition_step()
{
   //returns true if stop criterion reached; false otherwise
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (!gpu_updated) load_to_gpu();
    cpu_updated = false;
    //////////////////////////////////////////////////////
    
    // if(step==0 && verbosity>=2){
    //     printf("Initial conf_vector:");
    //     for(int nid=0; nid<n; nid++){
    //         if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
    //             printf("%.1f ", cublas_conf_vector[nid]);

    //         }else{
    //             printf("%d ", conf_vector[nid]);
    //         }
            
    //     }
    //     printf("\n");
    // }

    if(this->step>=1  && verbosity>=2){
        cudaStreamSynchronize(this->stream2);
        if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
            cpy_conf_vector<<<n+255,256,0,this->stream1>>>(d_cublas_conf_vector, d_conf_vector_cpy, n);

        }else{
            cpy_conf_vector<<<n+255,256,0,this->stream1>>>(d_conf_vector, d_conf_vector_cpy, n); //makes a copy of the conf_vector which will be sent to host while, at the same time, computation is performed.

        }
        cudaStreamSynchronize(this->stream1); //finish copy before transition is made
        load_to_cpu(this->stream1); 
 
    }

    
    calc_spiking_vector(); 
    int spv_size= ex_mode == GPU_OPTIMIZED ? n : m;
    if(ex_mode == GPU_CUBLAS || ex_mode ==GPU_CUSPARSE){
        if(this->verbosity>=2){
            does_it_calc_nxt_cu<<<1,1,0,this->stream2>>>(d_calc_next_trans, d_cublas_spiking_vector, d_cublas_spiking_vector_aux,spv_size, d_delays_vector, n, verbosity);
        }else{
            does_it_calc_nxt_cu<<<1,1>>>(d_calc_next_trans, d_cublas_spiking_vector, d_cublas_spiking_vector_aux,spv_size, d_delays_vector, n, verbosity);

        }
    }else{
        if(this->verbosity>=2){
            does_it_calc_nxt<<<1,1,0,this->stream2>>>(d_calc_next_trans, d_spiking_vector, spv_size, d_delays_vector, n, ex_mode, verbosity);

        }else{
            does_it_calc_nxt<<<1,1>>>(d_calc_next_trans, d_spiking_vector, spv_size, d_delays_vector, n, ex_mode, verbosity);

        }
    }
    
    if(this->verbosity>=2){
        CHECK_CUDA(cudaMemcpyAsync(this->calc_next_trans, this->d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost,this->stream2));
    
        cudaStreamSynchronize(this->stream2);
    }else{
        cudaMemcpy(this->calc_next_trans, this->d_calc_next_trans, sizeof(bool),cudaMemcpyDeviceToHost); /////////
    }
    

    if(this->calc_next_trans[0]){
        //wait until cpy_conf_vector_cusparse has finished
        calc_transition();
        
        if(this->verbosity>=2){
            cudaStreamSynchronize(this->stream1);
            printConfV();
            printf("\n---------------------------------------\n");
        }

        if(ex_mode == GPU_CUBLAS || ex_mode ==GPU_CUSPARSE){
            if(verbosity>=3){
                printVectorsK<<<1,1>>>(d_cublas_spiking_vector, d_cublas_spiking_vector_aux,spv_size, d_delays_vector, n, verbosity);

            }
        }//else, print is located in static_*.cu


        if((verbosity>=3 && (ex_mode==GPU_CUBLAS || ex_mode == GPU_CUSPARSE)) || (verbosity>=3 && !transMX_printed)){
            
            printf("Trans_MX:\n");
            printTransMX();
            transMX_printed = true;
        }
        

        if(ex_mode == GPU_CUBLAS || ex_mode == GPU_CUSPARSE){
            load_transition_matrix(); //stream 2
        }

        this->step++;

        return false;
    }

    if(this->verbosity==0 || this->verbosity==1){
        load_to_cpu(NULL); 
        
        
    }
    if(this->verbosity>=1){
        if(this->verbosity>=2){
            cudaStreamSynchronize(this->stream1);
        }
        
        printConfV();
        printf("\n---------------------------------------\n");
        
        
    }
    
    //stop criterion if no rule found active and all neurons are open 
    return true;
}

void SNP_model::load_to_gpu () 
{
    if(this->step==0){
        //////////////////////////////////////////////////////
        // check memory consistency, who has the updated copy?
        assert(gpu_updated || cpu_updated);
        if (gpu_updated) return;
        gpu_updated = true;
        //////////////////////////////////////////////////////

        // cublasGetVector (n , sizeof (* conf_vector ) , d_conf_vector ,1 ,conf_vector ,1);
        if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
            cudaMemcpy(d_cublas_conf_vector,   cublas_conf_vector,    sizeof(float)*n,   cudaMemcpyHostToDevice);
        }else{
            cudaMemcpy(d_conf_vector,   conf_vector,    sizeof(int)*n,   cudaMemcpyHostToDevice);
        }
        
        // cudaMemcpy(d_spiking_vector,spiking_vector, sizeof(ushort)*m,   cudaMemcpyHostToDevice);
        cudaMemcpy(d_rule_index,    rule_index,     sizeof(int)*(n+1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.Ei,      rules.Ei,       sizeof(int)*m,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.En,      rules.En,       sizeof(int)*m,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.c,       rules.c,        sizeof(int)*m,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.p,       rules.p,        sizeof(int)*m,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.d,       rules.d,        sizeof(uint)*m,    cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules.nid,     rules.nid,      sizeof(uint)*m,     cudaMemcpyHostToDevice);
        
        load_transition_matrix();

    }
    
    
    
}

void SNP_model::load_to_cpu (cudaStream_t stream)
{
    //////////////////////////////////////////////////////
    // check memory consistency, who has the updated copy?
    assert(gpu_updated || cpu_updated);
    if (cpu_updated) return;
    cpu_updated = true;
    //////////////////////////////////////////////////////
   
    if (this->step>=1){

        if(stream){
            cudaMemcpyAsync(conf_vector, d_conf_vector_cpy, sizeof(int)*n, cudaMemcpyDeviceToHost, stream);
    
        }else{
        //if stream is NULL
            if(ex_mode==GPU_CUBLAS || ex_mode==GPU_CUSPARSE){
                cudaMemcpy(cublas_conf_vector, d_cublas_conf_vector, sizeof(float)*n, cudaMemcpyDeviceToHost);
                
            }else{
                cudaMemcpy(conf_vector, d_conf_vector, sizeof(int)*n, cudaMemcpyDeviceToHost);
            }
    
        }

    }
    
    
    
    
}





